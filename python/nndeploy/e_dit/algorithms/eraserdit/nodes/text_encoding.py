"""EraserDiT erase – text encoding stage.

Port of ``LTXVideoToVideoPipeline._get_t5_prompt_embeds`` / ``encode_prompt``.
The positive and negative embeddings are kept separate (the baseline concatenates
them with the negative half first and then slices ``[0:1]`` / ``[1:]``; the
denoising stage performs the same split explicitly).
"""

from __future__ import annotations

from nndeploy.e_dit.nodes.base import operation

import torch

from nndeploy.e_dit.config.server_args import ServerArgs
from nndeploy.e_dit.algorithms.eraserdit.optimizations.component_offload import offload_component
from nndeploy.e_dit.nodes.schedule_batch import Req
from nndeploy.e_dit.nodes.base import StateNode
from nndeploy.e_dit.algorithms.eraserdit.nodes._common import field_summary, get_task_state
from nndeploy.e_dit.utils.logging_utils import init_logger
from nndeploy.e_dit.memory.tensor_ops import module_device

logger = init_logger(__name__)


class EraserDiTEraseTextEncodingNode(StateNode):
    def setup(self, text_encoder, tokenizer):
        self._text_encoder = text_encoder
        self._tokenizer = tokenizer

    def _t5_prompt_embeds(
        self,
        prompt: str,
        *,
        max_sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokenizer = self._tokenizer
        text_inputs = tokenizer(
            [prompt],
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.bool().to(device)

        untruncated_ids = tokenizer([prompt], padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(
                untruncated_ids[:, max_sequence_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` "
                "is set to %s tokens: %s",
                max_sequence_length,
                removed_text,
            )

        prompt_embeds = self._text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds, prompt_attention_mask.view(1, -1)

    @staticmethod
    def _install_cached_embeddings(
        batch: Req,
        values,
        *,
        device: torch.device,
        source: str,
    ) -> Req:
        (
            batch.prompt_embeds,
            batch.prompt_attention_mask,
            batch.negative_prompt_embeds,
            batch.negative_attention_mask,
        ) = (value.to(device, copy=True) for value in values)
        batch.extra["text_encoding_cache_source"] = source
        if batch.metrics is not None:
            batch.metrics.record_operation("text_encoding_cache_hit")
        return batch

    def _restore_cached_embeddings(
        self,
        batch: Req,
        *,
        signature,
        device: torch.device,
    ) -> Req | None:
        # The window runtime owns this cache and keys it by object, scene and
        # both prompts.  Check it before entering the text-encoder memory phase,
        # so later windows do not acquire or execute T5 at all.
        runtime_cached = batch.extra.get("cached_text_embeddings")
        required = (
            "prompt_embeds",
            "prompt_attention_mask",
            "negative_prompt_embeds",
            "negative_attention_mask",
        )
        if isinstance(runtime_cached, dict) and all(
            isinstance(runtime_cached.get(name), torch.Tensor) for name in required
        ):
            return self._install_cached_embeddings(
                batch,
                tuple(runtime_cached[name] for name in required),
                device=device,
                source="window_runtime",
            )

        # Keep the request-owned cache for direct stage execution and for the
        # interval before the window driver has committed its first cache entry.
        state = get_task_state(batch)
        cached = state.extra.get("text_encoding")
        if cached is not None and cached[0] == signature:
            return self._install_cached_embeddings(
                batch,
                cached[1],
                device=device,
                source="request",
            )
        return None

    @operation
    def run(self, batch: Req, server_args: ServerArgs) -> Req:
        text_encoder = self._text_encoder
        device = module_device(text_encoder)
        dtype = text_encoder.dtype
        max_sequence_length = int(batch.max_sequence_length or 128)

        prompt = batch.prompt or ""
        negative_prompt = batch.negative_prompt or ""
        signature = (prompt, negative_prompt, max_sequence_length, dtype,
                     id(text_encoder), id(self._tokenizer))
        restored = self._restore_cached_embeddings(
            batch,
            signature=signature,
            device=device,
        )
        if restored is not None:
            batch.max_sequence_length = max_sequence_length
            return restored

        return self._encode_uncached(
            batch,
            server_args,
            prompt=prompt,
            negative_prompt=negative_prompt,
            max_sequence_length=max_sequence_length,
            signature=signature,
        )

    @offload_component("text_encoder")
    def _encode_uncached(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        prompt: str,
        negative_prompt: str,
        max_sequence_length: int,
        signature,
    ) -> Req:
        del server_args
        text_encoder = self._text_encoder
        device = module_device(text_encoder)
        dtype = text_encoder.dtype
        state = get_task_state(batch)

        # The baseline calls ``encode_prompt`` inside ``autocast(bf16)``; the T5
        # layer norms are in autocast's fp32 category, so this changes their
        # outputs and must be reproduced.
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            prompt_embeds, prompt_attention_mask = self._t5_prompt_embeds(
                prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
            negative_prompt_embeds, negative_prompt_attention_mask = self._t5_prompt_embeds(
                negative_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        batch.prompt_embeds = prompt_embeds
        batch.prompt_attention_mask = prompt_attention_mask
        batch.negative_prompt_embeds = negative_prompt_embeds
        batch.negative_attention_mask = negative_prompt_attention_mask
        batch.max_sequence_length = max_sequence_length
        batch.extra["text_encoding_cache_source"] = "computed"
        # Request-owned, bounded to one prompt pair. CPU storage avoids keeping
        # encoder outputs on the GPU during subsequent windows' preprocessing.
        state.extra["text_encoding"] = (signature, tuple(
            value.detach().cpu().clone() for value in (
                prompt_embeds, prompt_attention_mask,
                negative_prompt_embeds, negative_prompt_attention_mask,
            )
        ))
        self.log_info(
            "%s | %s",
            field_summary("prompt_embeds", prompt_embeds),
            field_summary("negative_prompt_embeds", negative_prompt_embeds),
        )
        return batch
