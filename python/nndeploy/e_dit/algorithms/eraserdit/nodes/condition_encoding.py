"""EraserDiT erase – condition encoding stage.

Port of the first half of ``LTXVideoToVideoPipeline.prepare_latents``: the VAE
encode of the masked window video.  Two details differ from the EraserDiT path and
are load-bearing:

* The baseline keeps the VAE's ``latents_mean`` / ``latents_std`` normalisation
  with ``scaling_factor=1.0`` (the pipeline's own default), *not*
  ``vae.config.scaling_factor``.
* ``latent_dist.sample(generator)`` is used, **not** ``mode()`` -- the encode is
  a random draw and consumes the request generator.  The per-window draw order
  is encode sample -> init noise -> decode noise.
"""

from __future__ import annotations

from nndeploy.e_dit.nodes.base import operation

import torch

from nndeploy.e_dit.config.server_args import ServerArgs
from nndeploy.e_dit.algorithms.eraserdit.optimizations.component_offload import offload_component
from nndeploy.e_dit.algorithms.eraserdit.optimizations.memory_phase_controller import MemoryPhase
from nndeploy.e_dit.nodes.schedule_batch import Req
from nndeploy.e_dit.nodes.base import StateNode
from nndeploy.e_dit.algorithms.eraserdit.nodes._common import (
    field_summary,
    get_task_state,
    normalize_latents,
)
from nndeploy.e_dit.memory.tensor_ops import module_device, module_dtype


class EraserDiTEraseConditionEncodingNode(StateNode):
    @operation
    @offload_component("vae", phase=MemoryPhase.VAE_ENCODE)
    def run(self, batch: Req, server_args: ServerArgs) -> Req:
        vae = batch.modules["vae"]
        state = get_task_state(batch)

        # ``video_processor.preprocess`` normalises [0, 1] to [-1, 1].
        vae_input = (batch.padded_video * 2 - 1).to(
            device=module_device(vae), dtype=module_dtype(vae)
        )
        # Downstream stages use frame metadata, not these full-resolution pixels.
        # Drop both aliases before VAE activations are allocated.
        batch.padded_video = None
        batch.masked_video = None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            from nndeploy.e_dit.algorithms.eraserdit.parallel.vae import tiled_vae
            posterior = tiled_vae(vae, vae_input, server_args, batch, operation="encode")
            cond_latents = posterior.sample(state.generator)
        del vae_input, posterior
        cond_latents = cond_latents.to(torch.float32)
        batch.cond_latents = normalize_latents(
            cond_latents,
            vae.latents_mean,
            vae.latents_std,
            scaling_factor=1.0,
        )

        # Baseline: ``interpolate(masks, [h, w])`` with the default nearest mode,
        # then ``rearrange("b f c h w -> b c f h w")``.
        latent_height = int(batch.cond_latents.shape[-2])
        latent_width = int(batch.cond_latents.shape[-1])
        mask_latents = batch.padded_mask[0].permute(1, 0, 2, 3).to(torch.float32)
        mask_values = torch.nn.functional.interpolate(
            mask_latents, [latent_height, latent_width]
        )
        batch.mask_values = (
            mask_values.unsqueeze(0)
            .permute(0, 2, 1, 3, 4)
            .to(device=batch.cond_latents.device, dtype=batch.cond_latents.dtype)
        )
        batch.latent_shape = tuple(batch.cond_latents.shape)
        batch.padded_mask = None
        self.log_info(
            "%s | %s",
            field_summary("cond_latents", batch.cond_latents),
            field_summary("mask_values", batch.mask_values),
        )
        return batch
