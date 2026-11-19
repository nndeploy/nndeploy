# Copyright 2024 The Genmo team and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import math
import json

from typing import Any, Dict, Optional, Tuple, Union

from nndeploy.e_dit.memory.backends.layerwise_offload import OffloadableDiTMixin
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import PixArtAlphaTextProjection
from diffusers.models.modeling_outputs import Transformer2DModelOutput
# from diffusers.models.modeling_utils import ModelMixin
from nndeploy.e_dit.models.modeling_utils import ModelMixin
from nndeploy.e_dit.algorithms.eraserdit.models.block import forward_eraserdit_block
from nndeploy.e_dit.algorithms.eraserdit.models.attention import (
    EraserDiTAttentionProcessor,
    apply_rotary_emb,
)
from diffusers.models.normalization import AdaLayerNormSingle, RMSNorm


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def normalize_latents(
    latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
) -> torch.Tensor:
    # Normalize latents across the channel dimension [B, C, F, H, W]
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents = (latents - latents_mean) * scaling_factor / latents_std
    return latents


class LTXVideoAttentionProcessor2_0:  # noqa: N801 - reference implementation
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the LTX model. It applies a normalization layer and rotary embedding on the query and key vector.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "LTXVideoAttentionProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class LTXVideoRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        dim: int,
        base_num_frames: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        patch_size: int = 1,
        patch_size_t: int = 1,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.base_num_frames = base_num_frames
        self.base_height = base_height
        self.base_width = base_width
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.theta = theta

    def _prepare_video_coords(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        rope_interpolation_scale: Tuple[torch.Tensor, float, float],
        device: torch.device,
    ) -> torch.Tensor:
        # Always compute rope in fp32
        grid_h = torch.arange(height, dtype=torch.float32, device=device)
        grid_w = torch.arange(width, dtype=torch.float32, device=device)
        grid_f = torch.arange(num_frames, dtype=torch.float32, device=device)
        grid = torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij")
        grid = torch.stack(grid, dim=0)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)

        if rope_interpolation_scale is not None:
            grid[:, 0:1] = grid[:, 0:1] * rope_interpolation_scale[0] * self.patch_size_t / self.base_num_frames
            grid[:, 1:2] = grid[:, 1:2] * rope_interpolation_scale[1] * self.patch_size / self.base_height
            grid[:, 2:3] = grid[:, 2:3] * rope_interpolation_scale[2] * self.patch_size / self.base_width

        grid = grid.flatten(2, 4).transpose(1, 2)

        return grid

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        rope_interpolation_scale: Optional[Tuple[torch.Tensor, float, float]] = None,
        video_coords: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.size(0)

        if video_coords is None:
            grid = self._prepare_video_coords(
                batch_size,
                num_frames,
                height,
                width,
                rope_interpolation_scale=rope_interpolation_scale,
                device=hidden_states.device,
            )
        else:
            grid = torch.stack(
                [
                    video_coords[:, 0] / self.base_num_frames,
                    video_coords[:, 1] / self.base_height,
                    video_coords[:, 2] / self.base_width,
                ],
                dim=-1,
            )

        start = 1.0
        end = self.theta
        freqs = self.theta ** torch.linspace(
            math.log(start, self.theta),
            math.log(end, self.theta),
            self.dim // 6,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        freqs = freqs * math.pi / 2.0
        freqs = freqs * (grid.unsqueeze(-1) * 2 - 1)
        freqs = freqs.transpose(-1, -2).flatten(2)

        cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

        if self.dim % 6 != 0:
            cos_padding = torch.ones_like(cos_freqs[:, :, : self.dim % 6])
            sin_padding = torch.zeros_like(cos_freqs[:, :, : self.dim % 6])
            cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
            sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)

        return cos_freqs, sin_freqs


@maybe_allow_in_graph
class LTXVideoTransformerBlock(nn.Module):
    r"""
    Transformer block used in [LTX](https://huggingface.co/Lightricks/LTX-Video).

    Args:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        qk_norm (`str`, defaults to `"rms_norm"`):
            The normalization layer to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        qk_norm: str = "rms_norm_across_heads",
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            cross_attention_dim=None,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            processor=EraserDiTAttentionProcessor(),
        )

        self.norm2 = RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            processor=EraserDiTAttentionProcessor(),
        )

        self.ff = FeedForward(dim, activation_fn=activation_fn)

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        *,
        decision=None,
        text_cache=None,
        cache_probe_only=False,
    ) -> torch.Tensor:
        if cache_probe_only:
            # Invoke through block.forward so dynamic offload makes norm/table
            # weights resident before reading them. No attention/MLP is run.
            norm_hidden_states = self.norm1(hidden_states)
            ada_values = self.scale_shift_table[None, None] + temb.reshape(
                hidden_states.size(0), temb.size(1), self.scale_shift_table.shape[0], -1
            )
            shift_msa, scale_msa = ada_values[:, :, 0], ada_values[:, :, 1]
            return norm_hidden_states * (1 + scale_msa) + shift_msa
        if decision is not None:
            return forward_eraserdit_block(
                self, hidden_states, encoder_hidden_states, temb,
                image_rotary_emb, encoder_attention_mask, decision=decision, text_cache=text_cache,
            )
        batch_size = hidden_states.size(0)
        norm_hidden_states = self.norm1(hidden_states)

        num_ada_params = self.scale_shift_table.shape[0]
        ada_values = self.scale_shift_table[None, None] + temb.reshape(batch_size, temb.size(1), num_ada_params, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        attn_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=None,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa

        attn_hidden_states = self.attn2(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=None,
            attention_mask=encoder_attention_mask,
            text_cache=text_cache,
        )
        hidden_states = hidden_states + attn_hidden_states
        norm_hidden_states = self.norm2(hidden_states) * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp

        return hidden_states


@maybe_allow_in_graph
class EraserDiTLTXVideoTransformer3DModel(OffloadableDiTMixin, ModelMixin, ConfigMixin, FromOriginalModelMixin, PeftAdapterMixin):
    r"""
    A Transformer model for video-like data used in [LTX](https://huggingface.co/Lightricks/LTX-Video).

    Args:
        in_channels (`int`, defaults to `128`):
            The number of channels in the input.
        out_channels (`int`, defaults to `128`):
            The number of channels in the output.
        patch_size (`int`, defaults to `1`):
            The size of the spatial patches to use in the patch embedding layer.
        patch_size_t (`int`, defaults to `1`):
            The size of the tmeporal patches to use in the patch embedding layer.
        num_attention_heads (`int`, defaults to `32`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        cross_attention_dim (`int`, defaults to `2048 `):
            The number of channels for cross attention heads.
        num_layers (`int`, defaults to `28`):
            The number of layers of Transformer blocks to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        qk_norm (`str`, defaults to `"rms_norm_across_heads"`):
            The normalization layer to use.
    """

    _supports_gradient_checkpointing = True
    layer_names = ["transformer_blocks"]
    _skip_layerwise_casting_patterns = ["norm"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 128,
        patch_size: int = 1,
        patch_size_t: int = 1,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        cross_attention_dim: int = 2048,
        num_layers: int = 28,
        activation_fn: str = "gelu-approximate",
        qk_norm: str = "rms_norm_across_heads",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        caption_channels: int = 4096,
        attention_bias: bool = True,
        attention_out_bias: bool = True,
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels
        inner_dim = num_attention_heads * attention_head_dim

        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.time_embed = AdaLayerNormSingle(inner_dim, use_additional_conditions=False)

        self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)

        self.rope = LTXVideoRotaryPosEmbed(
            dim=inner_dim,
            base_num_frames=20,
            base_height=2048,
            base_width=2048,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            theta=10000.0,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                LTXVideoTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    qk_norm=qk_norm,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    attention_out_bias=attention_out_bias,
                    eps=norm_eps,
                    elementwise_affine=norm_elementwise_affine,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = nn.LayerNorm(inner_dim, eps=1e-6, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels)

        self.gradient_checkpointing = False
        # Resolved once at construction: the fusion decision is fixed for the
        # lifetime of the resident process (plan §M3).
        from nndeploy.e_dit.algorithms.eraserdit.optimizations.fusion_policy import get_operator_fusion_decision
        from nndeploy.e_dit.config.server_args import get_global_server_args

        self.operator_fusion_decision = get_operator_fusion_decision(
            get_global_server_args()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        rope_interpolation_scale: Optional[Union[Tuple[float, float, float], torch.Tensor]] = None,
        video_coords: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        cond_latents=None,
        mask_values=None,
        cache_adapter=None,
        text_cache=None,
        sequence_parallel=None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if (cache_adapter is not None and self.layerwise_offload_managers
                and cache_adapter.controller.mode.value != 'teacache'):
            raise ValueError("Only TeaCache supports planned layerwise offload")
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        if image_rotary_emb is None:
            image_rotary_emb = self.rope(hidden_states, num_frames, height, width, rope_interpolation_scale, video_coords)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        batch_size = hidden_states.size(0)
        if mask_values is not None:
            hidden_states = pack_latents(
                torch.cat([hidden_states, cond_latents, mask_values], dim=1), self.config.patch_size, self.config.patch_size_t
            )
        else:
            hidden_states = pack_latents(
                torch.cat([hidden_states, cond_latents], dim=1), self.config.patch_size, self.config.patch_size_t
            )

        # Keep the narrow input projection at the original token shape. BF16
        # GEMM kernel selection at this projection is sensitive to M; a single
        # rounding difference can be amplified by the full denoising chain.
        hidden_states = self.proj_in(hidden_states)
        shard = slice(None)
        if sequence_parallel is not None:
            if cache_adapter is not None:
                cache_adapter.global_sequence_length = hidden_states.shape[1]
            shard = sequence_parallel.partition(hidden_states.shape[1])
            hidden_states = hidden_states[:, shard]
            image_rotary_emb = tuple(t[:, shard] for t in image_rotary_emb)

        if cache_adapter is not None:
            cache_adapter.configure_probe_regions(mask_values, shape=(num_frames, height, width), shard=shard,
                patch_size=self.config.patch_size, patch_size_t=self.config.patch_size_t)

        temb, embedded_timestep = self.time_embed(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )

        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        encoder_hidden_states = (self.caption_projection(encoder_hidden_states) if text_cache is None
                                 else text_cache.project(encoder_hidden_states, self.caption_projection))
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

        def run_blocks_impl(hidden_states, start, end):
            for block in self.transformer_blocks[start:end]:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states = self._gradient_checkpointing_func(
                        block, hidden_states, encoder_hidden_states, temb,
                        image_rotary_emb, encoder_attention_mask,
                    )
                else:
                    # Always enter Module.__call__: offload hooks must run
                    # before the internal fused implementation reads weights.
                    # Disabled-fusion telemetry uses ContextVar; the native
                    # block arithmetic is equivalent and can be captured whole.
                    hidden_states = block(
                        hidden_states, encoder_hidden_states, temb,
                        image_rotary_emb, encoder_attention_mask,
                        decision=(None if torch.compiler.is_compiling()
                                  and not self.operator_fusion_decision.effective_ops
                                  else self.operator_fusion_decision), text_cache=text_cache,
                    )
            return hidden_states

        def run_blocks(hidden_states, start, end):
            if cache_adapter is None:
                return run_blocks_impl(hidden_states, start, end)
            with self.layerwise_execution_plan(range(start, end)):
                return run_blocks_impl(hidden_states, start, end)

        if cache_adapter is not None:
            if torch.is_grad_enabled():
                raise RuntimeError("EraserDiT caching is inference-only")
            cache_probe = temb
            if cache_adapter.controller.mode.value == 'teacache':
                first = self.transformer_blocks[0]
                with self.layerwise_execution_plan((0,)):
                    cache_probe = first(
                        hidden_states, encoder_hidden_states, temb, cache_probe_only=True,
                    )
            hidden_states = cache_adapter.run(
                hidden_states, cache_probe, run_blocks,
                num_blocks=len(self.transformer_blocks),
                layout=(num_frames, height, width, repr(rope_interpolation_scale)),
            )
        else:
            hidden_states = run_blocks(hidden_states, 0, len(self.transformer_blocks))

        scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale) + shift
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if sequence_parallel is None:
            output = unpack_latents(output, num_frames, height, width, self.config.patch_size, self.config.patch_size_t)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
    
    @classmethod
    def from_pretrained_3d(
        cls, pretrained_model_path, subfolder=None, transformer_additional_kwargs={},
        torch_dtype=torch.bfloat16
    ):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)
        print(f"loaded 3D transformer's pretrained weights from {pretrained_model_path} ...")

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)
        
        from diffusers.utils import WEIGHTS_NAME
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        model_file_safetensors = model_file.replace(".bin", ".bf16.safetensors")

        model = cls.from_config(config, **transformer_additional_kwargs)
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
        elif os.path.exists(model_file_safetensors):
            from safetensors.torch import load_file
            state_dict = load_file(model_file_safetensors)
        else:
            from safetensors.torch import load_file
            model_file_safetensors = glob.glob(os.path.join(pretrained_model_path, "*.safetensors"))
            state_dict = {}
            for _model_file_safetensors in model_file_safetensors:
                _state_dict = load_file(_model_file_safetensors)
                for key in _state_dict:
                    state_dict[key] = _state_dict[key]
        
        if model.state_dict()['proj_in.weight'].size() != state_dict['proj_in.weight'].size():
            new_shape = model.state_dict()['proj_in.weight'].size()
            if len(new_shape) == 2:
                if model.state_dict()['proj_in.weight'].size()[1] > state_dict['proj_in.weight'].size()[1]:
                    model.state_dict()['proj_in.weight'][:, :state_dict['proj_in.weight'].size()[1]] = state_dict['proj_in.weight']
                    model.state_dict()['proj_in.weight'][:, state_dict['proj_in.weight'].size()[1]:] = 0
                    state_dict['proj_in.weight'] = model.state_dict()['proj_in.weight']
                else:
                    model.state_dict()['proj_in.weight'][:, :] = state_dict["proj_in.weight"][:, :model.state_dict()['proj_in.weight'].size()[1]]
                    state_dict['proj_in.weight'] = model.state_dict()['proj_in.weight']
        

        if model.state_dict()['caption_projection.linear_1.weight'].size() != state_dict['caption_projection.linear_1.weight'].size():
            new_shape = model.state_dict()['caption_projection.linear_1.weight'].size()
            if len(new_shape) == 2:
                if model.state_dict()['caption_projection.linear_1.weight'].size()[1] > state_dict['caption_projection.linear_1.weight'].size()[1]:
                    model.state_dict()['caption_projection.linear_1.weight'][:, :state_dict['caption_projection.linear_1.weight'].size()[1]] = state_dict['caption_projection.linear_1.weight']
                    model.state_dict()['caption_projection.linear_1.weight'][:, state_dict['caption_projection.linear_1.weight'].size()[1]:] = 0
                    state_dict['caption_projection.linear_1.weight'] = model.state_dict()['caption_projection.linear_1.weight']
                else:
                    model.state_dict()['caption_projection.linear_1.weight'][:, :] = state_dict["caption_projection.linear_1.weight"][:, :model.state_dict()['caption_projection.linear_1.weight'].size()[1]]
                    state_dict['caption_projection.linear_1.weight'] = model.state_dict()['caption_projection.linear_1.weight']

        tmp_state_dict = {} 
        for key in state_dict:
            if key in model.state_dict().keys() and model.state_dict()[key].size() == state_dict[key].size():
                tmp_state_dict[key] = state_dict[key]
            else:
                print(key, "Size don't match, skip")
        

        state_dict = tmp_state_dict

        m, u = model.load_state_dict(state_dict, strict=False)
        print(f"### missing keys: {len(m)}; \n### unexpected keys: {len(u)};")
        print(m)

        params = [p.numel() if "." in n else 0 for n, p in model.named_parameters()]
        print(f"### All Parameters: {sum(params) / 1e6} M")

        return model



def pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
    # Unpacked latents of shape are [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
    # The patch dimensions are then permuted and collapsed into the channel dimension of shape:
    # [B, F // p_t * H // p * W // p, C * p_t * p * p] (an ndim=3 tensor).
    # dim=0 is the batch size, dim=1 is the effective video sequence length, dim=2 is the effective number of input features
    batch_size, num_channels, num_frames, height, width = latents.shape
    post_patch_num_frames = num_frames // patch_size_t
    post_patch_height = height // patch_size
    post_patch_width = width // patch_size
    latents = latents.reshape(
        batch_size,
        -1,
        post_patch_num_frames,
        patch_size_t,
        post_patch_height,
        patch_size,
        post_patch_width,
        patch_size,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


# Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._unpack_latents
def unpack_latents(
    latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
) -> torch.Tensor:
    # Packed latents of shape [B, S, D] (S is the effective video sequence length, D is the effective feature dimensions)
    # are unpacked and reshaped into a video tensor of shape [B, C, F, H, W]. This is the inverse operation of
    # what happens in the `_pack_latents` method.
    batch_size = latents.size(0)
    latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return latents


EntryClass = EraserDiTLTXVideoTransformer3DModel
