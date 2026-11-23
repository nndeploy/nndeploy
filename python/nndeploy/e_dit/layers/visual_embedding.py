"""Embedding helpers for the minimal runtime."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings as _CombinedTimestepGuidanceTextProjEmbeddings,
)
from diffusers.models.embeddings import (
    CombinedTimestepTextProjEmbeddings as _CombinedTimestepTextProjEmbeddings,
)
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding
from diffusers.models.embeddings import Timesteps as _Timesteps
from diffusers.models.embeddings import get_timestep_embedding as timestep_embedding_diffusers

from nndeploy.e_dit.layers.activation import get_act_fn
from nndeploy.e_dit.layers.linear import ColumnParallelLinear
from nndeploy.e_dit.layers.mlp import MLP


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
        dtype=None,
        prefix: str = "",
    ):
        super().__init__()
        del prefix
        if isinstance(patch_size, (list, tuple)):
            patch_size = tuple(patch_size)
            if len(patch_size) == 1:
                patch_size = (patch_size[0], patch_size[0], patch_size[0])
            elif len(patch_size) == 2:
                patch_size = (1, patch_size[0], patch_size[1])
        else:
            patch_size = (1, patch_size, patch_size)
        self.patch_size = patch_size
        self.flatten = flatten
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
            dtype=dtype,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


class Timesteps(_Timesteps):
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return timestep_embedding_diffusers(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class CombinedTimestepGuidanceTextProjEmbeddings(
    _CombinedTimestepGuidanceTextProjEmbeddings
):
    def __init__(self, embedding_dim, pooled_projection_dim):
        nn.Module.__init__(self)
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")


class CombinedTimestepTextProjEmbeddings(_CombinedTimestepTextProjEmbeddings):
    def __init__(self, embedding_dim, pooled_projection_dim):
        nn.Module.__init__(self)
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size,
        act_layer="silu",
        frequency_embedding_size=256,
        max_period=10000,
        dtype=None,
        freq_dtype=torch.float32,
        prefix: str = "",
        quant_config=None,
    ):
        super().__init__()
        if quant_config is not None:
            raise ValueError("Quantized embeddings are not supported")
        del prefix
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.mlp = MLP(
            frequency_embedding_size,
            hidden_size,
            hidden_size,
            act_type=act_layer,
            dtype=dtype,
        )
        self.freq_dtype = freq_dtype

    def forward(self, t: torch.Tensor, timestep_seq_len: int | None = None) -> torch.Tensor:
        t_freq = timestep_embedding(
            t, self.frequency_embedding_size, self.max_period, dtype=self.freq_dtype
        ).to(self.mlp.fc_in.weight.dtype)
        if timestep_seq_len is not None:
            if t_freq.shape[0] % timestep_seq_len != 0:
                raise ValueError("timestep length is not divisible by timestep_seq_len")
            batch_size = t_freq.shape[0] // timestep_seq_len
            t_freq = t_freq.unflatten(0, (batch_size, timestep_seq_len))
        return self.mlp(t_freq)


def timestep_embedding(
    t: torch.Tensor,
    dim: int,
    max_period: int = 10000,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=dtype, device=t.device)
        / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ModulateProjection(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        factor: int = 2,
        act_layer: str = "silu",
        dtype: torch.dtype | None = None,
        prefix: str = "",
        quant_config=None,
    ):
        super().__init__()
        if quant_config is not None:
            raise ValueError("Quantized embeddings are not supported")
        del prefix
        self.factor = factor
        self.hidden_size = hidden_size
        self.linear = ColumnParallelLinear(
            hidden_size,
            hidden_size * factor,
            bias=True,
            params_dtype=dtype,
        )
        self.act = get_act_fn(act_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(x)
        x, _ = self.linear(x)
        return x


def unpatchify(x, t, h, w, patch_size, channels) -> torch.Tensor:
    patch_t, patch_h, patch_w = patch_size
    x = x.view(x.shape[0], t, h, w, patch_t, patch_h, patch_w, channels)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
    return x.view(x.shape[0], channels, t * patch_t, h * patch_h, w * patch_w)
