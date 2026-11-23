"""Minimal normalization layers."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nndeploy.e_dit.layers.rotary_embedding.utils import _apply_rotary_emb


class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        var_hidden_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )

    def forward(
        self, x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.float()
        if residual is not None:
            x = x + residual.float()
            residual = x.to(orig_dtype)
        x_var = x if self.variance_size_override is None else x[..., : self.variance_size_override]
        variance = x_var.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = (x * self.weight.float()).to(orig_dtype)
        return x if residual is None else (x, residual)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.variance_epsilon}"


class LayerNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        bias: bool = True,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype))
            self.bias = nn.Parameter(torch.empty(hidden_size, device=device, dtype=dtype)) if bias else None
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(
        self, x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        x = x.float()
        if residual is not None:
            x = x + residual.float()
            residual = x.to(orig_dtype)
        mean = x.mean(dim=-1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            x = x * self.weight.float()
        if self.bias is not None:
            x = x + self.bias.float()
        x = x.to(orig_dtype)
        return x if residual is None else (x, residual)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}"


class FP32LayerNorm(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)


def _combine_gate(
    residual: torch.Tensor, x: torch.Tensor, gate: torch.Tensor | int
) -> torch.Tensor:
    if isinstance(gate, int):
        if gate != 1:
            raise ValueError(f"Only gate value of 1 is supported for int type, got {gate}")
        return residual + x
    if gate.dim() == 4:
        num_frames = gate.shape[1]
        frame_seqlen = x.shape[1] // num_frames
        return residual + (
            x.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * gate
        ).flatten(1, 2)
    return residual + x * gate


class _ScaleShiftBase(nn.Module):
    norm_type: str = "layer"

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        dtype: torch.dtype = torch.float32,
        prefix: str = "",
    ):
        super().__init__()
        del prefix
        if self.norm_type == "rms":
            self.norm = RMSNorm(hidden_size, eps=eps, dtype=dtype)
        else:
            self.norm = FP32LayerNorm(
                hidden_size, elementwise_affine=elementwise_affine, eps=eps, dtype=dtype
            )


class _NormScaleShift(_ScaleShiftBase):
    def forward(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return (self.norm(x) * (1 + scale) + shift).to(x.dtype)


class LayerNormScaleShift(_NormScaleShift):
    norm_type = "layer"


class RMSNormScaleShift(_NormScaleShift):
    norm_type = "rms"


class _ScaleResidualNormScaleShift(_ScaleShiftBase):
    def forward(
        self,
        residual: torch.Tensor,
        x: torch.Tensor,
        gate: torch.Tensor | int,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual_output = _combine_gate(residual, x, gate)
        modulated = (self.norm(residual_output) * (1 + scale) + shift).to(x.dtype)
        return modulated, residual_output


class ScaleResidualLayerNormScaleShift(_ScaleResidualNormScaleShift):
    norm_type = "layer"


class ScaleResidualRMSNormScaleShift(_ScaleResidualNormScaleShift):
    norm_type = "rms"


class _NormTanhMulAdd(_ScaleShiftBase):
    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return (self.norm(x) * torch.tanh(scale) + shift).to(x.dtype)


class LayerNormTanhMulAdd(_NormTanhMulAdd):
    norm_type = "layer"


class RMSNormTanhMulAdd(_NormTanhMulAdd):
    norm_type = "rms"


def apply_qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    head_dim: int,
    allow_inplace: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    del allow_inplace
    q_shape = q.shape
    k_shape = k.shape
    q_out = q_norm(q.reshape(-1, head_dim)).reshape(q_shape)
    k_out = k_norm(k.reshape(-1, head_dim)).reshape(k_shape)
    return q_out, k_out


def apply_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    head_dim: int,
    cos_sin_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    *,
    is_neox: bool = False,
    positions: Optional[torch.Tensor] = None,
    position_offset: int = 0,
    allow_inplace: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    del positions, position_offset, allow_inplace
    q_out, k_out = apply_qk_norm(q, k, q_norm, k_norm, head_dim)
    if isinstance(cos_sin_cache, tuple):
        cos, sin = cos_sin_cache
    else:
        cos, sin = cos_sin_cache.chunk(2, dim=-1)
    q_out = _apply_rotary_emb(q_out, (cos, sin), is_neox_style=is_neox)
    k_out = _apply_rotary_emb(k_out, (cos, sin), is_neox_style=is_neox)
    return q_out, k_out


def apply_qk_norm_with_optional_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    head_dim: int,
    cos_sin_cache: Optional[torch.Tensor] = None,
    *,
    is_neox: bool = False,
    positions: Optional[torch.Tensor] = None,
    position_offset: int = 0,
    allow_inplace: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cos_sin_cache is None:
        return apply_qk_norm(q, k, q_norm, k_norm, head_dim, allow_inplace=allow_inplace)
    return apply_qk_norm_rope(
        q,
        k,
        q_norm,
        k_norm,
        head_dim,
        cos_sin_cache,
        is_neox=is_neox,
        positions=positions,
        position_offset=position_offset,
        allow_inplace=allow_inplace,
    )

def tensor_parallel_rms_norm(x, norm):
    """RMS reduction across equally sized tensor-parallel feature shards."""
    from nndeploy.e_dit.distributed.model_parallel import (
        get_tp_rank, get_tp_world_size, tensor_model_parallel_all_reduce,
    )
    size = get_tp_world_size()
    variance = tensor_model_parallel_all_reduce(x.float().square().mean(-1, keepdim=True)) / size
    weight = norm.weight.tensor_split(size)[get_tp_rank()].float()
    epsilon = getattr(norm, 'variance_epsilon', getattr(norm, 'eps', 1e-6))
    return (x.float() * torch.rsqrt(variance + epsilon) * weight).to(x.dtype)
