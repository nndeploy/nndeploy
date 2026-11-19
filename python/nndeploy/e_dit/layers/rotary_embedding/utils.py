"""Primitive RoPE ops."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def _apply_rotary_emb(
    x: torch.Tensor,
    cos_sin: Tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
    is_neox_style: bool,
    interleaved: bool = False,
) -> torch.Tensor:
    del interleaved
    if isinstance(cos_sin, tuple):
        cos, sin = cos_sin
    else:
        cos, sin = cos_sin.chunk(2, dim=-1)
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    if is_neox_style:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    else:
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        rotated = torch.cat((-x2, x1), dim=-1)
    return (x.float() * cos + rotated.float() * sin).to(x.dtype)


def apply_flashinfer_rope_qk_inplace(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = _apply_rotary_emb(query, (cos, sin), is_neox_style=is_neox)
    key = _apply_rotary_emb(key, (cos, sin), is_neox_style=is_neox)
    return query, key
