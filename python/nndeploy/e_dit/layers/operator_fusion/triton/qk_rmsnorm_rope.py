"""Paired adjacent-feature RoPE after native Q/K RMSNorm.

Normalization stays in PyTorch to preserve its reduction and rounding. The
rotation keeps standalone FP32 multiply/add rounding, followed by the same
BF16 cast as the corresponding eager operations.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _mul_rn_f32(lhs, rhs):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;", "=f,f,f", [lhs, rhs],
        dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _add_rn_f32(lhs, rhs):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;", "=f,f,f", [lhs, rhs],
        dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _joint_qk_rope_kernel(
    query_ptr, key_ptr, cos_ptr, sin_ptr, query_output_ptr, key_output_ptr,
    NUMEL: tl.constexpr, BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < NUMEL
    # Width is even, so adjacent pairs never cross a token boundary.
    paired = offsets ^ 1
    sign = tl.where((offsets & 1) == 0, -1.0, 1.0)
    cos = tl.load(cos_ptr + offsets, valid, other=0)
    sin = tl.load(sin_ptr + offsets, valid, other=0)
    query = tl.load(query_ptr + offsets, valid, other=0).to(tl.float32)
    query_rotated = tl.load(query_ptr + paired, valid, other=0).to(tl.float32) * sign
    key = tl.load(key_ptr + offsets, valid, other=0).to(tl.float32)
    key_rotated = tl.load(key_ptr + paired, valid, other=0).to(tl.float32) * sign
    q_out = _add_rn_f32(_mul_rn_f32(query, cos), _mul_rn_f32(query_rotated, sin))
    k_out = _add_rn_f32(_mul_rn_f32(key, cos), _mul_rn_f32(key_rotated, sin))
    tl.store(query_output_ptr + offsets, q_out, valid)
    tl.store(key_output_ptr + offsets, k_out, valid)


def triton_qk_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate normalized Q/K; the public dispatcher validates the contract."""
    query_output = torch.empty_like(query)
    key_output = torch.empty_like(key)
    _joint_qk_rope_kernel[(triton.cdiv(query.numel(), 1024),)](
        query, key, cos, sin, query_output, key_output,
        NUMEL=query.numel(), BLOCK=1024, num_warps=4,
    )
    return query_output, key_output
