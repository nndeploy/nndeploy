"""BF16 AdaLN modulation kernel for RMSNorm + AdaLN."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _add_rn_f32(lhs, rhs):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [lhs, rhs],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_rn_f32(lhs, rhs):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [lhs, rhs],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _adaln_modulation_kernel(
    normalized_ptr,
    scale_ptr,
    shift_ptr,
    output_ptr,
    WIDTH: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, WIDTH)
    row_offsets = row_idx * WIDTH + offsets

    normalized = tl.load(normalized_ptr + row_offsets).to(tl.float32)
    scale = tl.load(scale_ptr + offsets).to(tl.float32)
    shift = tl.load(shift_ptr + offsets).to(tl.float32)

    # Eager executes three BF16 pointwise kernels. Keep the intermediate BF16
    # stores as local conversion boundaries and prevent multiply/add contraction.
    one_plus_scale = _add_rn_f32(scale, 1.0).to(tl.bfloat16)
    scaled = _mul_rn_f32(normalized, one_plus_scale.to(tl.float32)).to(tl.bfloat16)
    output = _add_rn_f32(scaled.to(tl.float32), shift)
    tl.store(output_ptr + row_offsets, output)


def triton_adaln_modulation(
    normalized_hidden_states: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    """Apply broadcast AdaLN modulation in one fixed-width launch."""

    width = normalized_hidden_states.shape[-1]
    row_count = normalized_hidden_states.numel() // width
    output = torch.empty_like(normalized_hidden_states)
    _adaln_modulation_kernel[(row_count,)](
        normalized_hidden_states,
        scale,
        shift,
        output,
        WIDTH=width,
        num_warps=8,
    )
    return output
