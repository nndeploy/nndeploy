"""Out-of-place residual update; preserves the intermediate BF16 product."""
import torch
import triton
import triton.language as tl

from nndeploy.e_dit.layers.operator_fusion.triton.rmsnorm_adaln import _add_rn_f32, _mul_rn_f32


@triton.jit
def _gated_residual_kernel(H, U, G, O, N: tl.constexpr, WIDTH: tl.constexpr,
                           TOKENS: tl.constexpr, GATE_BATCH_STRIDE: tl.constexpr,
                           BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    gate_offsets = offsets // (TOKENS * WIDTH) * GATE_BATCH_STRIDE + offsets % WIDTH
    h = tl.load(H + offsets, mask, other=0).to(tl.float32)
    u = tl.load(U + offsets, mask, other=0).to(tl.float32)
    g = tl.load(G + gate_offsets, mask, other=0).to(tl.float32)
    product = _mul_rn_f32(u, g).to(tl.bfloat16).to(tl.float32)
    tl.store(O + offsets, _add_rn_f32(h, product), mask)


def triton_gated_residual(hidden, update, gate):
    output = torch.empty_like(hidden)
    _gated_residual_kernel[(triton.cdiv(hidden.numel(), 1024),)](
        hidden, update, gate, output, hidden.numel(), hidden.shape[-1],
        hidden.shape[1], gate.stride(0), BLOCK=1024, num_warps=4,
    )
    return output
