"""Full-width non-affine RMSNorm + AdaLN with local BF16 boundaries."""
import torch
import triton
import triton.language as tl


@triton.jit
def _kernel(X, SCALE, SHIFT, Y, WIDTH: tl.constexpr, EPS: tl.constexpr):
    cols = tl.arange(0, WIDTH)
    offsets = tl.program_id(0) * WIDTH + cols
    x = tl.load(X + offsets).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, 0) / WIDTH + EPS)
    normalized = (x * rstd).to(tl.bfloat16).to(tl.float32)
    scale = (1. + tl.load(SCALE + cols).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    scaled = (normalized * scale).to(tl.bfloat16).to(tl.float32)
    shift = tl.load(SHIFT + cols).to(tl.float32)
    tl.store(Y + offsets, scaled + shift)


def triton_rmsnorm_adaln_fast(hidden, scale, shift, eps):
    output = torch.empty_like(hidden)
    _kernel[(hidden.numel() // hidden.shape[-1],)](
        hidden, scale, shift, output, WIDTH=hidden.shape[-1], EPS=eps,
        num_warps=4, enable_fp_fusion=False,
    )
    return output
