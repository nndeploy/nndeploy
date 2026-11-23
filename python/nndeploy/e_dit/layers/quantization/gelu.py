"""Exact BF16 GELU lookup shared by INT8 and FP8 activation quantization.

A 128 KiB lookup table preserves the installed PyTorch CUDA GELU rounding
for every BF16 bit pattern. No BF16 activation output is materialized.
"""
import torch
from torch import nn
import triton
import triton.language as tl


@torch.no_grad()
def make_gelu_lut(device):
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('build the GELU lookup table on the CUDA execution device')
    values = torch.arange(65536, device=device, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    return torch.nn.functional.gelu(values, approximate='tanh')


@triton.jit
def _gelu_quant_rows(X, LUT, Q, S, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    bits = tl.load(X + row * K + cols, cols < K, 0).to(tl.uint16, bitcast=True).to(tl.int32)
    x = tl.load(LUT + bits).to(tl.float32)
    maximum = tl.max(tl.abs(x), 0)
    scale = tl.where(maximum > 0, maximum / 127., 1.)
    quant = tl.extra.cuda.libdevice.nearbyint(x / scale)
    quant = tl.minimum(127., tl.maximum(-127., quant)).to(tl.int8)
    tl.store(Q + row * K + cols, quant, cols < K)
    tl.store(S + row, scale)


class ProjectionBeforeFusedGelu(nn.Module):
    """The matching next Linear owns GELU+quantization; names remain stable."""
    def __init__(self, projection):
        super().__init__()
        self.proj = projection

    def forward(self, value):
        return self.proj(value)
