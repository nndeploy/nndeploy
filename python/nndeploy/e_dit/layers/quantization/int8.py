"""Inference-only symmetric W8A8 Linear: Triton quantization + native INT8 GEMM.

Weights use one FP32 scale per output channel; activations one per token.
No floating weight copies or floating GEMM fallback are retained.
"""
import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _quant_rows(X, Q, S, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row*K + cols, cols < K, 0).to(tl.float32)
    maximum = tl.max(tl.abs(x), 0)
    scale = tl.where(maximum > 0, maximum/127., 1.)
    q = tl.extra.cuda.libdevice.nearbyint(x/scale)
    q = tl.minimum(127., tl.maximum(-127., q)).to(tl.int8)
    tl.store(Q + row*K + cols, q, cols < K)
    tl.store(S + row, scale)


@triton.jit
def _epilogue(C, A, W, B, Y, N: tl.constexpr, TOTAL: tl.constexpr,
              HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    mask = index < TOTAL
    value = tl.load(C+index, mask, 0).to(tl.float32)
    value *= tl.load(A+index//N, mask, 0)
    value *= tl.load(W+index%N, mask, 0)
    if HAS_BIAS:
        value += tl.load(B+index%N, mask, 0).to(tl.float32)
    tl.store(Y+index, value, mask)


class NativeInt8Linear(nn.Module):
    def __init__(self, weight_int8, weight_scale, bias):
        super().__init__()
        self.out_features, self.in_features = weight_int8.shape
        self.register_buffer('weight_int8', weight_int8)
        self.register_buffer('weight_scale', weight_scale)
        self.register_buffer('bias', bias)
        self.register_buffer('gelu_lut', None)
        self.calls = 0
        self.fused_calls = 0
        self.gelu_calls = 0

    @classmethod
    def from_linear(cls, source, *, execution_device=None):
        if source.weight.device.type not in ('cpu', 'cuda') or source.weight.dtype != torch.bfloat16:
            raise ValueError('INT8 conversion requires CPU/CUDA BF16 Linear')
        if source.in_features % 32 or source.out_features % 32:
            raise ValueError('INT8 Linear dimensions must be multiples of 32')
        with torch.no_grad():
            weight = source.weight.detach().to(device=execution_device or source.weight.device, dtype=torch.float32)
            scale = weight.abs().amax(dim=1).div(127.)
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            quant = (weight/scale[:, None]).round().clamp(-127,127).to(torch.int8)
            bias = source.bias.detach().to(device=weight.device, copy=True) if source.bias is not None else None
        return cls(quant.contiguous(), scale, bias)

    def forward(self, value):
        if value.device != self.weight_int8.device or value.dtype != torch.bfloat16:
            raise ValueError('INT8 Linear requires same-device BF16 activations')
        if value.shape[-1] != self.in_features:
            raise ValueError('INT8 Linear input feature mismatch')
        if torch.is_grad_enabled() and value.requires_grad:
            raise RuntimeError('INT8 Linear is inference-only')
        x = value.reshape(-1, self.in_features).contiguous()
        rows = x.shape[0]
        if rows == 0:
            return value.new_empty((*value.shape[:-1],self.out_features))
        padded_rows = max(32, triton.cdiv(rows, 8)*8)
        if padded_rows != rows:
            x = torch.nn.functional.pad(x, (0, 0, 0, padded_rows-rows))
        quant = torch.empty_like(x, dtype=torch.int8)
        scale = torch.empty(padded_rows, device=x.device, dtype=torch.float32)
        if self.gelu_lut is None:
            _quant_rows[(padded_rows,)](x, quant, scale, self.in_features,
                                 triton.next_power_of_2(self.in_features), num_warps=8)
        else:
            from nndeploy.e_dit.layers.quantization.gelu import _gelu_quant_rows
            _gelu_quant_rows[(padded_rows,)](x, self.gelu_lut, quant, scale,
                self.in_features, triton.next_power_of_2(self.in_features), num_warps=8)
            if not torch.compiler.is_compiling():
                self.gelu_calls += 1
        # Expansion layers otherwise write/read a large INT32 intermediate.
        # Keep cuBLAS for contraction and small matrices where it is faster.
        if rows >= 1024 and self.out_features >= 2 * self.in_features:
            from nndeploy.e_dit.layers.quantization.gemm import int8_scaled_gemm
            output = int8_scaled_gemm(quant, self.weight_int8, scale, self.weight_scale, self.bias)
            if not torch.compiler.is_compiling():
                self.calls += 1
                self.fused_calls += 1
            return output[:rows].reshape(*value.shape[:-1], self.out_features)
        accum = torch._int_mm(quant, self.weight_int8.t())
        output = torch.empty((padded_rows,self.out_features),device=x.device,dtype=value.dtype)
        _epilogue[(triton.cdiv(output.numel(),1024),)](
            accum, scale, self.weight_scale, self.bias, output,
            self.out_features, output.numel(), self.bias is not None, 1024)
        if not torch.compiler.is_compiling():
            self.calls += 1
        return output[:rows].reshape(*value.shape[:-1],self.out_features)
