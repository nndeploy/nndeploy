"""SM89-compatible E4M3 W8A8 Linear with token/channel or tensor scaling.

PyTorch 2.6's rowwise scaled_mm requires newer hardware. Use scalar unit
scales and FP32 output for token/channel mode, then fuse dequantization and bias.
Tensorwise mode directly produces scaled BF16 output with fused bias.
No BF16 weight copy or BF16 GEMM fallback is retained.
"""
import torch
from torch import nn
import triton
import triton.language as tl

from nndeploy.e_dit.layers.quantization.int8 import _epilogue


@triton.jit
def _load_activation(X, indices, mask, LUT, HAS_GELU: tl.constexpr):
    value = tl.load(X + indices, mask, 0)
    if HAS_GELU:
        bits = value.to(tl.uint16, bitcast=True).to(tl.int32)
        value = tl.load(LUT + bits)
    return value.to(tl.float32)


@triton.jit
def _quant_fp8_rows(X, Q, S, K: tl.constexpr, BLOCK: tl.constexpr,
                   LUT=None, HAS_GELU: tl.constexpr = False):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = _load_activation(X, row * K + cols, cols < K, LUT, HAS_GELU)
    maximum = tl.max(tl.abs(x), 0)
    scale = tl.where(maximum > 0, maximum * (1. / 448.), 1.)
    q = tl.minimum(448., tl.maximum(-448., tl.div_rn(x, scale)))
    # Avoid Triton's FP32 -> FP16 -> E4M3 double rounding on SM89.
    packed = tl.inline_asm_elementwise(
        "cvt.rn.satfinite.e4m3x2.f32 $0, 0.0, $1;", constraints="=h,f",
        args=[q], dtype=tl.uint16, is_pure=True, pack=1)
    q8 = packed.to(tl.uint8).to(tl.float8e4nv, bitcast=True)
    tl.store(Q + row * K + cols, q8, cols < K)
    tl.store(S + row, scale)


@triton.jit
def _partial_max(X, P, TOTAL: tl.constexpr, BLOCK: tl.constexpr,
                 LUT=None, HAS_GELU: tl.constexpr = False):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = _load_activation(X, idx, idx < TOTAL, LUT, HAS_GELU)
    tl.store(P + tl.program_id(0), tl.max(tl.abs(x), 0))


@triton.jit
def _finish_max(P, S, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    maximum = tl.max(tl.load(P + idx, idx < COUNT, 0), 0)
    tl.store(S, tl.where(maximum > 0, maximum * (1. / 448.), 1.))


@triton.jit
def _quant_fp8_tensor(X, Q, S, TOTAL: tl.constexpr, BLOCK: tl.constexpr,
                      LUT=None, HAS_GELU: tl.constexpr = False):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = _load_activation(X, idx, idx < TOTAL, LUT, HAS_GELU)
    q = tl.minimum(448., tl.maximum(-448., tl.div_rn(x, tl.load(S))))
    packed = tl.inline_asm_elementwise(
        "cvt.rn.satfinite.e4m3x2.f32 $0, 0.0, $1;", constraints="=h,f",
        args=[q], dtype=tl.uint16, is_pure=True, pack=1)
    q8 = packed.to(tl.uint8).to(tl.float8e4nv, bitcast=True)
    tl.store(Q + idx, q8, idx < TOTAL)


@triton.jit
def _quant_fp8_static(X, Q, TOTAL: tl.constexpr, BLOCK: tl.constexpr,
                      INVERSE_SCALE: tl.constexpr, LUT, HAS_GELU: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = _load_activation(X, idx, idx < TOTAL, LUT, HAS_GELU)
    q = tl.minimum(448., tl.maximum(-448., x * INVERSE_SCALE))
    # Speed-first mode allows the native Triton FP8 conversion and saturation.
    tl.store(Q + idx, q.to(tl.float8e4nv), idx < TOTAL)


def quantize_fp8_tensor(x, gelu_lut=None):
    """Dynamic GPU amax; optional exact GELU avoids a BF16 intermediate."""
    quant = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty((), device=x.device, dtype=torch.float32)
    count = triton.cdiv(x.numel(), 8192)
    partial = torch.empty(count, device=x.device, dtype=torch.float32)
    _partial_max[(count,)](x, partial, x.numel(), 8192,
                          gelu_lut, gelu_lut is not None, num_warps=8)
    _finish_max[(1,)](partial, scale, count, triton.next_power_of_2(count), num_warps=8)
    _quant_fp8_tensor[(count,)](x, quant, scale, x.numel(), 8192,
                               gelu_lut, gelu_lut is not None, num_warps=8)
    return quant, scale


def _quantize_activation(x, gelu_lut, tensorwise, static_scale=0.):
    if static_scale:
        quant = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        _quant_fp8_static[(triton.cdiv(x.numel(), 8192),)](
            x, quant, x.numel(), 8192, 1. / static_scale,
            gelu_lut, gelu_lut is not None, num_warps=8)
        # Static Linear owns its persistent GEMM scale; no reduction or allocation.
        return quant, None
    if tensorwise:
        return quantize_fp8_tensor(x, gelu_lut)
    rows, inner = x.shape
    quant = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty(rows, device=x.device, dtype=torch.float32)
    _quant_fp8_rows[(rows,)](x, quant, scale, inner, triton.next_power_of_2(inner),
                            gelu_lut, gelu_lut is not None, num_warps=8)
    return quant, scale


@torch.library.custom_op('nndeploy_dit::quantize_fp8_activation', mutates_args=())
def _compiled_quantize_activation(x: torch.Tensor, gelu_lut: torch.Tensor | None,
                                 tensorwise: bool) -> tuple[torch.Tensor, torch.Tensor]:
    # Torch 2.6's Triton mutation analysis cannot handle the E4M3 inline asm.
    # Keep this allocation-only operation opaque instead of treating inputs
    # (including the offloaded LUT) as mutated compiler intermediates.
    return _quantize_activation(x, gelu_lut, tensorwise)


@_compiled_quantize_activation.register_fake
def _compiled_quantize_activation_fake(x, gelu_lut, tensorwise):
    return (torch.empty_like(x, dtype=torch.float8_e4m3fn),
            torch.empty(() if tensorwise else (x.shape[0],), device=x.device, dtype=torch.float32))


@torch.library.custom_op('nndeploy_dit::quantize_static_fp8_activation', mutates_args=())
def _compiled_quantize_static(x: torch.Tensor, gelu_lut: torch.Tensor | None) -> torch.Tensor:
    return _quantize_activation(x, gelu_lut, True, .125)[0]


@_compiled_quantize_static.register_fake
def _compiled_quantize_static_fake(x, gelu_lut):
    return torch.empty_like(x, dtype=torch.float8_e4m3fn)


class NativeFp8Linear(nn.Module):
    tensorwise = False
    static_activation = False
    static_scale = 0.
    use_fast_accum = False

    def __init__(self, weight_fp8, weight_scale, bias):
        super().__init__()
        self.out_features, self.in_features = weight_fp8.shape
        self.register_buffer('weight_fp8', weight_fp8)
        self.register_buffer('weight_scale', weight_scale)
        self.register_buffer('bias', bias)
        self.register_buffer('unit_scale', torch.ones((), device=weight_fp8.device))
        self.register_buffer('gelu_lut', None)
        self.register_buffer('activation_scale', torch.tensor(self.static_scale, device=weight_fp8.device,
                             dtype=torch.float32) if self.static_scale else None)
        self.calls = 0
        self.gelu_calls = 0

    @classmethod
    def from_linear(cls, source, *, execution_device=None):
        if source.weight.device.type not in ('cpu', 'cuda') or source.weight.dtype != torch.bfloat16:
            raise ValueError('FP8 conversion requires CPU/CUDA BF16 Linear')
        if source.in_features % 32 or source.out_features % 32:
            raise ValueError('FP8 Linear dimensions must be multiples of 32')
        with torch.no_grad():
            weight = source.weight.detach().to(device=execution_device or source.weight.device, dtype=torch.float32)
            scale = (weight.abs().amax() if cls.tensorwise else weight.abs().amax(dim=1)).div(448.)
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            quant = (weight / (scale if cls.tensorwise else scale[:, None])).clamp(-448, 448).to(torch.float8_e4m3fn)
            bias = source.bias.detach().to(device=weight.device, copy=True) if source.bias is not None else None
        return cls(quant.contiguous(), scale, bias)

    def forward(self, value):
        if value.device != self.weight_fp8.device or value.dtype != torch.bfloat16:
            raise ValueError('FP8 Linear requires same-device BF16 activations')
        if value.shape[-1] != self.in_features:
            raise ValueError('FP8 Linear input feature mismatch')
        if torch.is_grad_enabled() and value.requires_grad:
            raise RuntimeError('FP8 Linear is inference-only')
        x = value.reshape(-1, self.in_features).contiguous()
        rows = x.shape[0]
        if rows == 0:
            return value.new_empty((*value.shape[:-1], self.out_features))
        if self.static_activation:
            quant = (_compiled_quantize_static(x, self.gelu_lut)
                     if torch.compiler.is_compiling() else
                     _quantize_activation(x, self.gelu_lut, True, .125)[0])
            scale = self.activation_scale
        else:
            quantize = _compiled_quantize_activation if torch.compiler.is_compiling() else _quantize_activation
            quant, scale = quantize(x, self.gelu_lut, self.tensorwise)
        if self.tensorwise:
            output = torch._scaled_mm(
                quant, self.weight_fp8.t(), scale_a=scale, scale_b=self.weight_scale,
                bias=self.bias, out_dtype=torch.bfloat16, use_fast_accum=self.use_fast_accum)
            if not torch.compiler.is_compiling():
                self.calls += 1
                self.gelu_calls += int(self.gelu_lut is not None)
            return output.reshape(*value.shape[:-1], self.out_features)
        accum = torch._scaled_mm(quant, self.weight_fp8.t(),
                                 scale_a=self.unit_scale, scale_b=self.unit_scale,
                                 out_dtype=torch.float32, use_fast_accum=False)
        output = torch.empty((rows, self.out_features), device=x.device, dtype=value.dtype)
        _epilogue[(triton.cdiv(output.numel(), 1024),)](
            accum, scale, self.weight_scale, self.bias, output,
            self.out_features, output.numel(), self.bias is not None, 1024)
        if not torch.compiler.is_compiling():
            self.calls += 1
            self.gelu_calls += int(self.gelu_lut is not None)
        return output.reshape(*value.shape[:-1], self.out_features)


class NativeTensorwiseFp8Linear(NativeFp8Linear):
    """Experimental tensorwise W8A8 with scaled BF16 output and fused bias.

    Weights are quantized once; activations use a fresh GPU amax each call.
    Scalar scaling supports the native SM89 epilogue without FP32 intermediates.
    """
    tensorwise = True


class NativeStaticFp8Linear(NativeTensorwiseFp8Linear):
    """Speed-first FP8: fixed activation scale 1/8, saturation at +/-56.

    One activation kernel replaces the dynamic amax/reduction/conversion chain.
    Tensorwise weight scaling and fast accumulation change numerical behavior;
    this opt-in mode needs video quality checks for the user's material.
    """
    static_scale = 0.125
    static_activation = True
    use_fast_accum = True
