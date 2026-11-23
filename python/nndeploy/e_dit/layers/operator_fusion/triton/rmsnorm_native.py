"""Fuse RMSNorm pointwise work while retaining PyTorch's FP32 reduction.

The square tensors have exactly the native contiguous shape and dtype, so
torch.mean retains its reduction order. Epsilon addition and reciprocal square
root also stay native. Every eager BF16 rounding boundary remains explicit.
"""
import torch
import triton
import triton.language as tl

from nndeploy.e_dit.layers.operator_fusion.triton.qk_rmsnorm_rope import _add_rn_f32, _mul_rn_f32


@triton.jit
def _square(X, Y, SX, SY, N: tl.constexpr, PAIRED: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    tl.store(SX + i, _mul_rn_f32(x, x), i < N)
    if PAIRED:
        y = tl.load(Y + i, i < N, 0).to(tl.float32)
        tl.store(SY + i, _mul_rn_f32(y, y), i < N)


def _inverse_rms(value, eps, other=None):
    squared = torch.empty_like(value, dtype=torch.float32)
    other_squared = torch.empty_like(other, dtype=torch.float32) if other is not None else squared
    _square[(triton.cdiv(value.numel(), 1024),)](
        value, value if other is None else other, squared, other_squared,
        value.numel(), other is not None, 1024, num_warps=4)
    inverse = torch.rsqrt(squared.mean(-1, keepdim=True) + eps)
    if other is None:
        return inverse
    return inverse, torch.rsqrt(other_squared.mean(-1, keepdim=True) + eps)


@triton.jit
def _norm_weight(X, W, R, offset, col, row, valid):
    x = tl.load(X + offset, valid, 0).to(tl.float32)
    r = tl.load(R + row, valid, 0)
    x = _mul_rn_f32(x, r).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + col, valid, 0).to(tl.float32)
    return _mul_rn_f32(x, w).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _qk_rope(Q, K, QW, KW, QR, KR, C, S, OQ, OK,
             N: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    row, col = i // WIDTH, i % WIDTH
    q = _norm_weight(Q, QW, QR, i, col, row, valid)
    k = _norm_weight(K, KW, KR, i, col, row, valid)
    qp = _norm_weight(Q, QW, QR, i ^ 1, col ^ 1, row, valid)
    kp = _norm_weight(K, KW, KR, i ^ 1, col ^ 1, row, valid)
    sign = tl.where(col % 2 == 0, -1., 1.)
    c = tl.load(C + i, valid, 0)
    s = tl.load(S + i, valid, 0)
    q = _add_rn_f32(_mul_rn_f32(q, c), _mul_rn_f32(qp * sign, s))
    k = _add_rn_f32(_mul_rn_f32(k, c), _mul_rn_f32(kp * sign, s))
    tl.store(OQ + i, q, valid)
    tl.store(OK + i, k, valid)


def triton_qk_rmsnorm_rope_native(query, key, qw, kw, cos, sin, eps):
    qr, kr = _inverse_rms(query, eps, key)
    oq, ok = torch.empty_like(query), torch.empty_like(key)
    _qk_rope[(triton.cdiv(query.numel(), 1024),)](
        query, key, qw, kw, qr, kr, cos, sin, oq, ok,
        query.numel(), query.shape[-1], 1024, num_warps=4)
    return oq, ok


@triton.jit
def _adaln(X, R, Scale, Shift, Out, N: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    x = tl.load(X + i, valid, 0).to(tl.float32)
    r = tl.load(R + i // WIDTH, valid, 0)
    norm = _mul_rn_f32(x, r).to(tl.bfloat16).to(tl.float32)
    scale = tl.load(Scale + i % WIDTH, valid, 0).to(tl.float32)
    shift = tl.load(Shift + i % WIDTH, valid, 0).to(tl.float32)
    scale = _add_rn_f32(scale, 1.).to(tl.bfloat16).to(tl.float32)
    scaled = _mul_rn_f32(norm, scale).to(tl.bfloat16).to(tl.float32)
    tl.store(Out + i, _add_rn_f32(scaled, shift), valid)


def triton_rmsnorm_adaln_native(hidden, scale, shift, eps):
    inverse = _inverse_rms(hidden, eps)
    output = torch.empty_like(hidden)
    _adaln[(triton.cdiv(hidden.numel(), 1024),)](
        hidden, inverse, scale, shift, output, hidden.numel(), hidden.shape[-1], 1024, num_warps=4)
    return output
