"""INT8 GEMM with fused row/channel scales and BF16 bias/output."""
import torch
import triton
import triton.language as tl


@triton.jit
def _gemm(A, W, AS, WS, BIAS, Y, M: tl.constexpr, N: tl.constexpr,
          K: tl.constexpr, HAS_BIAS: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    mi = tl.program_id(0) * BM + tl.arange(0, BM)
    ni = tl.program_id(1) * BN + tl.arange(0, BN)
    ki = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.int32)
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + ki
        a = tl.load(A + mi[:, None] * K + kk[None, :], (mi[:, None] < M) & (kk[None, :] < K), 0.)
        w = tl.load(W + ni[None, :] * K + kk[:, None], (ni[None, :] < N) & (kk[:, None] < K), 0.)
        acc = tl.dot(a, w, acc)
    value = acc.to(tl.float32) * tl.load(AS + mi, mi < M, 0)[:, None]
    value = value * tl.load(WS + ni, ni < N, 0)[None, :]
    if HAS_BIAS:
        value = value + tl.load(BIAS + ni, ni < N, 0).to(tl.float32)[None, :]
    tl.store(Y + mi[:, None] * N + ni[None, :], value,
             (mi[:, None] < M) & (ni[None, :] < N))


def int8_scaled_gemm(activation, weight, activation_scale, weight_scale, bias):
    rows, inner = activation.shape
    columns = weight.shape[0]
    output = torch.empty((rows, columns), device=activation.device, dtype=torch.bfloat16)
    # Measured SM89 FFN expansion: larger M tile improves reuse. Keep the
    # existing tile for other shapes/devices, including short windows.
    large_expansion = (rows >= 8192 and inner == 2048 and columns == 8192
                       and torch.cuda.get_device_capability(activation.device) == (8, 9))
    bm, bn, bk, warps = (128, 128, 64, 8) if large_expansion else (64, 128, 64, 4)
    # The 121-frame 1080p window benefits from additional activation reuse.
    # Shorter windows did not improve in paired measurements; retain their tile.
    if large_expansion and rows == 32640:
        bm = 256
    _gemm[(triton.cdiv(rows, bm), triton.cdiv(columns, bn))](
        activation, weight, activation_scale, weight_scale, bias, output,
        rows, columns, inner, bias is not None,
        bm, bn, bk, num_warps=warps, num_stages=3)
    return output
