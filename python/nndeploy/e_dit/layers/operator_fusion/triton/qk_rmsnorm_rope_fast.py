"""Inference-only full-width Q/K RMSNorm and adjacent-pair RoPE fusion.

FP32 reduction order and RoPE FMA may differ from eager PyTorch. Preserve
BF16 boundaries before/after affine weights to limit accumulated model error.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _kernel(Q, K, QW, KW, COS, SIN, OQ, OK, WIDTH: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, WIDTH)
    offset = row * WIDTH + col
    q = tl.load(Q + offset).to(tl.float32)
    k = tl.load(K + offset).to(tl.float32)
    qr = tl.rsqrt(tl.sum(q * q, 0) / WIDTH + EPS)
    kr = tl.rsqrt(tl.sum(k * k, 0) / WIDTH + EPS)
    q = (q * qr).to(tl.bfloat16).to(tl.float32)
    k = (k * kr).to(tl.bfloat16).to(tl.float32)
    q = (q * tl.load(QW + col).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    k = (k * tl.load(KW + col).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    paired = row * WIDTH + (col ^ 1)
    qp = (tl.load(Q + paired).to(tl.float32) * qr).to(tl.bfloat16).to(tl.float32)
    kp = (tl.load(K + paired).to(tl.float32) * kr).to(tl.bfloat16).to(tl.float32)
    qp = (qp * tl.load(QW + (col ^ 1)).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    kp = (kp * tl.load(KW + (col ^ 1)).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    sign = tl.where(col % 2 == 0, -1., 1.)
    c = tl.load(COS + offset)
    s = tl.load(SIN + offset)
    tl.store(OQ + offset, q * c + qp * sign * s)
    tl.store(OK + offset, k * c + kp * sign * s)


def triton_qk_rmsnorm_rope_fast(query, key, query_weight, key_weight, cos, sin, eps):
    out_q, out_k = torch.empty_like(query), torch.empty_like(key)
    _kernel[(query.numel() // query.shape[-1],)](
        query, key, query_weight, key_weight, cos, sin, out_q, out_k,
        WIDTH=query.shape[-1], EPS=eps, num_warps=4,
    )
    return out_q, out_k
