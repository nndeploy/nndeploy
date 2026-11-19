"""Copy Q/K/V directly to Ulysses destination-major wire layout."""
import torch
import triton
import triton.language as tl


@triton.jit
def _pack(Q, K, V, Out, N: tl.constexpr, B: tl.constexpr, L: tl.constexpr,
          H: tl.constexpr, D: tl.constexpr, LOCAL_H: tl.constexpr,
          DEST_H: tl.constexpr, HEAD_OFFSET: tl.constexpr,
          Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr, Q3: tl.constexpr,
          K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr, K3: tl.constexpr,
          V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr, V3: tl.constexpr,
          BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = index % D
    h = index // D % LOCAL_H
    l = index // (D * LOCAL_H) % L
    b = index // (D * LOCAL_H * L) % B
    component = index // (D * LOCAL_H * L * B) % 3
    destination = index // (D * LOCAL_H * L * B * 3)
    h += destination * DEST_H + HEAD_OFFSET
    q = tl.load(Q + b * Q0 + l * Q1 + h * Q2 + d * Q3, (index < N) & (component == 0), 0)
    k = tl.load(K + b * K0 + l * K1 + h * K2 + d * K3, (index < N) & (component == 1), 0)
    v = tl.load(V + b * V0 + l * V1 + h * V2 + d * V3, (index < N) & (component == 2), 0)
    # Select, rather than add: preserve NaNs, infinities and signed zeros.
    value = tl.where(component == 0, q, tl.where(component == 1, k, v))
    tl.store(Out + index, value, index < N)


def pack_qkv(query, key, value, degree, *, head_offset=0, head_count=None):
    if (query.ndim != 4 or any(t.shape != query.shape or t.device != query.device
                              or t.dtype != query.dtype for t in (key, value))):
        raise ValueError('QKV packing requires matching B,L,H,D shape/device/dtype')
    if not query.is_cuda or query.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError('QKV packing requires CUDA floating tensors')
    batch, length, heads, dim = query.shape
    if degree < 1 or heads % degree or min(query.shape) < 1:
        raise ValueError('QKV packing requires nonempty shape and heads divisible by degree')
    destination_heads = heads // degree
    count = destination_heads if head_count is None else head_count
    if head_offset < 0 or count < 1 or head_offset + count > destination_heads:
        raise ValueError('QKV head range must fit each destination head shard')
    output = query.new_empty(batch * length * degree * count * dim * 3)
    _pack[(triton.cdiv(output.numel(), 1024),)](
        query, key, value, output, output.numel(), batch, length, heads, dim, count, destination_heads, head_offset,
        *query.stride(), *key.stride(), *value.stride(), 1024)
    return output
