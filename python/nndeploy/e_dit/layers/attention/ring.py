"""Experimental two-rank Ring with online FP32 softmax/LSE merging.

Torch 2.6's native flash forward exposes LSE without quadratic probability
storage. This version-specific boundary stays outside torch.compile.
"""
import torch


def ring_attention(rank, query, key, value, impl, metadata):
    from nndeploy.e_dit.layers.attention.backends.sdpa import SDPAImpl
    if not isinstance(impl, SDPAImpl) or impl.causal or impl.dropout or metadata.attn_mask is not None:
        raise ValueError('Ring requires unmasked noncausal SDPA with dropout=0')
    q = query.transpose(1, 2)
    accumulator = total_lse = None
    for step in range(rank.degree):
        out, lse, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
            q, key.transpose(1, 2), value.transpose(1, 2), 0., False, False,
            scale=impl.softmax_scale)
        if accumulator is None:
            accumulator, total_lse = out.float(), lse
        else:
            combined = torch.logaddexp(total_lse, lse)
            accumulator.mul_(torch.exp(total_lse - combined).unsqueeze(-1))
            accumulator.add_(out.float() * torch.exp(lse - combined).unsqueeze(-1))
            total_lse = combined
        if step + 1 < rank.degree:
            key, value = rank.exchange.rotate(rank.rank, (key, value))
    return accumulator.to(query.dtype).transpose(1, 2).contiguous()
