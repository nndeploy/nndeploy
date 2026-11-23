"""Sequence partitions and peer attention transport, independent of model layout."""
import torch


class SequenceRank:
    def __init__(self, exchange, rank, attention_mode='ulysses'):
        self.exchange, self.rank = exchange, rank
        self.attention_mode = attention_mode
        self.degree = exchange.degree
        self.start = self.end = 0

    def partition(self, length):
        if length < self.degree:
            raise ValueError("sequence length must be at least SP degree")
        self.start = length * self.rank // self.degree
        self.end = length * (self.rank + 1) // self.degree
        self.length = length
        return slice(self.start, self.end)


    def attention(self, query, key, value, impl, metadata):
        if self.attention_mode == 'ring':
            from nndeploy.e_dit.layers.attention.ring import ring_attention
            return ring_attention(self, query, key, value, impl, metadata)
        heads = query.shape[2]
        if heads % self.degree:
            raise ValueError("attention heads must be divisible by SP degree")
        head_slice = slice(heads * self.rank // self.degree, heads * (self.rank + 1) // self.degree)
        from nndeploy.e_dit.layers.attention.backends.sage_attn import SageAttentionImpl
        if isinstance(impl, SageAttentionImpl):
            # Sage 1.x subtracts K.mean(sequence) in BF16. Changing the number
            # of heads changes PyTorch's reduction layout. Preserve the full
            # layout for this reduction before selecting the local heads.
            full_key = self.exchange.exchange(self.rank, (key,), lambda x: x, dim=1)[0]
            full_key.sub_(full_key.mean(dim=1, keepdim=True))
            q, v = self.exchange.exchange(self.rank, (query, value),
                                          lambda x: x[:, :, head_slice], dim=1)
            k = full_key[:, :, head_slice].contiguous()
            output = impl.forward(q, k, v, metadata, key_already_smoothed=True)
        else:
            q, k, v = self.exchange.exchange(self.rank, (query, key, value),
                                            lambda x: x[:, :, head_slice], dim=1)
            output = impl.forward(q, k, v, metadata)
        return self.exchange.exchange(self.rank, (output,),
                                      lambda x: x[:, self.start:self.end], dim=2)[0]
