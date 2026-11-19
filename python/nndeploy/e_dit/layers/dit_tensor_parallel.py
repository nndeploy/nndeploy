"""Column-sharded Linear with gathered outputs for conservative TP alignment.

Like SGLang's ColumnParallelLinear(gather_output=True), each rank owns only
its output-channel weights. Gathering preserves full RMSNorm/RoPE/GEGLU
semantics. Reference mode pads dummy channels to preserve BF16 GEMM shape;
sharded mode computes only local channels. No real full weight is retained.
"""
import torch
import torch.distributed as dist
from torch import nn

from nndeploy.e_dit.distributed.dit_groups import gather_variable


class GatheredColumnLinear(nn.Module):
    def __init__(self, linear, group, rank, degree, mode='reference'):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.start = self.out_features * rank // degree
        self.end = self.out_features * (rank + 1) // degree
        if self.start == self.end:
            raise ValueError('TP degree exceeds Linear output width')
        self.weight = nn.Parameter(linear.weight[self.start:self.end].detach().clone(), requires_grad=False)
        self.bias = None if linear.bias is None else nn.Parameter(
            linear.bias[self.start:self.end].detach().clone(), requires_grad=False)
        self.group, self.degree = group, degree
        self.mode = mode
        self.lengths = tuple(self.out_features * (i + 1) // degree - self.out_features * i // degree
                             for i in range(degree))

    def forward(self, value):
        if self.mode == 'reference':
            # Preserve original GEMM N and channel offsets for BF16 alignment.
            # Only this layer's zero-filled temporary has the full width; real
            # weights remain sharded. This saves persistent weights, not FLOPs.
            weight = torch.nn.functional.pad(self.weight, (0, 0, self.start, self.out_features - self.end))
            bias = None if self.bias is None else torch.nn.functional.pad(
                self.bias, (self.start, self.out_features - self.end))
            local = torch.nn.functional.linear(value, weight, bias)[..., self.start:self.end]
        else:
            local = torch.nn.functional.linear(value, self.weight, self.bias)
        return gather_variable(local, self.group, self.degree, dim=local.ndim - 1, lengths=self.lengths)


def shard_linear_weights(model, group, rank, degree, mode='reference', *, replicated_names=(),
                         reference_names=(), prefix=''):
    count = 0
    for name, module in list(model.named_children()):
        full_name = f'{prefix}.{name}' if prefix else name
        if full_name in replicated_names:
            continue
        if isinstance(module, nn.Linear):
            if module.out_features < degree:
                # Tiny output projections cannot be split into nonempty
                # shards. Keep those few parameters replicated explicitly.
                continue
            layer_mode = 'reference' if full_name in reference_names else mode
            setattr(model, name, GatheredColumnLinear(module, group, rank, degree, layer_mode))
            count += 1
        else:
            count += shard_linear_weights(module, group, rank, degree, mode,
                                          replicated_names=replicated_names,
                                          reference_names=reference_names, prefix=full_name)
    return count
