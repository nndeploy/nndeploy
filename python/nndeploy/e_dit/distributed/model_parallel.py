"""Model-facing access to nndeploy runtime groups."""
import torch
import torch.distributed as dist
from .parallel_state import get_parallel_context
from nndeploy.e_dit.platforms import get_local_torch_device

def _group(kind):
    context = get_parallel_context()
    if context is None or not context.enabled:
        return None
    matches = [g for name, g in context.groups.items() if name.startswith(kind) and g.is_member]
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError(f'Ambiguous {kind} model group')
    return matches[0]

def get_sp_world_size():
    group = _group('sp')
    return group.world_size if group else 1

def get_sp_parallel_rank():
    group = _group('sp')
    return group.group_rank if group else 0

def get_tp_world_size():
    group = _group('tp')
    return group.world_size if group else 1

def get_tp_rank():
    group = _group('tp')
    return group.group_rank if group else 0

def model_parallel_is_initialized():
    return get_parallel_context() is not None

def sequence_model_parallel_all_gather(tensor, dim=0):
    group = _group('sp')
    if group is None or group.world_size == 1:
        return tensor
    values = [torch.empty_like(tensor) for _ in range(group.world_size)]
    dist.all_gather(values, tensor.contiguous(), group=group.process_group)
    return torch.cat(values, dim=dim)

def tensor_model_parallel_all_reduce(tensor):
    group = _group('tp')
    if group is not None and group.world_size > 1:
        dist.all_reduce(tensor, group=group.process_group)
    return tensor

def divide(numerator, denominator):
    if numerator % denominator:
        raise ValueError(f'{numerator} is not divisible by {denominator}')
    return numerator // denominator

class _ModelGroup:
    def __init__(self, kind):
        self.group = _group(kind)
    @property
    def world_size(self):
        return self.group.world_size if self.group else 1
    @property
    def rank_in_group(self):
        return self.group.group_rank if self.group else 0
    @property
    def device_group(self):
        return self.group.process_group if self.group else None
    def all_gather(self, tensor, dim=0):
        return sequence_model_parallel_all_gather(tensor, dim)

def get_sp_group():
    return _ModelGroup('sp')
