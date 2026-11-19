"""Orthogonal process groups for DiT; no model or application dependencies.

The TP-fastest, Ulysses-before-Ring ordering follows SGLang's diffusion
parallel_state / parallel_groups design. This is a local implementation.
"""
import torch
import torch.distributed as dist


class DiTGroups:
    def __init__(self, topology):
        self.topology = topology
        self.rank = dist.get_rank()
        self.coordinates = topology.coordinates(self.rank)
        self.groups = {}
        self.owned = []
        # Axes with identical rank sets can share a communicator. In pure
        # Ulysses, for example, SP == Ulysses == WORLD. Creating three NCCL
        # communicators wastes device buffers without adding parallelism.
        cache = {tuple(range(topology.world_size)): dist.group.WORLD}
        for axis in ('tp', 'ulysses', 'ring', 'sp', 'cfg'):
            for ranks in topology.groups(axis):
                if len(ranks) == 1:
                    group = None
                else:
                    if ranks not in cache:
                        cache[ranks] = dist.new_group(list(ranks))
                    group = cache[ranks]
                if self.rank in ranks:
                    self.groups[axis] = (group, ranks, ranks.index(self.rank))
                    if group is not None and group is not dist.group.WORLD and group not in self.owned:
                        self.owned.append(group)
        self.control = dist.new_group(list(range(topology.world_size)), backend='gloo')
        if dist.get_backend() == 'nccl':
            # NCCL allocates communicator buffers lazily, outside PyTorch's
            # caching allocator. Initialize them before model/activation
            # allocations so cached tensors cannot starve a group's first
            # collective on a memory-constrained rank.
            scratch = torch.zeros(1, device=torch.device('cuda', torch.cuda.current_device()))
            dist.all_reduce(scratch)
            for group in self.owned:
                dist.all_reduce(scratch, group=group)
            torch.cuda.synchronize()

    def get(self, axis):
        return self.groups[axis]


def gather_variable(value, group, size, *, dim=1, lengths=None, dst=None):
    """Gather unequal shards, optionally returning only on global rank ``dst``.

    Non-destination ranks return None for a destination-only gather. The default
    remains all-gather for intermediate activations needed by every TP rank.
    """
    if size == 1:
        return value
    if lengths is None:
        count = torch.tensor([value.shape[dim]], device=value.device, dtype=torch.int64)
        counts = [torch.empty_like(count) for _ in range(size)]
        dist.all_gather(counts, count, group=group)
        lengths = [int(c.item()) for c in counts]
    elif len(lengths) != size or lengths[dist.get_rank(group)] != value.shape[dim]:
        raise ValueError('known gather lengths do not match local shard')
    shape = list(value.shape)
    shape[dim] = max(lengths)
    padded = value.new_zeros(shape)
    padded.narrow(dim, 0, value.shape[dim]).copy_(value)
    outputs = [torch.empty_like(padded) for _ in range(size)] if dst is None or dist.get_rank() == dst else None
    if dst is None:
        dist.all_gather(outputs, padded, group=group)
    else:
        dist.gather(padded, gather_list=outputs, dst=dst, group=group)
    if outputs is None:
        return None
    return torch.cat([v.narrow(dim, 0, n) for v, n in zip(outputs, lengths)], dim=dim)


class DiTCacheCoordinator:
    """Minimal branch-SP reduction contract for the existing cache controllers."""
    def __init__(self, groups):
        self.group, ranks, self.rank = groups.get('sp')
        self.world_size = len(ranks)

    def all_reduce(self, tensor):
        if self.world_size > 1:
            dist.all_reduce(tensor, group=self.group)
        return tensor
