"""Validated DiT worker topology. Ulysses is the low-order SP axis."""
from dataclasses import dataclass


def resolve_default_cfg_degree(
    requested,
    *,
    backend='peer',
    sp_degree=1,
    tp_degree=1,
    dit_fsdp_shard_degree=1,
    dit_fsdp_replicate_degree=1,
    parallel_devices=None,
):
    """Default an explicitly multi-GPU NCCL topology to CFG2.

    A caller-provided degree is always authoritative.  Automatic CFG2 is kept
    to the resident topology that the production profiles validate; uncommon
    TP/FSDP layouts must continue to state their CFG degree explicitly.
    """
    if requested is not None:
        return requested
    if backend != 'nccl':
        return 1
    if tp_degree != 1 or dit_fsdp_shard_degree != 1 or dit_fsdp_replicate_degree != 1:
        return 1
    import torch
    device_count = (
        len(parallel_devices)
        if parallel_devices is not None
        else torch.cuda.device_count()
    )
    return 2 if device_count >= 2 * int(sp_degree) else 1


@dataclass(frozen=True)
class DiTTopology:
    tp: int = 1
    ulysses: int = 1
    ring: int = 1
    cfg: int = 1
    replicas: int = 1

    def __post_init__(self):
        if any(type(n) is not int or n < 1 for n in self.sizes):
            raise ValueError('parallel degrees must be positive integers')
        if self.cfg not in (1, 2):
            raise ValueError('CFG degree must be 1 or 2')

    @property
    def sizes(self):
        return (self.tp, self.ulysses, self.ring, self.cfg, self.replicas)

    @property
    def world_size(self):
        import math
        return math.prod(self.sizes)

    @property
    def sp(self):
        return self.ulysses * self.ring

    def coordinates(self, rank):
        if not 0 <= rank < self.world_size:
            raise ValueError('rank outside topology')
        values = []
        for size in self.sizes:
            values.append(rank % size)
            rank //= size
        return tuple(values)

    def rank(self, coordinates):
        value, stride = 0, 1
        for index, size in zip(coordinates, self.sizes):
            if not 0 <= index < size:
                raise ValueError('coordinate outside topology')
            value += index * stride
            stride *= size
        return value

    def groups(self, axis):
        axes = {'tp': (0,), 'ulysses': (1,), 'ring': (2,), 'sp': (1, 2), 'cfg': (3,)}[axis]
        groups = {}
        for rank in range(self.world_size):
            coords = self.coordinates(rank)
            key = tuple(c for i, c in enumerate(coords) if i not in axes)
            groups.setdefault(key, []).append(rank)
        return tuple(tuple(ranks) for ranks in groups.values())


def resolve_dit_topology(config):
    sp = config.sp_degree
    if type(sp) is not int or sp < 1:
        raise ValueError('sp_degree must be a positive integer')
    ring = getattr(config, 'ring_degree', 1)
    ulysses = getattr(config, 'ulysses_degree', None)
    if ulysses is None:
        if config.sp_attention_mode == 'ring' and ring == 1:
            ring = sp
        if type(ring) is not int or ring < 1 or sp % ring:
            raise ValueError('sp_degree must be divisible by ring_degree')
        ulysses = sp // ring
    if sp != ulysses * ring:
        raise ValueError('sp_degree must equal ulysses_degree * ring_degree')
    tp = getattr(config, 'tp_degree', 1)
    base = DiTTopology(tp=tp, ulysses=ulysses, ring=ring, cfg=config.cfg_degree)
    shard = getattr(config, 'dit_fsdp_shard_degree', 1)
    replicate = getattr(config, 'dit_fsdp_replicate_degree', 1)
    if any(type(v) is not int or v < 1 for v in (shard, replicate)):
        raise ValueError('FSDP degrees must be positive integers')
    fsdp = shard * replicate
    if fsdp > 1:
        if tp != 1:
            raise ValueError('FSDP and TP cannot manage the same weights')
        if fsdp % base.world_size:
            raise ValueError('FSDP mesh size must be a multiple of CFG * SP')
    return DiTTopology(tp=tp, ulysses=ulysses, ring=ring, cfg=config.cfg_degree,
                       replicas=max(1, fsdp // base.world_size))
