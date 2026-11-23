from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParallelGroupSpec:
    name: str
    ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.name) is not str:
            raise TypeError("group name must be a string")
        if not self.name:
            raise ValueError("group name must not be empty")
        if type(self.ranks) is not tuple:
            raise TypeError("group ranks must be a tuple")
        if not self.ranks:
            raise ValueError("group ranks must not be empty")
        if any(type(rank) is not int for rank in self.ranks):
            raise TypeError("group ranks must contain integers")
        if any(rank < 0 for rank in self.ranks):
            raise ValueError("group ranks must be non-negative")
        if len(set(self.ranks)) != len(self.ranks):
            raise ValueError("group ranks must not contain duplicates")


@dataclass(frozen=True)
class ParallelTopology:
    world_group: ParallelGroupSpec
    sp_groups: tuple[ParallelGroupSpec, ...]
    cfg_groups: tuple[ParallelGroupSpec, ...]
    vae_group: ParallelGroupSpec

    def __post_init__(self) -> None:
        if not isinstance(self.world_group, ParallelGroupSpec):
            raise TypeError("world_group must be a ParallelGroupSpec")
        if type(self.sp_groups) is not tuple:
            raise TypeError("sp_groups must be a tuple")
        if type(self.cfg_groups) is not tuple:
            raise TypeError("cfg_groups must be a tuple")
        if not isinstance(self.vae_group, ParallelGroupSpec):
            raise TypeError("vae_group must be a ParallelGroupSpec")
        if any(not isinstance(group, ParallelGroupSpec) for group in self.sp_groups):
            raise TypeError("sp_groups must contain ParallelGroupSpec values")
        if any(not isinstance(group, ParallelGroupSpec) for group in self.cfg_groups):
            raise TypeError("cfg_groups must contain ParallelGroupSpec values")

        if self.world_group.name != "world":
            raise ValueError("world group name must be 'world'")
        if self.vae_group.name != "vae":
            raise ValueError("vae group name must be 'vae'")
        expected_world_ranks = tuple(range(len(self.world_group.ranks)))
        if self.world_group.ranks != expected_world_ranks:
            raise ValueError("world ranks must be contiguous from zero")

        expected_sp_names = tuple(f"sp_{index}" for index in range(len(self.sp_groups)))
        if tuple(group.name for group in self.sp_groups) != expected_sp_names:
            raise ValueError("SP group names must be ordered as sp_0, sp_1, ...")
        expected_cfg_names = tuple(
            f"cfg_{index}" for index in range(len(self.cfg_groups))
        )
        if tuple(group.name for group in self.cfg_groups) != expected_cfg_names:
            raise ValueError("CFG group names must be ordered as cfg_0, cfg_1, ...")

        world_size = len(expected_world_ranks)
        sp_degree = len(self.cfg_groups)
        cfg_degree = len(self.sp_groups)
        if sp_degree * cfg_degree != world_size:
            raise ValueError("sp_degree * cfg_degree must equal world_size")
        expected_sp_groups = tuple(
            ParallelGroupSpec(
                name=f"sp_{branch}",
                ranks=tuple(range(branch * sp_degree, (branch + 1) * sp_degree)),
            )
            for branch in range(cfg_degree)
        )
        if self.sp_groups != expected_sp_groups:
            raise ValueError("SP groups must match the deterministic rectangular mesh")
        expected_cfg_groups = tuple(
            ParallelGroupSpec(
                name=f"cfg_{shard}",
                ranks=tuple(branch * sp_degree + shard for branch in range(cfg_degree)),
            )
            for shard in range(sp_degree)
        )
        if self.cfg_groups != expected_cfg_groups:
            raise ValueError("CFG groups must match the deterministic rectangular mesh")

        group_names = (
            self.world_group.name,
            *(group.name for group in self.sp_groups),
            *(group.name for group in self.cfg_groups),
            self.vae_group.name,
        )
        if len(set(group_names)) != len(group_names):
            raise ValueError("parallel group names must be unique")

        sp_ranks = sorted(rank for group in self.sp_groups for rank in group.ranks)
        if sp_ranks != list(expected_world_ranks):
            raise ValueError("SP groups must cover every world rank exactly once")
        cfg_ranks = sorted(rank for group in self.cfg_groups for rank in group.ranks)
        if cfg_ranks != list(expected_world_ranks):
            raise ValueError("CFG groups must cover every world rank exactly once")
        if self.vae_group.ranks not in {(0,), expected_world_ranks}:
            raise ValueError("VAE group must contain rank 0 or every world rank")

    def ordered_subgroups(self) -> tuple[ParallelGroupSpec, ...]:
        return self.sp_groups + self.cfg_groups + (self.vae_group,)


def build_parallel_topology(
    *,
    world_size: int,
    sp_degree: int,
    cfg_degree: int,
    vae_degree: int,
) -> ParallelTopology:
    if type(world_size) is not int:
        raise TypeError("world_size must be an int")
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if any(type(degree) is not int for degree in (sp_degree, cfg_degree, vae_degree)):
        raise TypeError("parallel degrees must be integers")
    if sp_degree < 1 or cfg_degree < 1 or vae_degree < 1:
        raise ValueError("parallel degrees must be positive")
    if sp_degree * cfg_degree != world_size:
        raise ValueError("sp_degree * cfg_degree must equal world_size")
    if vae_degree not in {1, world_size}:
        raise ValueError("vae_degree must be 1 or equal world_size in P1")

    sp_groups = tuple(
        ParallelGroupSpec(
            name=f"sp_{branch}",
            ranks=tuple(range(branch * sp_degree, (branch + 1) * sp_degree)),
        )
        for branch in range(cfg_degree)
    )
    cfg_groups = tuple(
        ParallelGroupSpec(
            name=f"cfg_{shard}",
            ranks=tuple(branch * sp_degree + shard for branch in range(cfg_degree)),
        )
        for shard in range(sp_degree)
    )
    expected_ranks = list(range(world_size))
    if sorted(rank for group in sp_groups for rank in group.ranks) != expected_ranks:
        raise RuntimeError("SP groups must cover every world rank exactly once")
    if sorted(rank for group in cfg_groups for rank in group.ranks) != expected_ranks:
        raise RuntimeError("CFG groups must cover every world rank exactly once")

    return ParallelTopology(
        world_group=ParallelGroupSpec("world", tuple(expected_ranks)),
        sp_groups=sp_groups,
        cfg_groups=cfg_groups,
        vae_group=ParallelGroupSpec("vae", tuple(range(vae_degree))),
    )
