from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from nndeploy.e_dit.distributed.parallel_groups import (
    ParallelGroupSpec,
    ParallelTopology,
    build_parallel_topology,
)
from nndeploy.e_dit.config.parallel import ResolvedAccelerationPlan


@dataclass
class RuntimeGroup:
    spec: ParallelGroupSpec
    process_group: Any
    group_rank: int
    owns_process_group: bool = True

    @property
    def is_member(self) -> bool:
        return self.group_rank >= 0

    @property
    def global_rank(self) -> int:
        if not self.is_member:
            raise RuntimeError(f"rank is not a member of group {self.spec.name}")
        return self.spec.ranks[self.group_rank]

    @property
    def world_size(self) -> int:
        return len(self.spec.ranks)


@dataclass
class ParallelContext:
    plan: ResolvedAccelerationPlan
    global_rank: int
    local_rank: int
    topology: ParallelTopology | None
    groups: dict[str, RuntimeGroup]
    control_process_group: Any | None = None

    @property
    def enabled(self) -> bool:
        return self.plan.enabled

    def current_group(self, prefix: str) -> RuntimeGroup:
        matches = [
            group
            for name, group in self.groups.items()
            if name.startswith(prefix) and group.is_member
        ]
        if len(matches) != 1:
            raise RuntimeError(f"expected one member group for prefix={prefix}")
        return matches[0]

    def world_data_group(self) -> RuntimeGroup:
        if not self.enabled or self.topology is None:
            raise RuntimeError("world runtime group requires active parallel context")
        return RuntimeGroup(
            spec=self.topology.world_group,
            process_group=None,
            group_rank=self.global_rank,
            owns_process_group=False,
        )

    def world_control_group(self) -> RuntimeGroup:
        if not self.enabled or self.topology is None:
            raise RuntimeError("world runtime group requires active parallel context")
        if self.control_process_group is None:
            raise RuntimeError("world control process group is not initialized")
        return RuntimeGroup(
            spec=self.topology.world_group,
            process_group=self.control_process_group,
            group_rank=self.global_rank,
            owns_process_group=False,
        )

    def destroy(self) -> None:
        first_error = _destroy_member_groups(self.groups)
        if first_error is not None:
            raise first_error


def resolve_group_control_device(
    group: RuntimeGroup,
    *,
    local_rank: int,
    fallback_device: torch.device | str,
) -> torch.device:
    """Resolve a collective-safe device without leaking backend checks upward."""
    if dist.is_available() and dist.is_initialized():
        backend = str(dist.get_backend(group.process_group)).lower()
        if backend == "nccl":
            return torch.device("cuda", local_rank)
        return torch.device("cpu")

    requested_device = torch.device(fallback_device)
    if requested_device.type == "cuda":
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


_PARALLEL_CONTEXT: ParallelContext | None = None


def initialize_parallel_context(
    plan: ResolvedAccelerationPlan,
    *,
    global_rank: int,
    local_rank: int,
) -> ParallelContext:
    if not isinstance(plan, ResolvedAccelerationPlan):
        raise TypeError("plan must be a ResolvedAccelerationPlan")
    _validate_rank("global_rank", global_rank, world_size=plan.world_size)
    _validate_rank("local_rank", local_rank, world_size=plan.world_size)

    if not plan.enabled:
        return ParallelContext(plan, global_rank, local_rank, None, {})
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "parallel context requires an initialized default process group"
        )
    if dist.get_world_size() != plan.world_size:
        raise RuntimeError("default process-group world size does not match plan")
    if dist.get_rank() != global_rank:
        raise RuntimeError("default process-group rank does not match global_rank")

    topology = build_parallel_topology(
        world_size=plan.world_size,
        sp_degree=plan.sp_degree,
        cfg_degree=plan.cfg_degree,
        vae_degree=plan.vae_degree,
    )
    groups: dict[str, RuntimeGroup] = {}
    try:
        for spec in topology.ordered_subgroups():
            owns_process_group = spec.ranks != topology.world_group.ranks
            process_group = (
                dist.new_group(ranks=list(spec.ranks))
                if owns_process_group
                else None
            )
            group_rank = (
                spec.ranks.index(global_rank) if global_rank in spec.ranks else -1
            )
            groups[spec.name] = RuntimeGroup(
                spec,
                process_group,
                group_rank,
                owns_process_group=owns_process_group,
            )
    except BaseException:
        _destroy_member_groups(groups)
        raise
    return ParallelContext(plan, global_rank, local_rank, topology, groups)


def set_parallel_context(context: ParallelContext | None) -> None:
    global _PARALLEL_CONTEXT
    _PARALLEL_CONTEXT = context


def get_parallel_context() -> ParallelContext | None:
    return _PARALLEL_CONTEXT


def _destroy_member_groups(
    groups: dict[str, RuntimeGroup],
) -> BaseException | None:
    first_error: BaseException | None = None
    for group in reversed(tuple(groups.values())):
        if not group.is_member or not group.owns_process_group:
            continue
        try:
            dist.destroy_process_group(group.process_group)
        except BaseException as error:
            if first_error is None:
                first_error = error
    groups.clear()
    return first_error


def _validate_rank(name: str, rank: int, *, world_size: int) -> None:
    if type(rank) is not int:
        raise TypeError(f"{name} must be an int")
    if not 0 <= rank < world_size:
        raise ValueError(f"{name} must be inside world_size")
