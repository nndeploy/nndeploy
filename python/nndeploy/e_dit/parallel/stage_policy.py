"""Stage participation policy and cross-rank error synchronization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from nndeploy.e_dit.distributed.parallel_state import ParallelContext


class StageParticipation(str, Enum):
    REPLICATED = "replicated"
    GROUP = "group"
    OWNER_ONLY = "owner_only"


@dataclass(frozen=True)
class StageExecutionPolicy:
    participation: StageParticipation
    group_name: str | None = None
    owner_rank: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.participation, StageParticipation):
            raise TypeError("participation must be a StageParticipation")

        if self.participation is StageParticipation.REPLICATED:
            if self.group_name is not None or self.owner_rank is not None:
                raise ValueError("replicated policy cannot carry group/owner")
            return

        if self.participation is StageParticipation.GROUP:
            if type(self.group_name) is not str:
                raise TypeError("group_name must be a str")
            if not self.group_name.strip() or self.owner_rank is not None:
                raise ValueError("group policy requires only a non-empty group_name")
            return

        if type(self.owner_rank) is not int:
            raise TypeError("owner_rank must be an int")
        if self.owner_rank < 0 or self.group_name is not None:
            raise ValueError(
                "owner-only policy requires only a non-negative owner_rank"
            )

    @classmethod
    def replicated(cls) -> StageExecutionPolicy:
        return cls(StageParticipation.REPLICATED)

    @classmethod
    def group(cls, group_name: str) -> StageExecutionPolicy:
        return cls(StageParticipation.GROUP, group_name=group_name)

    @classmethod
    def owner_only(cls, owner_rank: int) -> StageExecutionPolicy:
        return cls(StageParticipation.OWNER_ONLY, owner_rank=owner_rank)

    def validate_context(
        self,
        *,
        group_names: set[str],
        world_size: int,
    ) -> None:
        if type(group_names) is not set or any(
            type(name) is not str for name in group_names
        ):
            raise TypeError("group_names must be a set of strings")
        if type(world_size) is not int:
            raise TypeError("world_size must be an int")
        if world_size <= 0:
            raise ValueError("world_size must be positive")

        if (
            self.participation is StageParticipation.GROUP
            and self.group_name not in group_names
        ):
            raise ValueError(
                f"group policy references unknown group {self.group_name!r}"
            )
        if (
            self.participation is StageParticipation.OWNER_ONLY
            and self.owner_rank >= world_size
        ):
            raise ValueError(
                "owner-only policy owner_rank must be less than world_size"
            )

    def should_execute(self, *, global_rank: int, group_names: set[str]) -> bool:
        if type(global_rank) is not int:
            raise TypeError("global_rank must be an int")
        if global_rank < 0:
            raise ValueError("global_rank must be non-negative")
        if type(group_names) is not set or any(
            type(name) is not str for name in group_names
        ):
            raise TypeError("group_names must be a set of strings")

        if self.participation is StageParticipation.REPLICATED:
            return True
        if self.participation is StageParticipation.OWNER_ONLY:
            return global_rank == self.owner_rank
        return self.group_name in group_names


class DistributedStageError(RuntimeError):
    """A peer rank failed while executing the current stage."""


def synchronize_stage_bool(
    value: bool,
    context: ParallelContext | None,
    *,
    field_name: str,
) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be a bool")
    if context is None or not context.enabled:
        return value
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("stage bool synchronization requires torch.distributed")

    control_group = context.control_process_group
    backend = str(
        dist.get_backend(control_group)
        if control_group is not None
        else dist.get_backend()
    ).lower()
    device = (
        torch.device("cuda", context.local_rank)
        if backend == "nccl"
        else torch.device("cpu")
    )
    enabled_count = torch.tensor(
        [int(value)],
        dtype=torch.int64,
        device=device,
    )
    if control_group is not None:
        dist.all_reduce(
            enabled_count,
            op=dist.ReduceOp.SUM,
            group=control_group,
        )
    else:
        dist.all_reduce(enabled_count, op=dist.ReduceOp.SUM)
    count = int(enabled_count.item())
    if count == 0:
        return False
    if count == context.plan.world_size:
        return True
    raise DistributedStageError(
        f"{field_name} differs across ranks: "
        f"enabled={count}, world_size={context.plan.world_size}"
    )


def synchronize_stage_error(
    error: Exception | None,
    context: ParallelContext | None,
) -> None:
    """Make every enabled rank observe a Python error at the stage boundary."""

    if context is None or not context.enabled:
        if error is not None:
            raise error
        return
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("stage error synchronization requires torch.distributed")

    control_group = context.control_process_group
    backend = str(
        dist.get_backend(control_group)
        if control_group is not None
        else dist.get_backend()
    ).lower()
    device = (
        torch.device("cuda", context.local_rank)
        if backend == "nccl"
        else torch.device("cpu")
    )
    sentinel = context.global_rank if error is not None else context.plan.world_size
    failed_rank = torch.tensor([sentinel], dtype=torch.int64, device=device)
    if control_group is not None:
        dist.all_reduce(
            failed_rank,
            op=dist.ReduceOp.MIN,
            group=control_group,
        )
    else:
        dist.all_reduce(failed_rank, op=dist.ReduceOp.MIN)
    first_failed_rank = int(failed_rank.item())
    if first_failed_rank == context.plan.world_size:
        return
    if error is not None:
        raise error
    raise DistributedStageError(f"stage failed on peer rank {first_failed_rank}")


__all__ = (
    "DistributedStageError",
    "StageExecutionPolicy",
    "StageParticipation",
    "synchronize_stage_bool",
    "synchronize_stage_error",
)
