from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from nndeploy.e_dit.distributed.group_coordinator import GroupCoordinator


@dataclass(frozen=True)
class SequenceShardPlan:
    """One rank's equal-size slice of a right-padded sequence."""

    global_length: int
    padded_length: int
    local_start: int
    local_end: int
    valid_local_length: int
    pad_right: int
    owner_rank: int
    sp_degree: int
    rank: int

    def __post_init__(self) -> None:
        for field in (
            "global_length",
            "padded_length",
            "local_start",
            "local_end",
            "valid_local_length",
            "pad_right",
            "owner_rank",
            "sp_degree",
            "rank",
        ):
            if type(getattr(self, field)) is not int:
                raise TypeError(f"{field} must be an int")

        if self.global_length < 0:
            raise ValueError("global_length must be non-negative")
        if self.sp_degree < 1:
            raise ValueError("sp_degree must be positive")
        if not 0 <= self.rank < self.sp_degree:
            raise ValueError("rank must be inside sp_degree")
        if not 0 <= self.owner_rank < self.sp_degree:
            raise ValueError("owner_rank must be inside sp_degree")
        if self.padded_length < self.global_length:
            raise ValueError("padded_length must cover global_length")
        if self.padded_length % self.sp_degree != 0:
            raise ValueError("padded_length must be divisible by sp_degree")
        if not 0 <= self.local_start <= self.local_end <= self.padded_length:
            raise ValueError("local range must fit padded_length")
        if self.local_start != self.rank * self.local_length:
            raise ValueError("local_start must match rank order")
        if self.local_length != self.padded_length // self.sp_degree:
            raise ValueError("all ranks must use an equal local_length")
        if not 0 <= self.valid_local_length <= self.local_length:
            raise ValueError("valid_local_length must fit local_length")
        if self.pad_right != self.local_length - self.valid_local_length:
            raise ValueError("pad_right must cover the invalid local suffix")
        expected_valid_length = max(
            0,
            min(self.local_length, self.global_length - self.local_start),
        )
        if self.valid_local_length != expected_valid_length:
            raise ValueError("valid_local_length must match the global range")

    @property
    def local_length(self) -> int:
        return self.local_end - self.local_start


def plan_sequence_shards(
    global_length: int,
    sp_degree: int,
    owner_rank: int,
) -> tuple[SequenceShardPlan, ...]:
    """Create immutable, rank-ordered metadata for equal sequence shards."""

    for name, value in (
        ("global_length", global_length),
        ("sp_degree", sp_degree),
        ("owner_rank", owner_rank),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} must be an int")
    if global_length < 0:
        raise ValueError("global_length must be non-negative")
    if sp_degree < 1:
        raise ValueError("sp_degree must be positive")
    if not 0 <= owner_rank < sp_degree:
        raise ValueError("owner_rank must be inside sp_degree")

    local_length = (global_length + sp_degree - 1) // sp_degree
    padded_length = local_length * sp_degree
    plans = []
    for rank in range(sp_degree):
        local_start = rank * local_length
        local_end = local_start + local_length
        valid_local_length = max(
            0,
            min(local_length, global_length - local_start),
        )
        plans.append(
            SequenceShardPlan(
                global_length=global_length,
                padded_length=padded_length,
                local_start=local_start,
                local_end=local_end,
                valid_local_length=valid_local_length,
                pad_right=local_length - valid_local_length,
                owner_rank=owner_rank,
                sp_degree=sp_degree,
                rank=rank,
            )
        )
    return tuple(plans)


def shard_sequence_tensor(
    tensor: torch.Tensor,
    plan: SequenceShardPlan,
    coordinator: GroupCoordinator,
    sequence_dim: int,
) -> torch.Tensor:
    """Scatter an owner tensor into equal local sequence shards.

    The owner supplies the unpadded global tensor. Every non-owner supplies only
    a local tensor template whose values are ignored; the template communicates
    the non-sequence shape, dtype, and device without allocating a full tensor.
    """

    _validate_coordinator(plan, coordinator)
    normalized_dim = _normalize_sequence_dim(sequence_dim, tensor)
    is_owner = coordinator.rank == plan.owner_rank
    actual_length = tensor.shape[normalized_dim]
    expected_length = plan.global_length if is_owner else plan.local_length
    if actual_length != expected_length:
        label = "global_length" if is_owner else "local_length"
        raise ValueError(
            f"tensor sequence length must equal plan {label} "
            f"({expected_length}), got {actual_length}"
        )

    local_shape = list(tensor.shape)
    local_shape[normalized_dim] = plan.local_length
    output = torch.empty(
        local_shape,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    scatter_list: list[torch.Tensor] | None = None
    if is_owner:
        padded_shape = list(tensor.shape)
        padded_shape[normalized_dim] = plan.padded_length
        padded = tensor.new_zeros(padded_shape)
        if plan.global_length:
            padded.narrow(
                normalized_dim,
                0,
                plan.global_length,
            ).copy_(tensor)
        if plan.local_length:
            scatter_list = list(
                torch.split(padded, plan.local_length, dim=normalized_dim)
            )
        else:
            empty = padded.narrow(normalized_dim, 0, 0)
            scatter_list = [empty for _ in range(plan.sp_degree)]

    coordinator.scatter(
        output,
        scatter_list,
        src=_owner_global_rank(plan, coordinator),
    )
    return output


def gather_sequence_to_owner(
    tensor: torch.Tensor,
    plan: SequenceShardPlan,
    coordinator: GroupCoordinator,
    sequence_dim: int,
) -> torch.Tensor | None:
    """Gather rank-ordered shards to owner and trim right-side padding."""

    _validate_coordinator(plan, coordinator)
    normalized_dim = _normalize_sequence_dim(sequence_dim, tensor)
    actual_length = tensor.shape[normalized_dim]
    if actual_length != plan.local_length:
        raise ValueError(
            "tensor sequence length must equal plan local_length "
            f"({plan.local_length}), got {actual_length}"
        )

    is_owner = coordinator.rank == plan.owner_rank
    gather_list = None
    if is_owner:
        gather_list = [
            torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
            for _ in range(plan.sp_degree)
        ]
    gathered = coordinator.gather_to_owner(
        tensor,
        gather_list,
        dst=_owner_global_rank(plan, coordinator),
    )
    if not is_owner:
        return None
    if gathered is None or len(gathered) != plan.sp_degree:
        raise RuntimeError("owner gather did not return every sequence shard")

    padded = torch.cat(gathered, dim=normalized_dim)
    if padded.shape[normalized_dim] != plan.padded_length:
        raise RuntimeError("gathered sequence length does not match padded_length")
    return padded.narrow(normalized_dim, 0, plan.global_length)


def _validate_coordinator(
    plan: SequenceShardPlan,
    coordinator: GroupCoordinator,
) -> None:
    if not isinstance(plan, SequenceShardPlan):
        raise TypeError("plan must be a SequenceShardPlan")
    if coordinator.world_size != plan.sp_degree:
        raise ValueError(
            "coordinator world_size must equal plan sp_degree "
            f"({plan.sp_degree}), got {coordinator.world_size}"
        )
    if coordinator.rank != plan.rank:
        raise ValueError(
            f"coordinator rank must equal plan rank ({plan.rank}), "
            f"got {coordinator.rank}"
        )
    group_ranks = coordinator.group.spec.ranks
    if len(group_ranks) != plan.sp_degree:
        raise ValueError("coordinator group ranks must equal plan sp_degree")


def _normalize_sequence_dim(sequence_dim: int, tensor: torch.Tensor) -> int:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    if type(sequence_dim) is not int:
        raise TypeError("sequence_dim must be an int")
    if not -tensor.ndim <= sequence_dim < tensor.ndim:
        raise IndexError(
            f"sequence_dim {sequence_dim} is invalid for tensor rank {tensor.ndim}"
        )
    return sequence_dim % tensor.ndim


def _owner_global_rank(
    plan: SequenceShardPlan,
    coordinator: GroupCoordinator,
) -> int:
    return coordinator.group.spec.ranks[plan.owner_rank]


__all__ = (
    "SequenceShardPlan",
    "gather_sequence_to_owner",
    "plan_sequence_shards",
    "shard_sequence_tensor",
)
