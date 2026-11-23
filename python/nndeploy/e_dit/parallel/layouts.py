from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TensorLayoutKind(str, Enum):
    REPLICATED = "replicated"
    SEQUENCE_SHARDED = "sequence_sharded"
    SPATIAL_SHARDED = "spatial_sharded"
    BRANCH_SHARDED = "branch_sharded"
    OWNER_ONLY = "owner_only"


@dataclass(frozen=True)
class ShardMetadata:
    shard_dim: int
    shard_index: int
    shard_count: int
    offset: int
    valid_length: int
    padded_length: int
    owner_rank: int | None = None

    def __post_init__(self) -> None:
        for field in (
            "shard_dim",
            "shard_index",
            "shard_count",
            "offset",
            "valid_length",
            "padded_length",
        ):
            if type(getattr(self, field)) is not int:
                raise TypeError(f"{field} must be an int")
        if self.owner_rank is not None and type(self.owner_rank) is not int:
            raise TypeError("owner_rank must be an int or None")

        if self.shard_count < 1:
            raise ValueError("shard_count must be positive")
        if not 0 <= self.shard_index < self.shard_count:
            raise ValueError("shard_index must be inside shard_count")
        if self.offset < 0 or self.valid_length < 0:
            raise ValueError("offset and valid_length must be non-negative")
        if self.padded_length < self.valid_length:
            raise ValueError("padded_length must cover valid_length")


@dataclass(frozen=True)
class TensorLayout:
    kind: TensorLayoutKind
    global_shape: tuple[int, ...]
    local_shape: tuple[int, ...]
    shard: ShardMetadata | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, TensorLayoutKind):
            raise ValueError("kind must be a TensorLayoutKind")
        for field in ("global_shape", "local_shape"):
            shape = getattr(self, field)
            if type(shape) is not tuple:
                raise TypeError(f"{field} must be a tuple")
            if any(type(dimension) is not int for dimension in shape):
                raise TypeError(f"{field} dimensions must be integers")
        if len(self.global_shape) != len(self.local_shape):
            raise ValueError("global_shape and local_shape must have equal rank")
        if any(dimension < 0 for dimension in self.global_shape + self.local_shape):
            raise ValueError("shape dimensions must be non-negative")

        if self.kind is TensorLayoutKind.REPLICATED:
            if self.shard is not None:
                raise ValueError("replicated layout cannot carry shard metadata")
            if self.global_shape != self.local_shape:
                raise ValueError("replicated layout requires identical shapes")
            return

        if self.shard is None:
            raise ValueError("sharded/owner layout requires shard metadata")
        if not isinstance(self.shard, ShardMetadata):
            raise TypeError("shard must be a ShardMetadata instance")
        if not (
            0 <= self.shard.shard_dim < len(self.global_shape)
            and self.shard.shard_dim < len(self.local_shape)
        ):
            raise ValueError("shard_dim must be inside global and local shapes")

        shard_dim = self.shard.shard_dim
        if any(
            local_dimension != global_dimension
            for dimension, (global_dimension, local_dimension) in enumerate(
                zip(self.global_shape, self.local_shape)
            )
            if dimension != shard_dim
        ):
            raise ValueError("non-sharded dimensions must match global_shape")
        if self.local_shape[shard_dim] != self.shard.padded_length:
            raise ValueError("local shard dimension must equal padded_length")
        if self.shard.offset + self.shard.valid_length > self.global_shape[shard_dim]:
            raise ValueError("shard valid range must fit global_shape")

        if self.kind is TensorLayoutKind.OWNER_ONLY:
            if self.shard.owner_rank is None or self.shard.owner_rank < 0:
                raise ValueError("owner-only layout requires non-negative owner_rank")
        elif self.shard.owner_rank is not None:
            raise ValueError("owner_rank is only valid for owner-only layout")

    @classmethod
    def replicated(cls, shape: tuple[int, ...]) -> "TensorLayout":
        return cls(
            kind=TensorLayoutKind.REPLICATED,
            global_shape=tuple(shape),
            local_shape=tuple(shape),
        )
