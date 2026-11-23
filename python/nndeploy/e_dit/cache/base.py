"""Common lifecycle and execution identity for Transformer caches."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CacheBranch(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass(frozen=True)
class CacheExecutionContext:
    request_id: str
    object_index: int
    window_index: int
    branch: CacheBranch
    step_index: int
    total_steps: int
    global_sequence_length: int
    local_sequence_length: int
    valid_local_sequence_length: int
    sp_degree: int
    sp_rank: int
    cfg_degree: int
    cfg_rank: int
    model_identity: str
    layout_signature: str
    sp_group_identity: str
    cfg_group_identity: str
    dtype: str
    device: str
    hidden_width: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("cache execution request_id must not be empty")
        for name in (
            "object_index",
            "window_index",
            "branch",
            "step_index",
            "total_steps",
            "global_sequence_length",
            "local_sequence_length",
            "valid_local_sequence_length",
            "sp_degree",
            "sp_rank",
            "cfg_degree",
            "cfg_rank",
            "hidden_width",
        ):
            if name == "branch":
                continue
            if type(getattr(self, name)) is not int:
                raise TypeError(f"cache execution {name} must be a non-bool int")
        if not isinstance(self.branch, CacheBranch):
            raise TypeError("cache execution branch must be a CacheBranch")
        if self.object_index < 0 or self.window_index < 0 or self.step_index < 0:
            raise ValueError("cache execution indices must be non-negative")
        if self.total_steps <= 0:
            raise ValueError("cache execution total_steps must be positive")
        if not 0 <= self.step_index < self.total_steps:
            raise ValueError("cache execution step_index must be inside total_steps")
        if self.global_sequence_length <= 0 or self.local_sequence_length <= 0:
            raise ValueError("cache execution sequence lengths must be positive")
        if not 0 < self.valid_local_sequence_length <= self.local_sequence_length:
            raise ValueError(
                "cache execution valid_local_sequence_length must fit local length"
            )
        if self.sp_degree < 1 or not 0 <= self.sp_rank < self.sp_degree:
            raise ValueError("cache execution SP identity is invalid")
        if self.cfg_degree not in {1, 2} or not 0 <= self.cfg_rank < self.cfg_degree:
            raise ValueError("cache execution CFG identity is invalid")
        if self.hidden_width <= 0:
            raise ValueError("cache execution hidden_width must be positive")
        if not all(
            (
                self.model_identity,
                self.layout_signature,
                self.sp_group_identity,
                self.cfg_group_identity,
                self.dtype,
                self.device,
            )
        ):
            raise ValueError("cache execution model/layout identity must not be empty")


__all__ = ("CacheBranch", "CacheExecutionContext")
