"""Parallel configuration and validated execution plan; no runtime policy imports."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ParallelMode(str, Enum):
    DISABLED = "disabled"
    AUTO = "auto"
    MANUAL = "manual"


@dataclass(frozen=True)
class AccelerationConfig:
    parallel_mode: ParallelMode = ParallelMode.DISABLED
    sp_degree: int = 0
    cfg_degree: int = 0
    vae_degree: int = 0
    writer_rank: int = 0


@dataclass(frozen=True)
class ResolvedAccelerationPlan:
    enabled: bool
    world_size: int
    sp_degree: int
    cfg_degree: int
    vae_degree: int
    writer_rank: int

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a bool")
        if type(self.world_size) is not int:
            raise TypeError("world_size must be an int")
        if self.world_size not in {1, 2, 4}:
            raise ValueError("world_size must be one of 1, 2, 4")

        degrees = (self.sp_degree, self.cfg_degree, self.vae_degree)
        if any(type(degree) is not int for degree in degrees):
            raise TypeError("parallel degrees must be integers")
        if any(degree < 1 for degree in degrees):
            raise ValueError("parallel degrees must be positive")
        if type(self.writer_rank) is not int:
            raise TypeError("writer_rank must be an int")
        if not 0 <= self.writer_rank < self.world_size:
            raise ValueError("writer_rank must be inside world_size")

        if not self.enabled:
            if degrees != (1, 1, 1):
                raise ValueError("disabled plan requires all degrees to be 1")
            return
        if self.world_size == 1:
            raise ValueError("enabled plan requires world_size greater than 1")
        if self.sp_degree * self.cfg_degree != self.world_size:
            raise ValueError("sp_degree * cfg_degree must equal world_size")
        if self.vae_degree not in {1, self.world_size}:
            raise ValueError("vae_degree must be 1 or equal world_size in P1")
