"""Shared TeaCache policy and request validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class TeaCacheCoefficientSelection:
    policy: str
    model_identity: str
    requested_global_sequence_length: int
    selected_fit_length: int
    fit_length_distance: int
    complexity: int
    calibration_size: int
    coefficients: tuple[float, ...]


@dataclass(frozen=True)
class TeaCacheParams:
    supported_coefficient_policies: ClassVar[tuple[str, ...]] = ()
    enabled: bool
    threshold: float = 0.3
    max_consecutive_skip: int = 1
    min_skip_step: int = 2
    end_guard_steps: int = 1
    coefficient_policy: str = ""
    calibrate: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("TeaCache enabled must be a bool")
        if isinstance(self.threshold, bool) or not isinstance(
            self.threshold, (int, float)
        ):
            raise TypeError("TeaCache threshold must be a finite float")
        threshold = float(self.threshold)
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("TeaCache threshold must be finite and > 0")
        object.__setattr__(self, "threshold", threshold)
        for name in ("max_consecutive_skip", "min_skip_step", "end_guard_steps"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"TeaCache {name} must be a non-bool int")
        if self.max_consecutive_skip < 1:
            raise ValueError("TeaCache max_consecutive_skip must be >= 1")
        if self.min_skip_step < 0:
            raise ValueError("TeaCache min_skip_step must be >= 0")
        if self.end_guard_steps < 0:
            raise ValueError("TeaCache end_guard_steps must be >= 0")
        if self.coefficient_policy not in self.supported_coefficient_policies:
            raise ValueError(
                "unsupported TeaCache coefficient_policy="
                f"{self.coefficient_policy!r}"
            )
        if not isinstance(self.calibrate, bool):
            raise TypeError("TeaCache calibrate must be a bool")
