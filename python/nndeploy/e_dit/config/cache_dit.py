"""Project-owned Cache-DiT DBCache request contract for diffusion inference."""

from __future__ import annotations

import math
from dataclasses import dataclass




@dataclass(frozen=True)
class CacheDitParams:
    enabled: bool
    front_blocks: int = 1
    back_blocks: int = 0
    warmup_steps: int = 4
    residual_diff_threshold: float = 0.3
    max_consecutive_cached_steps: int = 3
    end_guard_steps: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("Cache-DiT enabled must be a bool")
        for name in (
            "front_blocks",
            "back_blocks",
            "warmup_steps",
            "max_consecutive_cached_steps",
            "end_guard_steps",
        ):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"Cache-DiT {name} must be a non-bool int")
        if self.front_blocks < 1:
            raise ValueError("Cache-DiT front_blocks must be >= 1")
        if self.back_blocks < 0:
            raise ValueError("Cache-DiT back_blocks must be >= 0")
        if self.warmup_steps < 0:
            raise ValueError("Cache-DiT warmup_steps must be >= 0")
        if self.max_consecutive_cached_steps < 1:
            raise ValueError(
                "Cache-DiT max_consecutive_cached_steps must be >= 1"
            )
        if self.end_guard_steps < 0:
            raise ValueError("Cache-DiT end_guard_steps must be >= 0")
        if isinstance(self.residual_diff_threshold, bool) or not isinstance(
            self.residual_diff_threshold, (int, float)
        ):
            raise TypeError(
                "Cache-DiT residual_diff_threshold must be a finite float"
            )
        threshold = float(self.residual_diff_threshold)
        if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError(
                "Cache-DiT residual_diff_threshold must be finite and in (0, 1)"
            )
        object.__setattr__(self, "residual_diff_threshold", threshold)

    def validate_block_count(self, num_transformer_blocks: int) -> None:
        if type(num_transformer_blocks) is not int or num_transformer_blocks <= 0:
            raise ValueError(
                "Cache-DiT num_transformer_blocks must be a positive non-bool int"
            )
        if self.front_blocks + self.back_blocks >= num_transformer_blocks:
            raise ValueError(
                "Cache-DiT front_blocks + back_blocks must be smaller than "
                "num_transformer_blocks"
            )
