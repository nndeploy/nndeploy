"""Branch-local approximate-cache metric and decision consensus."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from nndeploy.e_dit.cache.base import CacheExecutionContext

if TYPE_CHECKING:
    from nndeploy.e_dit.distributed.group_coordinator import GroupCoordinator


@dataclass
class CacheConsensusStats:
    decision_count: int = 0
    collective_count: int = 0
    collective_bytes: int = 0
    decision_seconds: float = 0.0
    collective_seconds: float = 0.0
    mismatch_count: int = 0
    error_count: int = 0


class CacheDecisionConsensus:
    def __init__(
        self,
        coordinator: GroupCoordinator | None,
        *,
        stats: CacheConsensusStats,
    ) -> None:
        self._coordinator = coordinator
        self._stats = stats
        self.probe_regions = None

    def relative_l1(
        self,
        current: torch.Tensor,
        previous: torch.Tensor,
        *,
        context: CacheExecutionContext,
    ) -> float:
        if self.probe_regions is not None:
            sums, error = None, None
            try:
                sums = self.probe_regions.sums(current, previous)
            except BaseException as caught:
                error = caught
            self.synchronize_validation(error, device=current.device, phase='probe regions')
            if self._coordinator is not None:
                self._all_reduce(sums)
            return self.probe_regions.distance(sums)
        error: BaseException | None = None
        numerator = 0.0
        denominator = 0.0
        try:
            if not isinstance(current, torch.Tensor) or not isinstance(
                previous, torch.Tensor
            ):
                raise TypeError("cache probe inputs must be tensors")
            if current.shape != previous.shape:
                raise ValueError("cache probe input shape changed")
            current_value = current.float()
            previous_value = previous.float()
            numerator = float((current_value - previous_value).abs().sum().item())
            denominator = float(previous_value.abs().sum().item())
        except BaseException as caught:
            error = caught

        if self._coordinator is None:
            if error is not None:
                raise error
            return numerator / max(denominator, 1e-12)

        reduced = torch.tensor(
            [1.0 if error is not None else 0.0, numerator, denominator],
            device=current.device,
            dtype=torch.float64,
        )
        self._all_reduce(reduced)
        if int(reduced[0].item()) != 0:
            self._stats.error_count += 1
            if error is not None:
                raise error
            raise RuntimeError("cache metric validation failed on an SP peer")
        return float(reduced[1].item()) / max(float(reduced[2].item()), 1e-12)

    def synchronize_validation(
        self,
        error: BaseException | None,
        *,
        device: torch.device,
        phase: str,
    ) -> None:
        if self._coordinator is None:
            if error is not None:
                raise error
            return
        error_flag = torch.tensor(
            [1 if error is not None else 0],
            device=device,
            dtype=torch.int32,
        )
        self._all_reduce(error_flag)
        if int(error_flag.item()) == 0:
            return
        self._stats.error_count += 1
        if error is not None:
            raise error
        raise RuntimeError(f"cache {phase} validation failed on an SP peer")

    def assert_decision(self, decision_code: int, *, device: torch.device) -> None:
        if self._coordinator is None:
            return
        values = torch.tensor(
            [float(decision_code), float(decision_code * decision_code)],
            device=device,
            dtype=torch.float64,
        )
        self._all_reduce(values)
        world_size = int(self._coordinator.world_size)
        total = float(values[0].item())
        total_squared = float(values[1].item())
        if abs(total_squared * world_size - total * total) > 0.5:
            self._stats.mismatch_count += 1
            raise RuntimeError("cache decision mismatch inside the branch SP group")

    def _all_reduce(self, tensor: torch.Tensor) -> None:
        started = time.perf_counter()
        result = self._coordinator.all_reduce(tensor)
        elapsed = time.perf_counter() - started
        if result is not tensor:
            tensor.copy_(result)
        self._stats.collective_count += 1
        self._stats.collective_bytes += int(tensor.numel() * tensor.element_size())
        self._stats.collective_seconds += elapsed


__all__ = (
    "CacheConsensusStats",
    "CacheDecisionConsensus",
)
