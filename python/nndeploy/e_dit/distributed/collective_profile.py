from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import TypeVar

import torch

_T = TypeVar("_T")


@dataclass(frozen=True)
class CollectiveProfileRecord:
    call_index: int
    collective: str
    group_name: str
    rank: int
    group_rank: int
    world_size: int
    input_bytes: int
    output_bytes: int
    elapsed_ms: float
    timing_source: str
    success: bool
    error_type: str | None
    error_message: str | None


@dataclass(frozen=True)
class CollectiveProfileSummary:
    collective: str
    group_name: str
    rank: int
    group_rank: int
    world_size: int
    call_count: int
    input_bytes: int
    output_bytes: int
    cpu_enqueue_elapsed_ms: float
    failure_count: int
    first_error_type: str | None
    first_error_message: str | None


@dataclass
class _CollectiveProfileAccumulator:
    call_count: int = 0
    input_bytes: int = 0
    output_bytes: int = 0
    cpu_enqueue_elapsed_ms: float = 0.0
    failure_count: int = 0
    first_error_type: str | None = None
    first_error_message: str | None = None


class CollectiveProfiler:
    def __init__(self) -> None:
        self._records: list[CollectiveProfileRecord] = []

    @property
    def records(self) -> tuple[CollectiveProfileRecord, ...]:
        return tuple(self._records)

    def clear(self) -> None:
        self._records.clear()

    def profile_call(
        self,
        *,
        collective: str,
        group_name: str,
        rank: int,
        group_rank: int,
        world_size: int,
        input_bytes: int,
        output_bytes: int,
        use_cuda: bool,
        call: Callable[[], _T],
    ) -> _T:
        call_index = len(self._records) + 1
        start_ns = time.perf_counter_ns()
        timing_source = "cpu"
        start_event = end_event = None
        if use_cuda:
            try:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                timing_source = "cuda_event"
            except Exception:
                start_event = end_event = None
                timing_source = "cpu_fallback"

        error: BaseException | None = None
        error_traceback: TracebackType | None = None
        try:
            result = call()
        except BaseException as caught:
            error = caught
            error_traceback = caught.__traceback__

        cpu_elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        elapsed_ms = cpu_elapsed_ms
        if start_event is not None and end_event is not None:
            try:
                end_event.record()
                end_event.synchronize()
                elapsed_ms = float(start_event.elapsed_time(end_event))
            except Exception:
                elapsed_ms = cpu_elapsed_ms
                timing_source = "cpu_fallback"
        self._records.append(
            CollectiveProfileRecord(
                call_index=call_index,
                collective=collective,
                group_name=group_name,
                rank=rank,
                group_rank=group_rank,
                world_size=world_size,
                input_bytes=input_bytes,
                output_bytes=output_bytes,
                elapsed_ms=elapsed_ms,
                timing_source=timing_source,
                success=error is None,
                error_type=type(error).__name__ if error is not None else None,
                error_message=str(error) if error is not None else None,
            )
        )
        if error is not None:
            raise error.with_traceback(error_traceback)
        return result


class AggregatingCollectiveProfiler:
    """Keep CPU enqueue timing and counters bounded by collective identity."""

    def __init__(self) -> None:
        self._accumulators: dict[
            tuple[str, str, int, int, int], _CollectiveProfileAccumulator
        ] = {}

    @property
    def summaries(self) -> tuple[CollectiveProfileSummary, ...]:
        return tuple(
            CollectiveProfileSummary(
                collective=key[0],
                group_name=key[1],
                rank=key[2],
                group_rank=key[3],
                world_size=key[4],
                call_count=value.call_count,
                input_bytes=value.input_bytes,
                output_bytes=value.output_bytes,
                cpu_enqueue_elapsed_ms=value.cpu_enqueue_elapsed_ms,
                failure_count=value.failure_count,
                first_error_type=value.first_error_type,
                first_error_message=value.first_error_message,
            )
            for key, value in self._accumulators.items()
        )

    def clear(self) -> None:
        self._accumulators.clear()

    def profile_call(
        self,
        *,
        collective: str,
        group_name: str,
        rank: int,
        group_rank: int,
        world_size: int,
        input_bytes: int,
        output_bytes: int,
        use_cuda: bool,
        call: Callable[[], _T],
    ) -> _T:
        del use_cuda
        key = (collective, group_name, rank, group_rank, world_size)
        accumulator = self._accumulators.setdefault(
            key, _CollectiveProfileAccumulator()
        )
        started_ns = time.perf_counter_ns()
        error: BaseException | None = None
        try:
            result = call()
        except BaseException as caught:
            error = caught
            result = None
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        accumulator.call_count += 1
        accumulator.input_bytes += input_bytes
        accumulator.output_bytes += output_bytes
        accumulator.cpu_enqueue_elapsed_ms += elapsed_ms
        if error is not None:
            accumulator.failure_count += 1
            if accumulator.first_error_type is None:
                accumulator.first_error_type = type(error).__name__
                accumulator.first_error_message = str(error)
            raise error
        return result  # type: ignore[return-value]


__all__ = (
    "AggregatingCollectiveProfiler",
    "CollectiveProfileRecord",
    "CollectiveProfileSummary",
    "CollectiveProfiler",
)
