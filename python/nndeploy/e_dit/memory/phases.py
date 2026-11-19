"""Residency lifecycle with caller-defined phase order and scope identifiers."""
from __future__ import annotations
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Hashable
import torch


def _phase_name(phase):
    return str(phase.value if isinstance(phase, Enum) else phase)


@dataclass(frozen=True)
class ResidencyEvent:
    monotonic_ns: int
    timestamp_ns: int
    rank: int
    scope_key: Hashable | None
    phase: str | None
    component_name: str | None
    point: str
    allocated_bytes: int
    reserved_bytes: int
    adapter_resident_bytes: int
    active_component_name: str | None
    active_component_count: int


def _cuda_memory_snapshot(device: torch.device) -> tuple[int, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return (0, 0)
    return (
        int(torch.cuda.memory_allocated(device)),
        int(torch.cuda.memory_reserved(device)),
    )


class ResidencyController:
    def __init__(
        self,
        adapter: Any,
        *,
        phase_order,
        component_free_phases=(),
        event_factory=ResidencyEvent,
        rank: int,
        device: torch.device | str,
        cuda_memory_snapshot: Callable[[torch.device], Hashable]
        | None = None,
    ) -> None:
        self._phase_order = {phase: i for i, phase in enumerate(phase_order)}
        self._component_free_phases = frozenset(component_free_phases)
        self._event_factory = event_factory
        self.adapter = adapter
        self.rank = int(rank)
        self.device = torch.device(device)
        self._cuda_memory_snapshot = cuda_memory_snapshot or _cuda_memory_snapshot
        self._active_phase: str | Enum | None = None
        self._active_component: str | None = None
        self._scope_key: Hashable | None = None
        self._active_component_acquired = False
        self._last_phase_by_scope: dict[Hashable, str | Enum] = {}
        self._events: list[ResidencyEvent] = []
        self._closed_events: tuple[ResidencyEvent, ...] | None = None

    @property
    def events(self) -> tuple[ResidencyEvent, ...]:
        if self._closed_events is not None:
            return self._closed_events
        return tuple(self._events)

    @property
    def active_phase(self) -> str | Enum | None:
        return self._active_phase

    def enter(
        self,
        phase: str | Enum,
        *,
        component_name: str | None,
        scope_key: Hashable,
    ) -> None:
        if self._closed_events is not None:
            raise RuntimeError("memory phase controller is closed")
        if self._active_phase is not None:
            raise RuntimeError("an active component phase already exists")
        if phase not in self._phase_order:
            raise ValueError('unknown residency phase')
        if phase in self._component_free_phases and component_name is not None:
            raise ValueError('component-free phase cannot own a component')
        if phase not in self._component_free_phases and not component_name:
            raise ValueError('component_name is required for compute phases')
        previous = self._last_phase_by_scope.get(scope_key)
        if previous is not None and self._phase_order[phase] < self._phase_order[previous]:
            raise RuntimeError("memory phase order cannot regress within a scope")
        if component_name is not None:
            decision_fn = getattr(self.adapter, "plan_component_residency", None)
            decision = decision_fn(component_name) if callable(decision_fn) else None
            if decision is None or bool(getattr(decision, "enabled", True)):
                self.adapter.acquire_component_residency(
                    component_name,
                    reason=f"memory_phase:{_phase_name(phase)}:enter",
                    phase=_phase_name(phase),
                )
                self._active_component_acquired = True
            else:
                record_fallback = getattr(
                    self.adapter,
                    "record_component_residency_fallback",
                    None,
                )
                if callable(record_fallback):
                    record_fallback(decision)
        self._active_phase = phase
        self._active_component = component_name
        self._scope_key = scope_key
        self._last_phase_by_scope[scope_key] = phase
        self.record_point("enter")

    def exit(self, phase: str | Enum) -> None:
        if self._active_phase != phase:
            raise RuntimeError("memory phase exit does not match active phase")
        try:
            self.record_point("exit")
        finally:
            try:
                if self._active_component is not None and self._active_component_acquired:
                    self.adapter.release_component_residency(
                        self._active_component, reason=f"memory_phase:{_phase_name(phase)}:exit")
                    self.adapter.settle_component_transfers()
            finally:
                self._active_phase = None
                self._active_component = None
                self._active_component_acquired = False
                self._scope_key = None

    def record_point(self, point: str) -> ResidencyEvent:
        snapshot = self.adapter.snapshot()
        allocated, reserved = self._cuda_memory_snapshot(self.device)
        resident_bytes = int(snapshot.get("resident_bytes", 0))
        event = self._event_factory(
            monotonic_ns=time.monotonic_ns(),
            timestamp_ns=time.time_ns(),
            rank=self.rank,
            scope_key=self._scope_key,
            phase=_phase_name(self._active_phase)
            if self._active_phase is not None
            else None,
            component_name=self._active_component,
            point=str(point),
            allocated_bytes=int(allocated),
            reserved_bytes=int(reserved),
            adapter_resident_bytes=resident_bytes,
            active_component_name=snapshot.get("active_component_name"),
            active_component_count=int(
                snapshot.get("active_component_name") is not None
            ),
        )
        self._events.append(event)
        return event

    def close(self) -> tuple[ResidencyEvent, ...]:
        if self._closed_events is not None:
            return self._closed_events
        if self._active_phase is not None:
            self.exit(self._active_phase)
        self._closed_events = tuple(self._events)
        return self._closed_events


@contextmanager
def residency_scope(controller, phase, **kwargs):
    was_idle = controller.active_phase is None
    try:
        controller.enter(phase, **kwargs)
        yield
    finally:
        if was_idle and controller.active_phase == phase:
            controller.exit(phase)
