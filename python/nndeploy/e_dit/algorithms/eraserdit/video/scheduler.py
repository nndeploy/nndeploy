"""Task scheduler for the windowed videoerase runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from nndeploy.e_dit.algorithms.eraserdit.video.contracts import (
    EraseRuntimeContext,
    ObjectRuntimeState,
)
from nndeploy.e_dit.video.windowing import WindowSpec


@dataclass
class RuntimeTaskScheduler:
    event_history: list[dict[str, Any]] = field(default_factory=list)

    def record_event(self, event: str, **payload: Any) -> dict[str, Any]:
        item = {"event": str(event), **payload}
        self.event_history.append(item)
        return item

    def describe_states(
        self,
        object_states: list[ObjectRuntimeState],
    ) -> list[dict[str, Any]]:
        return [
            {
                "task_index": int(state.task_index),
                "object_index": int(state.object_index),
                "next_window_index": int(state.next_window_index),
                "window_count": int(state.window_count),
                "input_frontier": int(state.input_frontier),
                "input_cache_start": int(state.input_cache.start_index),
                "input_cache_end": int(state.input_cache.end_index),
                "output_cache_end": int(state.output_cache.end_index),
                "flush_frontier": int(state.flush_frontier),
                "scene_index": int(state.scene_index),
                "finished": bool(state.finished),
            }
            for state in object_states
        ]

    def select_next_task(
        self,
        object_states: list[ObjectRuntimeState],
        ready_check: Callable[[ObjectRuntimeState, WindowSpec], bool],
    ) -> ObjectRuntimeState | None:
        for state in reversed(object_states):
            if state.finished:
                continue
            if state.next_window_index >= state.window_count:
                continue
            spec = state.window_specs[state.next_window_index]
            if ready_check(state, spec):
                self.record_event(
                    "scheduler_select",
                    task_index=int(state.task_index),
                    object_index=int(state.object_index),
                    window_index=int(spec.window_index),
                    scene_index=int(spec.scene_index),
                    load_start=int(spec.load_start),
                    load_end=int(spec.load_end),
                    flush_frontier=int(state.flush_frontier),
                )
                return state
        self.record_event(
            "scheduler_no_ready_task",
            states=self.describe_states(object_states),
        )
        return None

    def compute_tail_flush_end(
        self,
        context: EraseRuntimeContext,
    ) -> int:
        if not context.object_states:
            return int(context.next_write_index)
        final_state = context.object_states[-1]
        final_cache = context.final_window_output_cache
        if final_cache is None:
            return int(context.next_write_index)
        flush_end = max(
            int(context.next_write_index),
            min(int(final_state.flush_frontier), int(final_cache.end_index)),
        )
        self.record_event(
            "scheduler_tail_flush_frontier",
            task_index=int(final_state.task_index),
            object_index=int(final_state.object_index),
            flush_end=int(flush_end),
            next_write_index=int(context.next_write_index),
            flush_frontier=int(final_state.flush_frontier),
            final_cache_end=int(final_cache.end_index),
        )
        return int(flush_end)
