"""Runtime event and task snapshot helpers for pipelines.runtime."""

from __future__ import annotations

from typing import Any

from nndeploy.e_dit.algorithms.eraserdit.video.contracts import (
    EraseRuntimeContext,
    RuntimeTaskState,
)


def record_runtime_event(
    context: EraseRuntimeContext,
    event: str,
    task_state: RuntimeTaskState | None = None,
    **payload: Any,
) -> dict[str, Any]:
    item: dict[str, Any] = {"event": str(event), **payload}
    if task_state is not None:
        item.setdefault("task_index", int(task_state.task_index))
        item.setdefault("object_index", int(task_state.object_index))
        item.setdefault("scene_index", int(task_state.scene_index))
        task_state.record_runtime_event(
            event,
            **{key: value for key, value in item.items() if key != "event"},
        )
    context.runtime_event_history.append(item)
    return item


def record_task_state_snapshot(
    context: EraseRuntimeContext,
    task_state: RuntimeTaskState,
    phase: str,
    **payload: Any,
) -> dict[str, Any]:
    scene_value: tuple[int, int] | None = None
    if 0 <= task_state.scene_index < len(task_state.window_specs):
        current_spec = task_state.window_specs[
            min(task_state.next_window_index, task_state.window_count - 1)
        ]
        scene_value = (int(current_spec.scene_start), int(current_spec.scene_end - current_spec.scene_start))
    snapshot = {
        "phase": str(phase),
        "task_index": int(task_state.task_index),
        "object_index": int(task_state.object_index),
        "scene_index": int(task_state.scene_index),
        "scene": scene_value,
        "input_role": task_state.input_role,
        "next_window_index": int(task_state.next_window_index),
        "window_count": int(task_state.window_count),
        "finished": bool(task_state.finished),
        "input_frontier": int(task_state.input_frontier),
        "flush_frontier": int(task_state.flush_frontier),
        "raw_input_cache_start": int(task_state.raw_input_cache.start_index),
        "raw_input_cache_end": int(task_state.raw_input_cache.end_index),
        "modified_output_cache_start": int(task_state.modified_output_cache.start_index),
        "modified_output_cache_end": int(task_state.modified_output_cache.end_index),
        "overlap_output_cache_start": (
            int(task_state.overlap_output_cache.start_index)
            if task_state.overlap_output_cache is not None
            else None
        ),
        "overlap_output_cache_end": (
            int(task_state.overlap_output_cache.end_index)
            if task_state.overlap_output_cache is not None
            else None
        ),
        "skip_count": int(task_state.skip_count),
        **payload,
    }
    context.task_state_history.append(snapshot)
    return snapshot


def update_window_state(
    context: EraseRuntimeContext,
    object_index: int,
    window_index: int,
    **payload: Any,
) -> None:
    key = (int(object_index), int(window_index))
    state = context.window_state_map.setdefault(
        key,
        {
            "object_index": int(object_index),
            "window_index": int(window_index),
        },
    )
    state.update(payload)
    context.window_state_history.append(
        {
            "object_index": int(object_index),
            "window_index": int(window_index),
            **payload,
        }
    )
    if 0 <= int(object_index) < len(context.object_states):
        task_state = context.object_states[int(object_index)]
        if "scene_index" in payload and payload["scene_index"] is not None:
            task_state.scene_index = int(payload["scene_index"])
        record_runtime_event(
            context,
            "window_state_update",
            task_state=task_state,
            window_index=int(window_index),
            status=payload.get("status"),
            load_start=payload.get("load_start"),
            load_end=payload.get("load_end"),
            commit_start=payload.get("commit_start"),
            commit_end=payload.get("commit_end"),
        )
        record_task_state_snapshot(
            context,
            task_state,
            phase=str(payload.get("status") or "window_state_update"),
            window_index=int(window_index),
        )
