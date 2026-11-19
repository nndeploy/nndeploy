# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import tempfile
from typing import Any


def write_videoedit_progress(progress_path: str | None, payload: dict[str, Any]) -> None:
    from nndeploy.dag.execution import current_execution
    context = current_execution()
    if context is not None and context.publish is not None:
        context.publish(payload)
    if not progress_path:
        return

    directory = os.path.dirname(os.path.abspath(progress_path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(progress_path)}.",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_path, progress_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def read_videoedit_progress(progress_path: str | None) -> dict[str, Any]:
    if not progress_path or not os.path.exists(progress_path):
        return {}
    try:
        with open(progress_path, "r") as f:
            data = json.load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def build_window_progress_payload(
    *,
    stage: str,
    total_frames: int | None,
    infer_len: int,
    overlap: int,
    total_windows: int,
    current_window_index: int | None = None,
    current_step_index: int | None = None,
    steps_per_window: int | None = None,
) -> dict[str, Any]:
    total_windows = max(1, int(total_windows))
    current_window = (current_window_index + 1) if current_window_index is not None else 0
    completed_windows = 0
    if current_window_index is not None:
        completed_windows = max(0, min(total_windows, current_window_index))

    completed_steps = None
    total_steps = None
    current_step = None
    if steps_per_window and current_window_index is not None:
        steps_per_window = max(1, int(steps_per_window))
        total_steps = total_windows * steps_per_window
        if current_step_index is None:
            completed_steps = current_window_index * steps_per_window
        else:
            current_step = current_step_index + 1
            completed_steps = current_window_index * steps_per_window + current_step
        completed_steps = max(0, min(total_steps, completed_steps))
        progress = min(99, max(1, int(completed_steps * 98 / total_steps) + 1))
        if completed_steps == total_steps:
            completed_windows = total_windows
    else:
        progress = 1

    if stage == "window_done" and current_window_index is not None:
        completed_windows = max(0, min(total_windows, current_window_index + 1))

    payload: dict[str, Any] = {
        "progress": progress,
        "stage": stage,
        "total_frames": total_frames,
        "infer_len": infer_len,
        "overlap": overlap,
        "total_windows": total_windows,
        "current_window": current_window,
        "completed_windows": completed_windows,
    }
    if steps_per_window is not None:
        payload["steps_per_window"] = steps_per_window
    if total_steps is not None:
        payload["total_steps"] = total_steps
    if current_step is not None:
        payload["current_step"] = current_step
    if completed_steps is not None:
        payload["completed_steps"] = completed_steps
    return payload
