"""Cache and forwarding helpers for the windowed videoerase runtime."""

from __future__ import annotations

import time

import numpy as np
import torch

from nndeploy.e_dit.algorithms.eraserdit.video.events import record_runtime_event, record_task_state_snapshot

from nndeploy.e_dit.algorithms.eraserdit.video.contracts import (
    EraseRuntimeContext,
    ObjectRuntimeState,
)
from nndeploy.e_dit.video.io import ArrayFrameCache, ChunkedFrameCache, TensorFrameCache

from nndeploy.e_dit.video.frame_cache import (
    FrameCache, _append_owned, append_cache_range, append_passthrough_gap, clone_cache, create_empty_cache_like, reset_cache_frames, reset_cache_range
)


def set_object_overlap_cache(
    object_state: ObjectRuntimeState,
    start_frame: int,
    frames: np.ndarray | torch.Tensor | None,
) -> None:
    overlap_cache = object_state.overlap_cache
    if overlap_cache is None:
        overlap_cache = create_empty_cache_like(
            object_state.output_cache,
            start_index=start_frame,
        )
    object_state.overlap_cache = reset_cache_frames(
        overlap_cache,
        object_state.output_cache,
        start_frame,
        frames,
    )


def forward_task_channel_range(
    *,
    context: EraseRuntimeContext,
    source_state: ObjectRuntimeState,
    target_state: ObjectRuntimeState,
    start_index: int,
    length: int,
    channel: str,
) -> None:
    if length <= 0:
        return
    channel_name = str(channel).strip().lower()
    if channel_name not in {"raw", "modified"}:
        raise ValueError(f"Unsupported forwarding channel: {channel}")

    end_index = int(start_index) + int(length)
    source_cache = (
        source_state.input_cache if channel_name == "raw" else source_state.output_cache
    )
    if target_state.input_cache.end_index != int(start_index):
        raise ValueError(
            f"Forwarding channel `{channel_name}` for object {source_state.object_index} "
            f"expects target object {target_state.object_index} input cache at {start_index}, "
            f"got {target_state.input_cache.end_index}"
        )

    transition_started = time.perf_counter()
    copied_bytes, transferred_bytes = append_cache_range(
        target_state.input_cache,
        source_cache,
        int(start_index),
        end_index,
    )
    context.record_runtime_transfer("object_chain_copy", copied_bytes)
    context.record_runtime_transfer("object_chain_transfer", transferred_bytes)
    target_state.input_frontier = max(target_state.input_frontier, end_index)
    transition_history_item = {
            "from_object_index": source_state.object_index,
            "to_object_index": target_state.object_index,
            "window_index": int(source_state.next_window_index),
            "scene_index": int(source_state.scene_index),
            "transition_start": int(start_index),
            "transition_end": end_index,
            "length": int(length),
            "channel": channel_name,
            "source_cache": (
                "raw_input_cache" if channel_name == "raw" else "stable_output_cache"
            ),
        }
    context.object_transition_history.append(transition_history_item)
    record_runtime_event(
        context,
        f"task_forward_{channel_name}",
        task_state=source_state,
        to_task_index=target_state.task_index,
        to_object_index=target_state.object_index,
        window_index=int(source_state.next_window_index),
        transition_start=int(start_index),
        transition_end=end_index,
        length=int(length),
        channel=channel_name,
    )
    record_task_state_snapshot(
        context,
        target_state,
        phase=f"forwarded_{channel_name}_input",
        from_task_index=source_state.task_index,
        from_object_index=source_state.object_index,
        transition_start=int(start_index),
        transition_end=end_index,
        channel=channel_name,
    )
    transition_duration_seconds = time.perf_counter() - transition_started
    transition_history_item["duration_seconds"] = transition_duration_seconds
    context.record_runtime_timing(
        "object_transition", transition_duration_seconds
    )


def register_runtime_task_chain_hooks(
    *,
    context: EraseRuntimeContext,
    object_states: list[ObjectRuntimeState],
) -> None:
    for object_index, object_state in enumerate(object_states[:-1]):
        next_state = object_states[object_index + 1]
        object_state.register_pop_raw_input_hook(
            f"forward_raw_to_task_{next_state.task_index}",
            lambda start_index, length, src=object_state, dst=next_state: forward_task_channel_range(
                context=context,
                source_state=src,
                target_state=dst,
                start_index=start_index,
                length=length,
                channel="raw",
            ),
        )
        object_state.register_pop_modified_hook(
            f"forward_modified_to_task_{next_state.task_index}",
            lambda start_index, length, src=object_state, dst=next_state: forward_task_channel_range(
                context=context,
                source_state=src,
                target_state=dst,
                start_index=start_index,
                length=length,
                channel="modified",
            ),
        )
