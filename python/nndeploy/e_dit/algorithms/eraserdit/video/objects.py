"""Object runtime state builder helpers for pipelines.runtime."""

from __future__ import annotations

from nndeploy.e_dit.algorithms.eraserdit.config import EraserDiTEraseSamplingParams
from nndeploy.e_dit.algorithms.eraserdit.video.events import record_runtime_event, record_task_state_snapshot
from nndeploy.e_dit.algorithms.eraserdit.video.metadata import window_cache_impl_name
from nndeploy.e_dit.algorithms.eraserdit.video.cache_ops import (
    create_empty_cache_like,
    register_runtime_task_chain_hooks,
)
from nndeploy.e_dit.algorithms.eraserdit.video.tracks import (
    _resolve_object_scenes,
    _resolve_object_value,
)
from nndeploy.e_dit.algorithms.eraserdit.video.contracts import (
    EraseRuntimeContext,
    ObjectRuntimeState,
)
from nndeploy.e_dit.algorithms.eraserdit.video.planner import build_window_specs


def build_object_runtime_states(
    *,
    context: EraseRuntimeContext,
    params: EraserDiTEraseSamplingParams,
) -> tuple[list[ObjectRuntimeState], int]:
    if context.video_frame_cache is None:
        raise ValueError("windowed runtime missing source video cache")

    object_count = context.object_count
    object_states: list[ObjectRuntimeState] = []
    shared_window_specs = []
    prompt_sources = []
    negative_prompt_sources = []
    scene_sources = []

    total_windows = 0
    for object_index in range(object_count):
        object_scenes = _resolve_object_scenes(
            params.scenes,
            object_index=object_index,
            object_count=object_count,
        )
        window_specs = build_window_specs(params, scenes=object_scenes)
        shared_window_specs.append(window_specs)
        total_windows += len(window_specs)
        scene_sources.append(object_scenes)
        prompt_sources.append(
            _resolve_object_value(
                params.prompt,
                object_index=object_index,
                object_count=object_count,
            )
        )
        negative_prompt_sources.append(
            _resolve_object_value(
                params.negative_prompt,
                object_index=object_index,
                object_count=object_count,
            )
        )

    for object_index in range(object_count):
        if object_index == 0:
            input_cache = context.video_frame_cache
            input_role = "source_video_cache"
            input_frontier = input_cache.end_index
        else:
            input_cache = create_empty_cache_like(context.video_frame_cache, start_index=0)
            input_role = f"object_{object_index - 1}_output"
            input_frontier = input_cache.end_index

        output_cache = create_empty_cache_like(context.video_frame_cache, start_index=0)
        overlap_cache = create_empty_cache_like(context.video_frame_cache, start_index=0)
        object_states.append(
            ObjectRuntimeState(
                task_index=object_index,
                object_index=object_index,
                window_specs=shared_window_specs[object_index],
                raw_input_cache=input_cache,
                modified_output_cache=output_cache,
                input_role=input_role,
                next_window_index=0,
                finished=False,
                flush_frontier=0,
                input_frontier=input_frontier,
                overlap_output_cache=overlap_cache,
                scene_index=(
                    shared_window_specs[object_index][0].scene_index
                    if shared_window_specs[object_index]
                    else 0
                ),
                skip_count=0,
                scenes=scene_sources[object_index],
                prompt_source=prompt_sources[object_index],
                negative_prompt_source=negative_prompt_sources[object_index],
            )
        )

    context.object_states = object_states
    context.final_window_output_cache = (
        object_states[-1].output_cache if object_states else None
    )
    context.task_state_history = []
    register_runtime_task_chain_hooks(context=context, object_states=object_states)
    for state in object_states:
        record_runtime_event(
            context,
            "task_state_initialized",
            task_state=state,
            input_role=state.input_role,
            scene_index=state.scene_index,
            window_count=state.window_count,
            raw_input_cache_impl=window_cache_impl_name(state.raw_input_cache),
            modified_output_cache_impl=window_cache_impl_name(
                state.modified_output_cache
            ),
            overlap_output_cache_impl=window_cache_impl_name(
                state.overlap_output_cache
            ),
        )
        record_task_state_snapshot(context, state, phase="initialized")
    return object_states, total_windows
