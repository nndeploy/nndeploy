"""Windowed runtime driver for the video erase pipeline."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

import torch

from nndeploy.e_dit.algorithms.eraserdit.config import EraserDiTEraseSamplingParams
from nndeploy.e_dit.config.server_args import ServerArgs
from nndeploy.e_dit.nodes.schedule_batch import Req
from nndeploy.e_dit.algorithms.eraserdit.video.events import (
    record_runtime_event,
    record_task_state_snapshot,
    update_window_state,
)
from nndeploy.e_dit.algorithms.eraserdit.video.streaming import (
    ensure_window_cache_loaded,
    evict_cache_before,
    flush_windowed_frames,
    materialize_object_window_mask,
    release_mask_frames,
)
from nndeploy.e_dit.algorithms.eraserdit.video.tracks import _select_sequence_item
from nndeploy.e_dit.algorithms.eraserdit.video.cache_ops import set_object_overlap_cache
from nndeploy.e_dit.algorithms.eraserdit.video.commit_ops import (
    commit_window_to_object_output,
    record_skipped_object_window,
)
from nndeploy.e_dit.algorithms.eraserdit.video.commit_sync import (
    release_window_commit_payload,
)
from nndeploy.e_dit.algorithms.eraserdit.video.objects import (
    build_object_runtime_states,
)
from nndeploy.e_dit.algorithms.eraserdit.video.metadata import (
    collect_window_torch_compile_status,
    collect_window_transformer_cache_status,
    collect_window_vae_parallel_history,
    compute_runtime_final_video_shape,
    window_cache_impl_name,
    write_runtime_history_batch_extra,
    write_runtime_mode_batch_extra,
    write_runtime_result_batch_extra,
)
from nndeploy.e_dit.algorithms.eraserdit.video.scheduler import RuntimeTaskScheduler
from nndeploy.e_dit.algorithms.eraserdit.video.contracts import (
    EraseRuntimeContext,
    ObjectRuntimeState,
)
from nndeploy.e_dit.algorithms.eraserdit.video.sp_dispatch import (
    SPWindowCommand,
    SPWindowCommandCode,
    dispatch_sp_window_command,
    resolve_active_window_commit_context,
    synchronize_window_runtime_boundary,
)
from nndeploy.e_dit.algorithms.eraserdit.video.materializer import (
    build_sp_peer_window_batch,
    cache_window_text_embeddings,
    materialize_window_batch,
)
from nndeploy.e_dit.algorithms.eraserdit.video.planner import infer_window_bbox
from nndeploy.e_dit.algorithms.eraserdit.video.reclaim import (
    reclaim_completed_window,
    transfer_completed_window,
)
from nndeploy.e_dit.algorithms.eraserdit.optimizations.memory_phase_controller import MemoryPhase
from nndeploy.e_dit.nodes.control import service_checkpoint
from nndeploy.e_dit.algorithms.eraserdit.timing import record_diagnostic_stage
from nndeploy.e_dit.video.bbox import resolve_single_window_crop_bbox
from nndeploy.e_dit.video.contracts import (
    resolve_runtime_sp_degree,
    resolve_spatial_alignment,
)
from nndeploy.e_dit.video.io import TensorFrameCache, frames_uint8_to_tensor
from nndeploy.e_dit.video.windowing import WindowSpec


def object_input_ready(
    object_state: ObjectRuntimeState,
    spec: WindowSpec,
) -> bool:
    return object_state.input_frontier >= spec.load_end


def prime_windowed_object_chain_inputs(
    *,
    context: EraseRuntimeContext,
    object_states: list[ObjectRuntimeState],
) -> bool:
    if context.window_runtime_mode != "streaming":
        return False
    if context.video_frame_cache is None:
        return False

    for object_state in object_states:
        if object_state.finished:
            continue
        if object_state.next_window_index >= object_state.window_count:
            continue
        if object_state.object_index != 0:
            continue
        spec = object_state.window_specs[object_state.next_window_index]
        cache_end_before = context.video_frame_cache.end_index
        if cache_end_before >= spec.load_end:
            return False
        ensure_window_cache_loaded(context, spec)
        return context.video_frame_cache.end_index > cache_end_before
    return False


def finalize_object_window_step(
    *,
    context: EraseRuntimeContext,
    object_state: ObjectRuntimeState,
    spec: WindowSpec,
) -> None:
    record_runtime_event(
        context,
        "stage_end_begin",
        task_state=object_state,
        window_index=spec.window_index,
        scene_index=spec.scene_index,
        flush_frontier=int(object_state.flush_frontier),
    )
    current_scene_index = object_state.scene_index
    object_state.next_window_index += 1
    if object_state.next_window_index >= object_state.window_count:
        object_state.finished = True
        set_object_overlap_cache(
            object_state,
            start_frame=object_state.flush_frontier,
            frames=None,
        )
        record_runtime_event(
            context,
            "task_finished",
            task_state=object_state,
            window_index=spec.window_index,
        )
    else:
        next_spec = object_state.window_specs[object_state.next_window_index]
        if next_spec.scene_index != current_scene_index:
            set_object_overlap_cache(
                object_state,
                start_frame=next_spec.load_start,
                frames=None,
            )
            next_scene = (
                int(next_spec.scene_start),
                int(next_spec.scene_end - next_spec.scene_start),
            )
            object_state.emit_pop_scene(
                next_spec.scene_index,
                len(object_state.scenes or []),
                next_scene,
            )
            record_runtime_event(
                context,
                "scene_switch",
                task_state=object_state,
                window_index=spec.window_index,
                previous_scene_index=current_scene_index,
                new_scene_index=next_spec.scene_index,
                new_scene=next_scene,
                overlap_reset=True,
            )
        object_state.scene_index = next_spec.scene_index
    record_runtime_event(
        context,
        "stage_end_complete",
        task_state=object_state,
        window_index=spec.window_index,
        next_window_index=int(object_state.next_window_index),
        finished=bool(object_state.finished),
        flush_frontier=int(object_state.flush_frontier),
    )
    record_task_state_snapshot(
        context,
        object_state,
        phase="window_step_finalized",
        window_index=spec.window_index,
    )


def compute_streaming_stable_end(
    context: EraseRuntimeContext,
) -> int:
    if context.scheduler is None:
        return int(context.next_write_index)
    return context.scheduler.compute_tail_flush_end(context)


def maybe_flush_streaming_runtime(
    *,
    context: EraseRuntimeContext,
) -> None:
    if context.window_runtime_mode != "streaming":
        return
    stable_end = compute_streaming_stable_end(context)
    if context.sequential_video_writer is None:
        if stable_end > context.next_write_index:
            context.next_write_index = stable_end
        if context.object_states:
            evict_cache_before(context, frame_index=stable_end)
            release_mask_frames(
                context,
                release_end=stable_end,
                source="streaming_logical_flush_after_evict",
            )
        return
    flush_windowed_frames(context, flush_end=stable_end)


class _SPWriterCommandController:
    def __init__(
        self,
        *,
        server_args: ServerArgs,
        runtime_mode: str,
    ) -> None:
        self.server_args = server_args
        self.runtime_mode = runtime_mode
        self.command_dispatched = False

    def _dispatch(self, command: SPWindowCommand, *, phase: str) -> None:
        self.command_dispatched = True
        dispatched = dispatch_sp_window_command(
            self.server_args,
            runtime_mode=self.runtime_mode,
            prepare_writer_command=lambda: command,
            phase=phase,
        )
        if dispatched != command:
            raise RuntimeError("writer received a mismatched SP window command")

    def dispatch_run(
        self,
        *,
        object_index: int,
        window_index: int,
        scene_index: int,
    ) -> None:
        self._dispatch(
            SPWindowCommand(
                code=SPWindowCommandCode.RUN,
                object_index=object_index,
                window_index=window_index,
                scene_index=scene_index,
                reason_code=0,
            ),
            phase="writer RUN command",
        )

    def dispatch_skip(
        self,
        *,
        object_index: int,
        window_index: int,
        scene_index: int,
        reason_code: int,
    ) -> None:
        self._dispatch(
            SPWindowCommand(
                code=SPWindowCommandCode.SKIP,
                object_index=object_index,
                window_index=window_index,
                scene_index=scene_index,
                reason_code=reason_code,
            ),
            phase="writer SKIP command",
        )

    def dispatch_end(self) -> None:
        self._dispatch(
            SPWindowCommand(
                code=SPWindowCommandCode.END,
                object_index=-1,
                window_index=-1,
                scene_index=-1,
                reason_code=0,
            ),
            phase="writer END command",
        )

    def dispatch_preparation_error(self, error: BaseException) -> None:
        def raise_preparation_error() -> SPWindowCommand:
            raise error

        dispatch_sp_window_command(
            self.server_args,
            runtime_mode=self.runtime_mode,
            prepare_writer_command=raise_preparation_error,
            phase="writer window preparation",
        )

    def complete_boundary(self) -> None:
        self.command_dispatched = False


def _reclaim_pending_window(
    *,
    context: EraseRuntimeContext,
    server_args: ServerArgs,
) -> None:
    pending = context.pending_window_reclaim
    window_key = pending.window_key if pending is not None else None
    reclaim_error = None
    try:
        if pending is not None:
            controller = context.memory_phase_controller
            snapshot = (
                (lambda _device: controller.record_point("window_reclaim"))
                if controller is not None
                else None
            )
            reclaim_completed_window(
                pending,
                device=getattr(server_args, "device", "cpu"),
                snapshot=snapshot,
            )
            context.pending_window_reclaim = None
    except BaseException as error:
        reclaim_error = error
    synchronize_window_runtime_boundary(reclaim_error, server_args)
    context.window_reclaim_events.append(
        {
            "event": "window_reclaim",
            "monotonic_ns": time.monotonic_ns(),
            "window_key": window_key,
            "status": "complete" if reclaim_error is None else "error",
            "error": (
                None
                if reclaim_error is None
                else f"{type(reclaim_error).__name__}: {reclaim_error}"
            ),
        }
    )


def _drop_pending_window_refs(
    context: EraseRuntimeContext,
) -> None:
    pending = context.pending_window_reclaim
    if pending is not None:
        context.window_reclaim_events.append(
            {
                "event": "window_end_drop_refs",
                "monotonic_ns": time.monotonic_ns(),
                "window_key": pending.window_key,
                "status": "complete",
            }
        )
        pending.batch = None
        context.pending_window_reclaim = None


def _close_memory_phase_controller(
    context: EraseRuntimeContext,
) -> None:
    controller = context.memory_phase_controller
    if controller is not None:
        controller.close()


def _run_writer_window_runtime(
    *,
    batch: Req,
    context: EraseRuntimeContext,
    params: EraserDiTEraseSamplingParams,
    server_args: ServerArgs,
    logger,
    command_controller: _SPWriterCommandController | None,
) -> Req:
    runtime_setup_start = time.perf_counter()
    object_states, total_windows = build_object_runtime_states(
        context=context,
        params=params,
    )
    batch.extra["runtime_object_state_count"] = context.object_count
    batch.extra["runtime_total_windows"] = total_windows
    object_count = len(object_states)
    flat_window_specs: list[dict[str, Any]] = []

    write_runtime_mode_batch_extra(
        batch=batch,
        context=context,
        params=params,
        object_count=object_count,
        runtime_window_cache_impl=window_cache_impl_name(
            context.video_frame_cache
        ),
    )

    if not batch.suppress_logs:
        logger.info(
            "Erase pipeline objects=%d infer_len=%d overlap=%d time_sample=%d time_shift=%d fps=%.3f runtime_requested=%s runtime_effective=%s window_backend=%s cache_impl=%s flush_policy=%s distributed_compute_mode=%s native_vae_parallel=%s degree=%d mode=%s",
            object_count,
            params.infer_len,
            params.overlap,
            params.time_sample,
            params.time_shift,
            context.fps,
            context.requested_runtime_mode,
            context.effective_runtime_mode,
            context.runtime_window_backend,
            window_cache_impl_name(context.video_frame_cache),
            context.runtime_flush_policy,
            context.official_parallel_metadata.get(
                "distributed_compute_mode", "entry_only"
            ),
            context.official_parallel_metadata.get(
                "native_vae_parallel_enabled", False
            ),
            int(
                context.official_parallel_metadata.get(
                    "native_vae_parallel_degree", 1
                )
            ),
            context.official_parallel_metadata.get(
                "native_vae_parallel_mode", "disabled"
            ),
        )

    if context.progress_state is not None:
        context.progress_state.add_pipeline_task(total_windows)

    for state in object_states:
        context.window_specs = state.window_specs
        flat_window_specs.extend(
            [
                {**asdict(spec), "object_index": state.object_index}
                for spec in state.window_specs
            ]
        )
        for spec in state.window_specs:
            update_window_state(
                context,
                state.object_index,
                spec.window_index,
                status="ready",
                scene_index=spec.scene_index,
                load_start=spec.load_start,
                load_end=spec.load_end,
                commit_start=spec.commit_start,
                commit_end=spec.commit_end,
                stable_end=spec.commit_end - spec.overlap_right,
            )
        if not batch.suppress_logs:
            logger.info(
                "object %d/%d windows=%d scenes=%s input_role=%s",
                state.object_index + 1,
                object_count,
                state.window_count,
                state.scenes,
                state.input_role,
            )

    record_diagnostic_stage(
        batch.metrics,
        "diagnostic.runtime.setup",
        time.perf_counter() - runtime_setup_start,
    )
    completed_windows = 0
    window_prepare_seconds = 0.0
    executor_seconds = 0.0
    window_commit_seconds = 0.0
    while completed_windows < total_windows:
        window_prepare_start = time.perf_counter()
        scheduler = context.scheduler or RuntimeTaskScheduler()
        context.scheduler = scheduler
        ready_state = scheduler.select_next_task(object_states, object_input_ready)
        if ready_state is None:
            primed = prime_windowed_object_chain_inputs(
                context=context,
                object_states=object_states,
            )
            scheduler.record_event(
                "scheduler_prime_source",
                primed=bool(primed),
            )
            if primed:
                ready_state = scheduler.select_next_task(
                    object_states, object_input_ready
                )
        if ready_state is None:
            state_debug = scheduler.describe_states(object_states)
            raise RuntimeError(
                f"No ready object found in windowed object-chain scheduler. state={state_debug}"
            )

        spec = ready_state.window_specs[ready_state.next_window_index]
        update_window_state(
            context,
            ready_state.object_index,
            spec.window_index,
            status="running",
            scene_index=spec.scene_index,
            load_start=spec.load_start,
            load_end=spec.load_end,
            commit_start=spec.commit_start,
            commit_end=spec.commit_end,
            input_role=ready_state.input_role,
        )
        object_bbox_frames = (
            context.bbox_tracks[ready_state.object_index]
            if context.bbox_tracks
            and ready_state.object_index < len(context.bbox_tracks)
            else None
        )
        if object_bbox_frames is not None:
            window_bbox = infer_window_bbox(
                bbox_frames=object_bbox_frames,
                spec=spec,
                window_mask=None,
            )
            window_mask = None
        else:
            window_mask = materialize_object_window_mask(
                context, ready_state, spec
            )
            window_bbox = infer_window_bbox(
                bbox_frames=None,
                spec=spec,
                window_mask=window_mask,
            )
        if window_bbox is None:
            if command_controller is not None:
                command_controller.dispatch_skip(
                    object_index=ready_state.object_index,
                    window_index=spec.window_index,
                    scene_index=spec.scene_index,
                    reason_code=2,
                )
            if not batch.suppress_logs:
                logger.info(
                    "object %d window %d skip: load=%d:%d deal=%d:%d reason=no_valid_bbox",
                    ready_state.object_index,
                    spec.window_index,
                    spec.load_start,
                    spec.load_end,
                    spec.deal_start,
                    spec.commit_end,
                )
            lifecycle_error = None
            try:
                record_skipped_object_window(
                    context=context,
                    object_state=ready_state,
                    spec=spec,
                    reason="no_valid_bbox",
                    prompt=_select_sequence_item(
                        ready_state.prompt_source, spec.scene_index
                    ),
                    negative_prompt=_select_sequence_item(
                        ready_state.negative_prompt_source,
                        spec.scene_index,
                    ),
                )
                finalize_object_window_step(
                    context=context,
                    object_state=ready_state,
                    spec=spec,
                )
                maybe_flush_streaming_runtime(
                    context=context,
                )
            except BaseException as error:
                lifecycle_error = error
            synchronize_window_runtime_boundary(lifecycle_error, server_args)
            service_checkpoint(
                batch,
                server_args,
                phase="window_skip_complete",
            )
            if command_controller is not None:
                command_controller.complete_boundary()
            completed_windows += 1
            if not batch.suppress_logs:
                _log_runtime_progress(
                logger=logger,
                    context=context,
                    completed_windows=completed_windows,
                    total_windows=total_windows,
                    current_object_index=ready_state.object_index,
                    current_window_index=spec.window_index,
                )
            elif context.progress_state is not None:
                context.progress_state.update_pipeline(
                    completed=completed_windows,
                    total=total_windows,
                    object_index=ready_state.object_index,
                    object_count=context.object_count,
                    window_index=spec.window_index,
                    window_count=len(context.window_specs),
                )
            window_prepare_seconds += time.perf_counter() - window_prepare_start
            continue

        if command_controller is not None:
            command_controller.dispatch_run(
                object_index=ready_state.object_index,
                window_index=spec.window_index,
                scene_index=spec.scene_index,
            )
        align_w, align_h = resolve_spatial_alignment(
            resolve_runtime_sp_degree(server_args)
        )
        aligned_crop_bbox = resolve_single_window_crop_bbox(
            bbox=window_bbox,
            video_width=int(params.width),
            video_height=int(params.height),
            align_h=align_h,
            align_w=align_w,
            scale_area_ratio=float(params.scale_area_ratio),
            min_pixels=int(params.min_pixels),
            max_pixels=int(params.max_pixels),
            force_crop_align=bool(params.force_crop_align),
        )
        if window_mask is None:
            window_mask = materialize_object_window_mask(
                context, ready_state, spec, crop_bbox=aligned_crop_bbox
            )
        elif isinstance(ready_state.input_cache, TensorFrameCache):
            x, y, width, height = aligned_crop_bbox
            window_mask = window_mask[..., y : y + height, x : x + width].contiguous()
        _reclaim_pending_window(
            context=context,
            server_args=server_args,
        )
        window_batch = materialize_window_batch(
            batch=batch,
            context=context,
            params=params,
            spec=spec,
            object_index=ready_state.object_index,
            bbox_frames=object_bbox_frames,
            object_state=ready_state,
            video_cache=ready_state.input_cache,
            window_mask=window_mask,
            window_bbox=window_bbox,
            aligned_crop_bbox=(
                aligned_crop_bbox
                if isinstance(ready_state.input_cache, TensorFrameCache)
                else None
            ),
        )
        window_batch.mask = window_mask

        if not batch.suppress_logs:
            logger.info(
                "object %d window %d run: load=%d:%d deal=%d:%d commit=%d:%d overlap_left=%d overlap_right=%d future=%d bbox=%s fuse=%s input_role=%s",
                ready_state.object_index,
                spec.window_index,
                spec.load_start,
                spec.load_end,
                spec.deal_start,
                spec.commit_end,
                spec.commit_start,
                spec.commit_end,
                spec.overlap_left,
                spec.overlap_right,
                spec.future_length,
                window_batch.bbox,
                window_batch.overlap_fuse_mode,
                ready_state.input_role,
            )
        if context.progress_state is not None:
            context.progress_state.reset_denoise_task(
                object_index=ready_state.object_index,
                object_count=object_count,
                window_index=spec.window_index,
                window_count=ready_state.window_count,
                total_steps=(
                    len(window_batch.timesteps)
                    if window_batch.timesteps is not None
                    else int(window_batch.num_inference_steps)
                ),
            )
        window_batch.extra["runtime_progress_state"] = context.progress_state
        window_batch.extra["runtime_progress_object_index"] = ready_state.object_index
        window_batch.extra["runtime_progress_object_count"] = object_count
        window_batch.extra["runtime_progress_window_index"] = spec.window_index
        window_batch.extra["runtime_progress_window_count"] = ready_state.window_count
        window_batch.extra["runtime_object_input_role"] = ready_state.input_role
        window_prepare_seconds += time.perf_counter() - window_prepare_start
        executor_start = time.perf_counter()
        window_batch = yield window_batch
        collect_window_vae_parallel_history(
            context=context,
            window_batch=window_batch,
        )
        collect_window_torch_compile_status(
            batch=batch,
            window_batch=window_batch,
        )
        collect_window_transformer_cache_status(
            batch=batch,
            window_batch=window_batch,
        )
        executor_seconds += time.perf_counter() - executor_start
        window_commit_start = time.perf_counter()
        lifecycle_error = None
        controller = context.memory_phase_controller
        try:
            if controller is not None:
                controller.enter(
                    MemoryPhase.COMMIT,
                    component_name=None,
                    window_key=(
                        int(ready_state.object_index),
                        int(spec.window_index),
                    ),
                )
            if (
                getattr(window_batch, "prompt_embeds", None) is not None
                and getattr(window_batch, "prompt_attention_mask", None) is not None
                and getattr(window_batch, "negative_prompt_embeds", None) is not None
                and getattr(window_batch, "negative_attention_mask", None) is not None
                and not isinstance(
                    window_batch.extra.get("cached_text_embeddings"), dict
                )
            ):
                cache_window_text_embeddings(
                    context=context,
                    object_index=ready_state.object_index,
                    scene_index=spec.scene_index,
                    prompt=window_batch.prompt,
                    negative_prompt=window_batch.negative_prompt,
                    prompt_embeds=window_batch.prompt_embeds,
                    prompt_attention_mask=window_batch.prompt_attention_mask,
                    negative_prompt_embeds=window_batch.negative_prompt_embeds,
                    negative_attention_mask=window_batch.negative_attention_mask,
                )
            commit_started = time.perf_counter()
            try:
                commit_window_to_object_output(
                    context=context,
                    object_state=ready_state,
                    spec=spec,
                    window_batch=window_batch,
                )
            finally:
                context.record_runtime_timing(
                    "cache_commit", time.perf_counter() - commit_started
                )
                release_window_commit_payload(window_batch, server_args)
            finalize_object_window_step(
                context=context,
                object_state=ready_state,
                spec=spec,
            )
            maybe_flush_streaming_runtime(
                context=context,
            )
        except BaseException as error:
            lifecycle_error = error
        finally:
            if (
                controller is not None
                and controller.active_phase is MemoryPhase.COMMIT
            ):
                try:
                    controller.exit(MemoryPhase.COMMIT)
                except BaseException as error:
                    if lifecycle_error is None:
                        lifecycle_error = error
        if lifecycle_error is None:
            context.pending_window_reclaim = transfer_completed_window(
                window_batch
            )
            window_batch = None
        synchronize_window_runtime_boundary(lifecycle_error, server_args)
        service_checkpoint(
            batch,
            server_args,
            phase="window_commit_complete",
        )
        if command_controller is not None:
            command_controller.complete_boundary()
        completed_windows += 1
        if not batch.suppress_logs:
            _log_runtime_progress(
                logger=logger,
                context=context,
                completed_windows=completed_windows,
                total_windows=total_windows,
                current_object_index=ready_state.object_index,
                current_window_index=spec.window_index,
            )
        elif context.progress_state is not None:
            context.progress_state.update_pipeline(
                completed=completed_windows,
                total=total_windows,
                object_index=ready_state.object_index,
                object_count=context.object_count,
                window_index=spec.window_index,
                window_count=len(context.window_specs),
            )
        if context.progress_state is not None:
            context.progress_state.hide_denoise()
        window_commit_seconds += time.perf_counter() - window_commit_start

    runtime_finalize_start = time.perf_counter()
    context.object_history = [
        {
            "object_index": state.object_index,
            "scenes": state.scenes,
            "prompt_source": state.prompt_source,
            "negative_prompt_source": state.negative_prompt_source,
            "window_count": state.window_count,
            "skip_count": state.skip_count,
            "input_role": state.input_role,
            "flush_frontier": state.flush_frontier,
            "finished": state.finished,
            "input_cache_end": state.input_cache.end_index,
            "output_cache_end": state.output_cache.end_index,
        }
        for state in object_states
    ]

    if context.window_runtime_mode == "streaming":
        maybe_flush_streaming_runtime(
            context=context,
        )

    if object_states:
        context.final_window_output_cache = object_states[-1].output_cache
        if context.window_runtime_mode != "streaming" or not params.save_output:
            context.video_frame_cache = object_states[-1].output_cache
        # File-only clients do not need an additional full-window FP32 result.
        # Keep the API's tensor return by default, and whenever no file is saved.
        return_tensor = batch.extra.get("return_output_tensor", True) or not params.save_output
        if return_tensor and (context.window_runtime_mode != "streaming" or not params.save_output):
            cached_final_frames = context.final_window_output_cache.slice(
                context.final_window_output_cache.start_index,
                context.final_window_output_cache.end_index,
            )
            final_frames = (
                cached_final_frames
                if isinstance(cached_final_frames, torch.Tensor)
                else frames_uint8_to_tensor(cached_final_frames)
            )
            context.final_video = final_frames.permute(1, 0, 2, 3).unsqueeze(0)
        else:
            context.final_video = None

    write_runtime_result_batch_extra(
        batch=batch,
        context=context,
        flat_window_specs=flat_window_specs,
        final_video_shape=compute_runtime_final_video_shape(
            context=context,
            num_frames=int(params.num_frames),
            height=int(params.height),
            width=int(params.width),
        ),
        include_window_state_history=True,
        include_text_embedding_cache_keys=True,
        object_chain_final_cache_impl=window_cache_impl_name(
            context.final_window_output_cache
        ),
    )
    write_runtime_history_batch_extra(
        batch=batch,
        context=context,
        include_text_embedding_history=True,
        include_load_event_history=True,
        include_flush_event_history=True,
        include_evict_event_history=True,
        include_object_transition_history=True,
        include_runtime_role_metadata=False,
    )
    if command_controller is not None:
        command_controller.dispatch_end()
    finalize_error = None
    try:
        _drop_pending_window_refs(context)
        _close_memory_phase_controller(context)
    except BaseException as error:
        finalize_error = error
    if command_controller is not None:
        synchronize_window_runtime_boundary(finalize_error, server_args)
    elif finalize_error is not None:
        raise finalize_error
    batch.extra["runtime_window_reclaim_events"] = list(
        context.window_reclaim_events
    )
    if context.memory_phase_controller is not None:
        batch.extra["runtime_phase_events"] = [
            asdict(event) for event in context.memory_phase_controller.events
        ]
    record_diagnostic_stage(
        batch.metrics,
        "diagnostic.runtime.window_prepare",
        window_prepare_seconds,
    )
    record_diagnostic_stage(
        batch.metrics,
        "diagnostic.runtime.executor",
        executor_seconds,
    )
    record_diagnostic_stage(
        batch.metrics,
        "diagnostic.runtime.window_commit",
        window_commit_seconds,
    )
    record_diagnostic_stage(
        batch.metrics,
        "diagnostic.runtime.finalize",
        time.perf_counter() - runtime_finalize_start,
    )
    return batch


def run_sp_writer_window_runtime(
    *,
    batch: Req,
    context: EraseRuntimeContext,
    params: EraserDiTEraseSamplingParams,
    server_args: ServerArgs,
    logger,
) -> Req:
    controller = _SPWriterCommandController(
        server_args=server_args,
        runtime_mode=context.effective_runtime_mode,
    )
    try:
        return (yield from _run_writer_window_runtime(
            batch=batch,
            context=context,
            params=params,
            server_args=server_args,
            logger=logger,
            command_controller=controller,
        ))
    except BaseException as error:
        if not controller.command_dispatched:
            controller.dispatch_preparation_error(error)
        raise


def run_sp_peer_window_runtime(
    *,
    batch: Req,
    context: EraseRuntimeContext,
    params: EraserDiTEraseSamplingParams,
    server_args: ServerArgs,
    logger,
) -> Req:
    del logger
    while True:
        command = dispatch_sp_window_command(
            server_args,
            runtime_mode=context.effective_runtime_mode,
            prepare_writer_command=None,
            phase="peer window command receive",
        )
        if command is None:
            raise RuntimeError("active SP peer did not receive a window command")
        if command.code is SPWindowCommandCode.END:
            finalize_error = None
            try:
                _drop_pending_window_refs(context)
                _close_memory_phase_controller(context)
            except BaseException as error:
                finalize_error = error
            synchronize_window_runtime_boundary(finalize_error, server_args)
            return batch
        if command.code is SPWindowCommandCode.SKIP:
            synchronize_window_runtime_boundary(None, server_args)
            service_checkpoint(
                batch,
                server_args,
                phase="window_skip_complete",
            )
            continue
        if command.code is not SPWindowCommandCode.RUN:
            raise RuntimeError(f"unsupported SP peer window command: {command.code}")
        _reclaim_pending_window(
            context=context,
            server_args=server_args,
        )
        window_batch = build_sp_peer_window_batch(
            batch=batch,
            params=params,
            object_index=command.object_index,
            window_index=command.window_index,
            scene_index=command.scene_index,
        )
        window_batch = yield window_batch
        collect_window_vae_parallel_history(
            context=context,
            window_batch=window_batch,
        )
        collect_window_torch_compile_status(
            batch=batch,
            window_batch=window_batch,
        )
        collect_window_transformer_cache_status(
            batch=batch,
            window_batch=window_batch,
        )
        context.pending_window_reclaim = transfer_completed_window(
            window_batch
        )
        window_batch = None
        synchronize_window_runtime_boundary(None, server_args)
        service_checkpoint(
            batch,
            server_args,
            phase="window_commit_complete",
        )


def run_windowed_runtime(
    *,
    batch: Req,
    context: EraseRuntimeContext,
    params: EraserDiTEraseSamplingParams,
    server_args: ServerArgs,
    logger,
) -> Req:
    active = (
        resolve_active_window_commit_context(server_args)
        if context.sp_writer_owned_runtime and context.effective_runtime_mode == "windowed_streaming"
        else None
    )
    if active is not None:
        runtime_fn = (
            run_sp_writer_window_runtime
            if active.is_writer
            else run_sp_peer_window_runtime
        )
        return (yield from runtime_fn(
            batch=batch,
            context=context,
            params=params,
            server_args=server_args,
            logger=logger,
        ))
    return (yield from _run_writer_window_runtime(
        command_controller=None,
        batch=batch,
        context=context,
        params=params,
        server_args=server_args,
        logger=logger,
    ))


def _log_runtime_progress(
    logger,
    context: EraseRuntimeContext,
    completed_windows: int,
    total_windows: int,
    current_object_index: int,
    current_window_index: int,
) -> None:
    if total_windows <= 0:
        return
    elapsed = time.time() - context.pipeline_start_time
    rate = elapsed / max(completed_windows, 1)
    remaining = max(total_windows - completed_windows, 0) * rate
    logger.info(
        "Progress %d/%d %.1f%% object=%d window=%d elapsed=%.1fs eta=%.1fs",
        completed_windows,
        total_windows,
        100.0 * completed_windows / max(total_windows, 1),
        current_object_index + 1,
        current_window_index + 1,
        elapsed,
        remaining,
    )
    if context.progress_state is not None:
        context.progress_state.update_pipeline(
            completed=completed_windows,
            total=total_windows,
            object_index=current_object_index,
            object_count=context.object_count,
            window_index=current_window_index,
            window_count=len(context.window_specs),
        )
