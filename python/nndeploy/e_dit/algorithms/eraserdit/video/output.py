"""Output finalization and resource closing helpers for pipelines.runtime."""

from __future__ import annotations

import time

from nndeploy.e_dit.algorithms.eraserdit.video.streaming import flush_windowed_frames, release_mask_frames

from nndeploy.e_dit.algorithms.eraserdit.config import EraserDiTEraseSamplingParams
from nndeploy.e_dit.nodes.schedule_batch import Req
from nndeploy.e_dit.algorithms.eraserdit.video.metadata import write_runtime_history_batch_extra
from nndeploy.e_dit.algorithms.eraserdit.video.contracts import (
    EraseRuntimeContext,
    _is_windowed_runtime_mode,
)
from nndeploy.e_dit.nodes.control import service_checkpoint
from nndeploy.e_dit.video.io import (
    SequentialVideoWriter,
    frames_tensor_to_uint8,
)


def _create_output_writer(
    *,
    output_file_path: str,
    context: EraseRuntimeContext,
    params: EraserDiTEraseSamplingParams,
) -> SequentialVideoWriter:
    # Resolve through the same knob the streaming writer uses so both runtime
    # modes emit with identical encoder settings.
    from nndeploy.e_dit.algorithms.eraserdit.video.context import _resolve_ffmpeg_thread_count

    request_batch = getattr(context, "request_batch", None)
    thread_count = (
        _resolve_ffmpeg_thread_count(request_batch)
        if request_batch is not None
        else 4
    )
    if context.encoding_profile is not None:
        return SequentialVideoWriter(
            output_file_path,
            encoding_profile=context.encoding_profile,
            thread_count=thread_count,
        )
    return SequentialVideoWriter(
        output_file_path,
        width=int(params.width),
        height=int(params.height),
        fps=context.fps,
        codec_name=context.codec_name,
        thread_count=thread_count,
    )


def finalize_output(
    *,
    batch: Req,
    context: EraseRuntimeContext,
    params: EraserDiTEraseSamplingParams,
) -> None:
    finalization_started = time.perf_counter()
    service_checkpoint(
        batch,
        batch.extra.get("service_server_args"),
        phase="output_finalize_start",
    )
    if batch.metrics is not None:
        batch.metrics.ensure_operation("video_io")
    writer_enabled = bool(context.distributed_metadata.get("is_writer_rank", True))
    configured_output_file_path = batch.output_file_path() if params.save_output else None
    output_file_path = context.output_file_path or configured_output_file_path
    if context.progress_state is not None:
        context.progress_state.hide_denoise()
        context.progress_state.set_pipeline_meta(
            "saving output" if writer_enabled and params.save_output else "finalizing rank"
        )
    if writer_enabled and params.save_output and output_file_path is not None:
        if _is_windowed_runtime_mode(context.runtime_mode):
            if context.window_runtime_mode == "streaming":
                final_cache = context.final_window_output_cache
                if final_cache is None:
                    raise ValueError(
                        "windowed_streaming runtime missing final output cache at finalize"
                    )
                flush_windowed_frames(context, flush_end=final_cache.end_index)
                if context.sequential_video_writer is not None:
                    close_started = time.perf_counter()
                    context.sequential_video_writer.close()
                    context.record_runtime_timing(
                        "video_encode_close", time.perf_counter() - close_started
                    )
                    context.sequential_video_writer = None
            else:
                if context.video_frame_cache is None:
                    raise ValueError(
                        f"{context.runtime_mode} runtime missing frame cache at finalize"
                    )
                writer = _create_output_writer(
                    output_file_path=output_file_path,
                    context=context,
                    params=params,
                )
                chunk = 16
                cache_start = context.video_frame_cache.start_index
                cache_end = context.video_frame_cache.end_index
                for start in range(cache_start, cache_end, chunk):
                    write_started = time.perf_counter()
                    writer.write_frames(
                        context.video_frame_cache.slice(start, min(start + chunk, cache_end))
                    )
                    context.record_runtime_timing(
                        "video_encode_write", time.perf_counter() - write_started
                    )
                close_started = time.perf_counter()
                writer.close()
                context.record_runtime_timing(
                    "video_encode_close", time.perf_counter() - close_started
                )
                release_mask_frames(
                    context,
                    release_end=cache_end,
                    source="preload_finalize",
                )
        else:
            assert context.final_video is not None
            writer = _create_output_writer(
                output_file_path=output_file_path,
                context=context,
                params=params,
            )
            frames = context.final_video[0].permute(1, 0, 2, 3)
            for start in range(0, int(frames.shape[0]), 16):
                write_started = time.perf_counter()
                writer.write_frames(frames_tensor_to_uint8(frames[start : start + 16]))
                context.record_runtime_timing(
                    "video_encode_write", time.perf_counter() - write_started
                )
            close_started = time.perf_counter()
            writer.close()
            context.record_runtime_timing(
                "video_encode_close", time.perf_counter() - close_started
            )
        if batch.metrics is not None:
            batch.metrics.record_operation("video_io")
    service_checkpoint(
        batch,
        batch.extra.get("service_server_args"),
        phase="output_finalize_prepublish",
    )
    batch.extra["output_file_path"] = output_file_path if writer_enabled else None
    batch.extra["output_configured_file_path"] = configured_output_file_path
    batch.extra["output_written_by_rank"] = int(
        context.distributed_metadata.get("writer_rank", 0)
    )
    batch.extra["output_write_skipped_reason"] = (
        None if writer_enabled else "non_writer_rank"
    )
    batch.extra["output_audio_muxed"] = False
    batch.extra["output_audio_mux_error"] = None
    batch.extra["output_audio_policy"] = "origin_video_only"
    context.record_runtime_timing(
        "output_finalization", time.perf_counter() - finalization_started
    )
    write_runtime_history_batch_extra(
        batch=batch,
        context=context,
        include_text_embedding_history=True,
        include_load_event_history=True,
        include_flush_event_history=True,
        include_evict_event_history=True,
        include_object_transition_history=True,
        include_runtime_role_metadata=True,
    )


def close_runtime_resources(
    context: EraseRuntimeContext,
) -> None:
    try:
        if context.sequential_video_reader is not None:
            context.sequential_video_reader.close()
        if context.sequential_mask_reader is not None:
            context.sequential_mask_reader.close()
        if context.sequential_video_writer is not None:
            context.sequential_video_writer.close()
        if context.video_store is not None:
            context.video_store.close()
        if context.progress_state is not None:
            context.progress_state.stop()
    finally:
        # Returned requests used to own this context while it owned the request.
        # Break that cycle after output publication, without waiting for cyclic
        # GC to reclaim full-window frame caches. Preserve the returned output
        # and the diagnostic metadata already copied into batch.extra.
        batch = context.request_batch
        if batch is not None and batch.extra.get("runtime_context") is context:
            batch.extra.pop("runtime_context")
        context.request_batch = None
        # Multi-object forwarding callbacks capture both the context and task
        # states; they have no purpose once the request resources are closed.
        for state in context.object_states:
            state.hook_registry.pop_raw_input_hooks.clear()
            state.hook_registry.pop_modified_hooks.clear()
            state.hook_registry.pop_scene_hooks.clear()
