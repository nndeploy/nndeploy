"""Runtime metadata helpers for the video erase pipeline."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

from nndeploy.e_dit.algorithms.eraserdit.config import EraserDiTEraseSamplingParams
from nndeploy.e_dit.nodes.schedule_batch import Req
from nndeploy.e_dit.algorithms.eraserdit.video.contracts import EraseRuntimeContext

RUNTIME_TASK_CONTRACT_VERSION = "phase3_scheduler_scene_tail_flush_parity"
VAE_PARALLEL_HISTORY_KEY = "vae_parallel_history"
TORCH_COMPILE_STATUS_KEY = "torch_compile"
TRANSFORMER_CACHE_STATUS_KEY = "transformer_cache"
TRANSFORMER_CACHE_HISTORY_KEY = "transformer_cache_history"


def collect_window_torch_compile_status(
    *,
    batch: Req,
    window_batch: Req,
) -> None:
    status = window_batch.extra.get(TORCH_COMPILE_STATUS_KEY)
    if status is None:
        return
    if not isinstance(status, dict):
        raise TypeError(
            f"{TORCH_COMPILE_STATUS_KEY} must be a dict, "
            f"got {type(status).__name__}"
        )
    batch.extra[TORCH_COMPILE_STATUS_KEY] = deepcopy(status)


def collect_window_transformer_cache_status(
    *,
    batch: Req,
    window_batch: Req,
) -> None:
    if "attention_backend" in window_batch.extra:
        batch.extra["attention_backend"] = deepcopy(window_batch.extra["attention_backend"])
    cfg_status = window_batch.extra.get("cfg_parallel")
    if cfg_status is not None:
        batch.extra.setdefault("cfg_parallel", []).append(deepcopy(cfg_status))
    for key in ("dit_parallel", "vae_parallel_encode", "vae_parallel_decode"):
        if key in window_batch.extra:
            batch.extra.setdefault("parallel_history", []).append({key: deepcopy(window_batch.extra[key])})
    status = window_batch.extra.get(TRANSFORMER_CACHE_STATUS_KEY)
    if status is None:
        return
    if not isinstance(status, dict):
        raise TypeError(
            f"{TRANSFORMER_CACHE_STATUS_KEY} must be a dict, "
            f"got {type(status).__name__}"
        )
    history = batch.extra.setdefault(TRANSFORMER_CACHE_HISTORY_KEY, [])
    if not isinstance(history, list):
        raise TypeError(
            f"{TRANSFORMER_CACHE_HISTORY_KEY} must be a list, "
            f"got {type(history).__name__}"
        )
    history.append(deepcopy(status))


def collect_window_vae_parallel_history(
    *,
    context: EraseRuntimeContext,
    window_batch: Req,
) -> None:
    history = window_batch.extra.get(VAE_PARALLEL_HISTORY_KEY)
    if history is None:
        return
    if not isinstance(history, list):
        raise TypeError(
            f"{VAE_PARALLEL_HISTORY_KEY} must be a list, "
            f"got {type(history).__name__}"
        )
    context.vae_parallel_history.extend(history)


def build_runtime_video_metadata(
    *,
    context: EraseRuntimeContext,
    params: EraserDiTEraseSamplingParams,
    runtime_window_cache_impl: str,
) -> dict[str, object]:
    return {
        "fps": context.fps,
        "codec_name": context.codec_name,
        "encoding_profile": (
            asdict(context.encoding_profile)
            if context.encoding_profile is not None
            else None
        ),
        "video_path": context.video_path,
        "mask_path": context.mask_path,
        "num_frames": int(params.num_frames),
        "height": int(params.height),
        "width": int(params.width),
        "runtime_mode": context.runtime_mode,
        "runtime_mode_requested": context.requested_runtime_mode,
        "runtime_mode_effective": context.effective_runtime_mode,
        "window_runtime_mode": context.window_runtime_mode,
        "runtime_window_backend": context.runtime_window_backend,
        "runtime_window_io_policy": context.runtime_window_io_policy,
        "runtime_flush_policy": context.runtime_flush_policy,
        "runtime_window_cache_impl": runtime_window_cache_impl,
        "distributed_enabled": bool(
            context.distributed_metadata.get("distributed_enabled", False)
        ),
        "rank": int(context.distributed_metadata.get("rank", 0)),
        "local_rank": int(context.distributed_metadata.get("local_rank", 0)),
        "world_size": int(context.distributed_metadata.get("world_size", 1)),
        "writer_rank": int(context.distributed_metadata.get("writer_rank", 0)),
        "progress_rank": int(context.distributed_metadata.get("progress_rank", 0)),
        "writer_enabled": bool(
            context.distributed_metadata.get("is_writer_rank", True)
        ),
        "progress_enabled": bool(
            context.distributed_metadata.get("is_progress_rank", True)
        ),
        "distributed_compute_mode": str(
            context.official_parallel_metadata.get(
                "distributed_compute_mode", "entry_only"
            )
        ),
        "official_parallel_enabled": bool(
            context.official_parallel_metadata.get("enabled", False)
        ),
        "vae_parallel_enabled": bool(
            context.official_parallel_metadata.get(
                "native_vae_parallel_enabled", False
            )
        ),
        "vae_parallel_degree": int(
            context.official_parallel_metadata.get(
                "native_vae_parallel_degree", 1
            )
        ),
        "vae_parallel_mode": str(
            context.official_parallel_metadata.get(
                "native_vae_parallel_mode", "disabled"
            )
        ),
    }


def write_runtime_mode_batch_extra(
    *,
    batch: Req,
    context: EraseRuntimeContext,
    params: EraserDiTEraseSamplingParams,
    object_count: int,
    runtime_window_cache_impl: str,
) -> None:
    batch.extra["runtime_video_metadata"] = build_runtime_video_metadata(
        context=context,
        params=params,
        runtime_window_cache_impl=runtime_window_cache_impl,
    )
    batch.extra["runtime_bbox_track_count"] = int(object_count)


def write_runtime_history_batch_extra(
    *,
    batch: Req,
    context: EraseRuntimeContext,
    include_text_embedding_history: bool,
    include_load_event_history: bool,
    include_flush_event_history: bool,
    include_evict_event_history: bool,
    include_object_transition_history: bool,
    include_runtime_role_metadata: bool,
) -> None:
    if include_text_embedding_history:
        batch.extra["runtime_text_embedding_history"] = context.text_embedding_history
    if include_load_event_history:
        batch.extra["runtime_load_event_history"] = context.load_event_history
    if include_flush_event_history:
        batch.extra["runtime_flush_event_history"] = context.flush_event_history
    if include_evict_event_history:
        batch.extra["runtime_evict_event_history"] = context.evict_event_history
    if include_object_transition_history:
        batch.extra["runtime_object_transition_history"] = (
            context.object_transition_history
        )

    batch.extra["runtime_mask_release_event_history"] = (
        context.mask_release_event_history
    )
    batch.extra["runtime_mask_release_frontier"] = int(context.mask_release_frontier)
    batch.extra["runtime_scheduler_event_history"] = (
        list(context.scheduler.event_history) if context.scheduler is not None else []
    )
    batch.extra["runtime_event_history"] = context.runtime_event_history
    batch.extra["runtime_task_state_history"] = context.task_state_history
    batch.extra["runtime_task_contract_enabled"] = True
    batch.extra["runtime_task_contract_version"] = RUNTIME_TASK_CONTRACT_VERSION
    batch.extra[VAE_PARALLEL_HISTORY_KEY] = list(
        context.vae_parallel_history
    )
    controller = context.memory_phase_controller
    batch.extra["runtime_phase_events"] = (
        [asdict(event) for event in controller.events]
        if controller is not None
        else []
    )
    batch.extra["runtime_window_reclaim_events"] = list(
        context.window_reclaim_events
    )
    batch.extra["runtime_timing_seconds"] = dict(context.runtime_timing_seconds)
    batch.extra["runtime_timing_counts"] = dict(context.runtime_timing_counts)
    batch.extra["runtime_transfer_bytes"] = dict(context.runtime_transfer_bytes)
    batch.extra["runtime_transfer_counts"] = dict(context.runtime_transfer_counts)
    batch.extra["runtime_encoding_profile"] = (
        asdict(context.encoding_profile)
        if context.encoding_profile is not None
        else None
    )

    if include_runtime_role_metadata:
        batch.extra["runtime_distributed_metadata"] = dict(context.distributed_metadata)
        batch.extra["runtime_official_parallel_metadata"] = dict(
            context.official_parallel_metadata
        )
        batch.extra.setdefault("runtime_official_parallel_history", [])


def compute_runtime_final_video_shape(
    *,
    context: EraseRuntimeContext,
    num_frames: int,
    height: int,
    width: int,
) -> tuple[int, int, int, int, int]:
    if context.final_video is not None:
        return tuple(context.final_video.shape)
    effective_frames = (
        int(context.next_write_index)
        if context.window_runtime_mode == "streaming"
        else int(num_frames)
    )
    return (1, 3, effective_frames, int(height), int(width))


def write_runtime_result_batch_extra(
    *,
    batch: Req,
    context: EraseRuntimeContext,
    flat_window_specs: list[dict[str, Any]],
    final_video_shape: tuple[int, int, int, int, int],
    include_window_state_history: bool,
    include_text_embedding_cache_keys: bool,
    object_chain_final_cache_impl: str | None = None,
) -> None:
    batch.output_video = context.final_video
    batch.output = context.final_video
    batch.extra["runtime_window_specs"] = flat_window_specs
    batch.extra["runtime_window_history"] = context.window_history
    batch.extra["runtime_object_history"] = context.object_history

    if include_window_state_history:
        batch.extra["runtime_window_state_history"] = context.window_state_history

    if include_text_embedding_cache_keys:
        batch.extra["runtime_text_embedding_cache_keys"] = [
            {
                "object_index": key[0],
                "scene_index": key[1],
                "prompt": key[2],
                "negative_prompt": key[3],
            }
            for key in context.text_embedding_cache.keys()
        ]

    if object_chain_final_cache_impl is not None:
        batch.extra["runtime_object_chain_enabled"] = True
        batch.extra["runtime_object_chain_final_cache_impl"] = (
            object_chain_final_cache_impl
        )

    batch.extra["runtime_final_video_shape"] = final_video_shape


def window_cache_impl_name(cache) -> str:
    return "none" if cache is None else type(cache).__name__
