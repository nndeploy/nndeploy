"""Runtime contracts and mode helpers for the video erase pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from nndeploy.e_dit.algorithms.eraserdit.config import EraserDiTEraseSamplingParams
from nndeploy.e_dit.nodes.schedule_batch import Req
from nndeploy.e_dit.video.hooks import RuntimeHookRegistry
from nndeploy.e_dit.video.encoding import VideoEncodingProfile
from nndeploy.e_dit.utils.runtime_progress import RuntimeProgressState
from nndeploy.e_dit.video.io import (
    ArrayFrameCache,
    ChunkedFrameCache,
    SequentialVideoReader,
    SequentialVideoWriter,
    TensorFrameCache,
    WindowedVideoStore,
)
from nndeploy.e_dit.video.windowing import WindowSpec


@dataclass(frozen=True)
class RuntimeModeSelection:
    requested_mode: str
    effective_mode: str
    window_runtime_mode: str | None
    window_input_policy: str
    window_output_policy: str
    window_cache_policy: str
    runtime_window_backend: str
    runtime_window_io_policy: str
    runtime_flush_policy: str


@dataclass
class WindowedPreloadRuntimeState:
    active: bool = False
    video_reader: str = "read_video_array"
    mask_reader: str = "read_mask_array"
    video_cache_impl: str = "ArrayFrameCache"
    mask_cache_impl: str = "ArrayFrameCache"
    writer_policy: str = "finalize_only"


@dataclass
class WindowedStreamingRuntimeState:
    active: bool = False
    status: str = "ready_for_incremental_load"
    video_reader: str = "SequentialVideoReader"
    mask_reader: str = "SequentialVideoReader"
    video_cache_impl: str = "ChunkedFrameCache"
    mask_cache_impl: str = "ChunkedFrameCache"
    writer_policy: str = "tail_writer"
    loaded_until_frame: int = 0


def _normalize_requested_runtime_mode(mode_value: str | None) -> str:
    mode = str(mode_value or "auto").strip().lower()
    if mode not in {"auto", "full", "windowed", "windowed_preload", "windowed_streaming"}:
        raise ValueError(f"Unsupported runtime_mode: {mode_value}")
    return mode


def _is_windowed_runtime_mode(mode: str | None) -> bool:
    return str(mode or "").lower() in {"windowed_preload", "windowed_streaming"}


def _resolve_runtime_mode(
    params: EraserDiTEraseSamplingParams,
    height: int,
    width: int,
) -> RuntimeModeSelection:
    requested_mode = _normalize_requested_runtime_mode(params.runtime_mode)

    if requested_mode == "auto":
        if max(height, width) >= 2160:
            return RuntimeModeSelection(
                requested_mode=requested_mode,
                effective_mode="windowed_streaming",
                window_runtime_mode="streaming",
                window_input_policy="sequential_reader_incremental_load",
                window_output_policy="streaming_tail_writer",
                window_cache_policy="chunked_frame_cache_incremental_load",
                runtime_window_backend="streaming",
                runtime_window_io_policy="sequential_reader_plus_chunked_cache",
                runtime_flush_policy="stable_prefix_tail_writer",
            )
        return RuntimeModeSelection(
            requested_mode=requested_mode,
            effective_mode="full",
            window_runtime_mode=None,
            window_input_policy="full_tensor_inputs",
            window_output_policy="full_tensor_finalize",
            window_cache_policy="tensor_full_video_state",
            runtime_window_backend="full_tensor",
            runtime_window_io_policy="in_memory_full_tensor",
            runtime_flush_policy="finalize_only",
        )

    if requested_mode == "full":
        return RuntimeModeSelection(
            requested_mode=requested_mode,
            effective_mode="windowed_streaming",
            window_runtime_mode="streaming",
            window_input_policy="sequential_reader_incremental_load",
            window_output_policy="streaming_tail_writer",
            window_cache_policy="chunked_frame_cache_incremental_load",
            runtime_window_backend="streaming",
            runtime_window_io_policy="sequential_reader_plus_chunked_cache",
            runtime_flush_policy="stable_prefix_tail_writer",
        )

    if requested_mode in {"windowed", "windowed_preload"}:
        return RuntimeModeSelection(
            requested_mode=requested_mode,
            effective_mode="windowed_preload",
            window_runtime_mode="preload",
            window_input_policy="file_backed_preload",
            window_output_policy="finalize_sequential_writer",
            window_cache_policy="array_frame_cache_full_span",
            runtime_window_backend="preload",
            runtime_window_io_policy="preload_arrays",
            runtime_flush_policy="finalize_only",
        )

    return RuntimeModeSelection(
        requested_mode=requested_mode,
        effective_mode="windowed_streaming",
        window_runtime_mode="streaming",
        window_input_policy="sequential_reader_incremental_load",
        window_output_policy="streaming_tail_writer",
        window_cache_policy="chunked_frame_cache_incremental_load",
        runtime_window_backend="streaming",
        runtime_window_io_policy="sequential_reader_plus_chunked_cache",
        runtime_flush_policy="stable_prefix_tail_writer",
    )

@dataclass
class RuntimeTaskState:
    task_index: int
    object_index: int
    window_specs: list[WindowSpec]
    raw_input_cache: ArrayFrameCache | ChunkedFrameCache | TensorFrameCache
    modified_output_cache: ArrayFrameCache | ChunkedFrameCache | TensorFrameCache
    input_role: str = "object_chain"
    next_window_index: int = 0
    finished: bool = False
    flush_frontier: int = 0
    input_frontier: int = 0
    overlap_output_cache: ArrayFrameCache | ChunkedFrameCache | TensorFrameCache | None = None
    scene_index: int = 0
    skip_count: int = 0
    hook_registry: RuntimeHookRegistry = field(default_factory=RuntimeHookRegistry)
    runtime_event_history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def input_cache(self) -> ArrayFrameCache | ChunkedFrameCache | TensorFrameCache:
        return self.raw_input_cache

    @input_cache.setter
    def input_cache(self, value: ArrayFrameCache | ChunkedFrameCache | TensorFrameCache) -> None:
        self.raw_input_cache = value

    @property
    def output_cache(self) -> ArrayFrameCache | ChunkedFrameCache | TensorFrameCache:
        return self.modified_output_cache

    @output_cache.setter
    def output_cache(self, value: ArrayFrameCache | ChunkedFrameCache | TensorFrameCache) -> None:
        self.modified_output_cache = value

    @property
    def overlap_cache(self) -> ArrayFrameCache | ChunkedFrameCache | TensorFrameCache | None:
        return self.overlap_output_cache

    @overlap_cache.setter
    def overlap_cache(self, value: ArrayFrameCache | ChunkedFrameCache | TensorFrameCache | None) -> None:
        self.overlap_output_cache = value

    @property
    def window_count(self) -> int:
        return len(self.window_specs)

    def register_pop_raw_input_hook(
        self,
        key: str,
        hook_func: Callable[[int, int], None],
    ) -> None:
        self.hook_registry.register_pop_raw_input_hook(key, hook_func)

    def remove_pop_raw_input_hook(
        self,
        key: str,
    ) -> Callable[[int, int], None] | None:
        return self.hook_registry.remove_pop_raw_input_hook(key)

    def register_pop_modified_hook(
        self,
        key: str,
        hook_func: Callable[[int, int], None],
    ) -> None:
        self.hook_registry.register_pop_modified_hook(key, hook_func)

    def remove_pop_modified_hook(
        self,
        key: str,
    ) -> Callable[[int, int], None] | None:
        return self.hook_registry.remove_pop_modified_hook(key)

    def register_pop_scene_hook(
        self,
        key: str,
        hook_func: Callable[[int, int, tuple[int, int] | None], None],
    ) -> None:
        self.hook_registry.register_pop_scene_hook(key, hook_func)

    def remove_pop_scene_hook(
        self,
        key: str,
    ) -> Callable[[int, int, tuple[int, int] | None], None] | None:
        return self.hook_registry.remove_pop_scene_hook(key)

    def emit_pop_raw_input(self, start_index: int, length: int) -> None:
        self.hook_registry.emit_pop_raw_input(start_index, length)

    def emit_pop_modified(self, start_index: int, length: int) -> None:
        self.hook_registry.emit_pop_modified(start_index, length)

    def emit_pop_scene(
        self,
        new_scene_index: int,
        scene_count: int,
        new_scene: tuple[int, int] | None,
    ) -> None:
        self.hook_registry.emit_pop_scene(new_scene_index, scene_count, new_scene)

    def record_runtime_event(self, event: str, **payload: Any) -> dict[str, Any]:
        item = {
            "event": str(event),
            "task_index": int(self.task_index),
            "object_index": int(self.object_index),
            **payload,
        }
        self.runtime_event_history.append(item)
        return item


@dataclass
class ObjectRuntimeState(RuntimeTaskState):
    scenes: list[tuple[int, int]] | None = None
    prompt_source: Any = None
    negative_prompt_source: Any = None


@dataclass
class EraseRuntimeContext:
    original_video: torch.Tensor | None
    working_video: torch.Tensor | None
    final_video: torch.Tensor | None
    mask_cache: torch.Tensor | None
    fps: float
    codec_name: str | None
    encoding_profile: VideoEncodingProfile | None = None
    request_batch: Req | None = None
    video_path: str | None = None
    mask_path: str | None = None
    output_file_path: str | None = None
    runtime_mode: str = "full"
    requested_runtime_mode: str = "auto"
    effective_runtime_mode: str = "full"
    window_runtime_mode: str | None = None
    window_input_policy: str = "full_tensor_inputs"
    window_output_policy: str = "full_tensor_finalize"
    window_cache_policy: str = "tensor_full_video_state"
    runtime_window_backend: str = "full_tensor"
    runtime_window_io_policy: str = "in_memory_full_tensor"
    runtime_flush_policy: str = "finalize_only"
    video_store: WindowedVideoStore | None = None
    bbox_tracks: list[torch.Tensor] | None = None
    object_count: int = 1
    window_specs: list[WindowSpec] = field(default_factory=list)
    window_history: list[dict[str, Any]] = field(default_factory=list)
    object_history: list[dict[str, Any]] = field(default_factory=list)
    pipeline_start_time: float = 0.0
    progress_state: RuntimeProgressState | None = None
    text_embedding_cache: dict[tuple[int, int, str, str], dict[str, torch.Tensor]] = field(
        default_factory=dict
    )
    window_state_map: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)
    window_state_history: list[dict[str, Any]] = field(default_factory=list)
    load_event_history: list[dict[str, Any]] = field(default_factory=list)
    flush_event_history: list[dict[str, Any]] = field(default_factory=list)
    evict_event_history: list[dict[str, Any]] = field(default_factory=list)
    object_transition_history: list[dict[str, Any]] = field(default_factory=list)
    text_embedding_history: list[dict[str, Any]] = field(default_factory=list)
    mask_release_event_history: list[dict[str, Any]] = field(default_factory=list)
    runtime_event_history: list[dict[str, Any]] = field(default_factory=list)
    task_state_history: list[dict[str, Any]] = field(default_factory=list)
    vae_parallel_history: list[dict[str, Any]] = field(default_factory=list)
    window_reclaim_events: list[dict[str, Any]] = field(default_factory=list)
    scheduler: Any | None = None
    sequential_video_reader: SequentialVideoReader | None = None
    sequential_mask_reader: SequentialVideoReader | None = None
    sequential_video_writer: SequentialVideoWriter | None = None
    video_frame_cache: ArrayFrameCache | ChunkedFrameCache | TensorFrameCache | None = None
    mask_frame_cache: ArrayFrameCache | ChunkedFrameCache | None = None
    mask_source_frames: Any | None = None
    preload_state: WindowedPreloadRuntimeState = field(default_factory=WindowedPreloadRuntimeState)
    streaming_state: WindowedStreamingRuntimeState = field(
        default_factory=WindowedStreamingRuntimeState
    )
    sp_writer_owned_runtime: bool = False
    object_states: list[ObjectRuntimeState] = field(default_factory=list)
    final_window_output_cache: ArrayFrameCache | ChunkedFrameCache | TensorFrameCache | None = None
    next_write_index: int = 0
    mask_release_frontier: int = 0
    streaming_mask_threshold: float | None = None
    distributed_metadata: dict[str, Any] = field(default_factory=dict)
    official_parallel_metadata: dict[str, Any] = field(default_factory=dict)
    runtime_timing_seconds: dict[str, float] = field(default_factory=dict)
    runtime_timing_counts: dict[str, int] = field(default_factory=dict)
    runtime_transfer_bytes: dict[str, int] = field(default_factory=dict)
    runtime_transfer_counts: dict[str, int] = field(default_factory=dict)
    memory_phase_controller: Any | None = None
    pending_window_reclaim: Any | None = None

    def record_runtime_timing(self, name: str, duration_seconds: float) -> None:
        key = str(name)
        self.runtime_timing_seconds[key] = (
            self.runtime_timing_seconds.get(key, 0.0) + float(duration_seconds)
        )
        self.runtime_timing_counts[key] = self.runtime_timing_counts.get(key, 0) + 1

    def record_runtime_transfer(self, name: str, byte_count: int) -> None:
        key = str(name)
        self.runtime_transfer_bytes[key] = (
            self.runtime_transfer_bytes.get(key, 0) + int(byte_count)
        )
        self.runtime_transfer_counts[key] = (
            self.runtime_transfer_counts.get(key, 0) + 1
        )
