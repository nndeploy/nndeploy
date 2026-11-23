"""Window request materialization helpers for pipelines.runtime."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

import numpy as np
import torch

from nndeploy.e_dit.algorithms.eraserdit.config import EraserDiTEraseSamplingParams
from nndeploy.e_dit.nodes.schedule_batch import Req
from nndeploy.e_dit.algorithms.eraserdit.video.events import record_runtime_event
from nndeploy.e_dit.algorithms.eraserdit.video.masks import materialize_window_mask
from nndeploy.e_dit.algorithms.eraserdit.video.streaming import ensure_window_cache_loaded
from nndeploy.e_dit.algorithms.eraserdit.video.tracks import _resolve_object_value, _select_sequence_item
from nndeploy.e_dit.algorithms.eraserdit.video.contracts import (
    EraseRuntimeContext,
    ObjectRuntimeState,
    _is_windowed_runtime_mode,
)
from nndeploy.e_dit.algorithms.eraserdit.video.planner import infer_window_bbox
from nndeploy.e_dit.video.generators import build_window_generator
from nndeploy.e_dit.video.io import (
    ArrayFrameCache,
    ChunkedFrameCache,
    TensorFrameCache,
    frames_uint8_to_tensor,
)
from nndeploy.e_dit.video.windowing import WindowSpec

FrameCache = ArrayFrameCache | ChunkedFrameCache | TensorFrameCache

_SP_PEER_EXTRA_KEYS = (
    "memory_phase_controller",
    "sp_writer_owned_runtime",
    "runtime_mode_requested",
    "runtime_mode_effective",
    "runtime_distributed_metadata",
    "runtime_official_parallel_metadata",
    "runtime_resource_policy",
    "cpu_resources",
)


def _cpu_detach_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu").contiguous()


def build_sp_peer_window_batch(
    *,
    batch: Req,
    params: EraserDiTEraseSamplingParams,
    object_index: int,
    window_index: int,
    scene_index: int,
) -> Req:
    """Build a metadata-only request for an active SP peer window."""

    peer_extra = {
        key: batch.extra[key] for key in _SP_PEER_EXTRA_KEYS if key in batch.extra
    }
    peer_extra.update(
        {
            "object_index": int(object_index),
            "window_index": int(window_index),
            "scene_index": int(scene_index),
            "patch_commit_only": True,
            "runtime_progress_enabled": True,
        }
    )
    peer_batch = Req(
        sampling_params=deepcopy(params),
        generator=None,
        modules=batch.modules,
        extra=peer_extra,
        is_warmup=batch.is_warmup,
    )
    peer_batch.metrics = batch.metrics

    # Keep writer-owned inputs visibly absent. These assignments also guard
    # against future Req defaults accidentally retaining a large tensor.
    for field_name in (
        "video",
        "mask",
        "bbox",
        "crop_bbox",
        "crop_video",
        "crop_mask",
        "padded_video",
        "padded_mask",
        "masked_video",
        "cond_latents",
        "cond_masks",
        "mask_values",
        "latents",
        "noisy_latents",
        "crop_video_modified",
        "decoded_video",
        "output_video",
        "output",
    ):
        setattr(peer_batch, field_name, None)
    return peer_batch


def materialize_window_batch(
    *,
    batch: Req,
    context: EraseRuntimeContext,
    params: EraserDiTEraseSamplingParams,
    spec: WindowSpec,
    object_index: int,
    bbox_frames: torch.Tensor | None,
    object_state: ObjectRuntimeState | None = None,
    video_cache: FrameCache | None = None,
    window_mask: torch.Tensor | None = None,
    window_bbox: tuple[int, int, int, int] | None = None,
    aligned_crop_bbox: tuple[int, int, int, int] | None = None,
) -> Req:
    if _is_windowed_runtime_mode(context.runtime_mode):
        ensure_window_cache_loaded(context, spec)
        source_video_cache = video_cache or context.video_frame_cache
        if source_video_cache is None or context.mask_frame_cache is None:
            raise ValueError(f"{context.runtime_mode} runtime missing frame caches")
        if isinstance(source_video_cache, TensorFrameCache):
            if aligned_crop_bbox is None:
                raise ValueError("TensorFrameCache materialization requires aligned_crop_bbox")
            if (
                object_state is not None
                and spec.overlap_left > 0
                and object_state.overlap_cache is not None
                and object_state.overlap_cache.start_index == spec.load_start
                and object_state.overlap_cache.end_index
                == spec.load_start + spec.overlap_left
            ):
                overlap_frames = object_state.overlap_cache.slice_crop(
                    spec.load_start,
                    spec.load_start + spec.overlap_left,
                    aligned_crop_bbox,
                )
                if spec.load_start + spec.overlap_left < spec.load_end:
                    new_frames = source_video_cache.slice_crop(
                        spec.load_start + spec.overlap_left,
                        spec.load_end,
                        aligned_crop_bbox,
                    )
                    crop_frames = torch.cat([overlap_frames, new_frames], dim=0)
                else:
                    crop_frames = overlap_frames
            else:
                crop_frames = source_video_cache.slice_crop(
                    spec.load_start, spec.load_end, aligned_crop_bbox
                )
            window_video = crop_frames.permute(1, 0, 2, 3).unsqueeze(0)
        elif object_state is not None:
            if (
                spec.overlap_left > 0
                and object_state.overlap_cache is not None
                and object_state.overlap_cache.start_index == spec.load_start
                and object_state.overlap_cache.end_index
                == spec.load_start + spec.overlap_left
            ):
                overlap_frames = object_state.overlap_cache.slice(
                    spec.load_start,
                    spec.load_start + spec.overlap_left,
                )
                if spec.load_start + spec.overlap_left < spec.load_end:
                    new_frames = source_video_cache.slice(
                        spec.load_start + spec.overlap_left,
                        spec.load_end,
                    )
                    window_frames = np.concatenate([overlap_frames, new_frames], axis=0)
                else:
                    window_frames = overlap_frames
            else:
                window_frames = source_video_cache.slice(spec.load_start, spec.load_end)
        else:
            window_frames = source_video_cache.slice(spec.load_start, spec.load_end)
        if not isinstance(source_video_cache, TensorFrameCache):
            window_video = (
                frames_uint8_to_tensor(window_frames).permute(1, 0, 2, 3).unsqueeze(0)
            )
        if window_mask is None:
            window_mask = materialize_window_mask(
                context=context,
                spec=spec,
            )
    else:
        assert context.working_video is not None
        window_video = context.working_video[:, :, spec.load_start : spec.load_end].clone()
        if window_mask is None:
            window_mask = materialize_window_mask(
                context=context,
                spec=spec,
            )

    if window_bbox is None:
        if spec.overlap_left > 0:
            window_mask[:, :, : spec.overlap_left] = 0
        if spec.active_end_offset < window_mask.shape[2]:
            window_mask[:, :, spec.active_end_offset :] = 0
        window_bbox = infer_window_bbox(
            bbox_frames=bbox_frames,
            spec=spec,
            window_mask=window_mask,
        )

    window_batch = Req(
        sampling_params=deepcopy(params),
        generator=build_window_generator(
            seed=params.seed,
            request_generator=batch.generator,
        ),
        modules=batch.modules,
        extra=dict(batch.extra),
        is_warmup=batch.is_warmup,
    )
    window_batch.metrics = batch.metrics
    window_batch.video = window_video
    window_batch.mask = window_mask
    window_batch.bbox = window_bbox
    window_batch.height = int(window_video.shape[-2])
    window_batch.width = int(window_video.shape[-1])
    window_batch.num_frames = int(window_video.shape[2])
    window_batch.fps = params.fps

    object_prompt = _resolve_object_value(
        params.prompt,
        object_index=object_index,
        object_count=context.object_count,
    )
    object_negative_prompt = _resolve_object_value(
        params.negative_prompt,
        object_index=object_index,
        object_count=context.object_count,
    )
    window_batch.prompt = _select_sequence_item(object_prompt, spec.scene_index)
    window_batch.negative_prompt = _select_sequence_item(
        object_negative_prompt,
        spec.scene_index,
    )

    cache_key = (
        object_index,
        spec.scene_index,
        window_batch.prompt or "",
        window_batch.negative_prompt or "",
    )
    cached_text = context.text_embedding_cache.get(cache_key)
    if cached_text is not None:
        window_batch.extra["cached_text_embeddings"] = cached_text
        context.text_embedding_history.append(
            {
                "event": "text_embedding_cache_hit",
                "object_index": object_index,
                "scene_index": spec.scene_index,
                "prompt": window_batch.prompt,
                "negative_prompt": window_batch.negative_prompt,
            }
        )
        if object_state is not None:
            record_runtime_event(
                context,
                "text_embedding_cache_hit",
                task_state=object_state,
                window_index=spec.window_index,
                scene_index=spec.scene_index,
                prompt=window_batch.prompt,
                negative_prompt=window_batch.negative_prompt,
            )

    window_batch.extra["window_spec"] = asdict(spec)
    window_batch.extra["window_index"] = spec.window_index
    window_batch.extra["scene_index"] = spec.scene_index
    window_batch.extra["object_index"] = object_index
    window_batch.extra["patch_commit_only"] = True
    window_batch.extra["runtime_progress_enabled"] = True
    if aligned_crop_bbox is not None:
        window_batch.extra["prealigned_crop_bbox"] = aligned_crop_bbox
    if _is_windowed_runtime_mode(context.runtime_mode) and not isinstance(
        source_video_cache, TensorFrameCache
    ):
        window_batch.extra["window_source_frames_uint8"] = window_frames
    if context.memory_phase_controller is not None:
        window_batch.extra["memory_phase_controller"] = (
            context.memory_phase_controller
        )
    window_batch.extra["object_prompt_source"] = object_prompt
    window_batch.extra["object_negative_prompt_source"] = object_negative_prompt
    return window_batch


def cache_window_text_embeddings(
    *,
    context: EraseRuntimeContext,
    object_index: int,
    scene_index: int,
    prompt: str | None,
    negative_prompt: str | None,
    prompt_embeds: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    negative_attention_mask: torch.Tensor,
) -> None:
    cache_key = (
        object_index,
        scene_index,
        prompt or "",
        negative_prompt or "",
    )
    context.text_embedding_cache[cache_key] = {
        "prompt_embeds": _cpu_detach_tensor(prompt_embeds),
        "prompt_attention_mask": _cpu_detach_tensor(prompt_attention_mask),
        "negative_prompt_embeds": _cpu_detach_tensor(negative_prompt_embeds),
        "negative_attention_mask": _cpu_detach_tensor(negative_attention_mask),
    }
    context.text_embedding_history.append(
        {
            "event": "text_embedding_cache_store",
            "object_index": object_index,
            "scene_index": scene_index,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "prompt_embeds_shape": tuple(prompt_embeds.shape),
            "negative_prompt_embeds_shape": tuple(negative_prompt_embeds.shape),
            "cache_device": "cpu",
        }
    )
    task_state = (
        context.object_states[object_index]
        if 0 <= int(object_index) < len(context.object_states)
        else None
    )
    record_runtime_event(
        context,
        "text_embedding_cache_store",
        task_state=task_state,
        scene_index=scene_index,
        prompt=prompt,
        negative_prompt=negative_prompt,
        cache_device="cpu",
        prompt_embeds_shape=tuple(prompt_embeds.shape),
        negative_prompt_embeds_shape=tuple(negative_prompt_embeds.shape),
    )
