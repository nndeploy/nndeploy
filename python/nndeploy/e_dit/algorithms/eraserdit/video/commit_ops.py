"""Window commit and skip kernel helpers for pipelines.runtime."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from nndeploy.e_dit.algorithms.eraserdit.video.events import record_runtime_event, record_task_state_snapshot, update_window_state
from nndeploy.e_dit.algorithms.eraserdit.video.cache_ops import append_passthrough_gap, set_object_overlap_cache

from nndeploy.e_dit.nodes.schedule_batch import Req
from nndeploy.e_dit.algorithms.eraserdit.video.masks import consume_window_mask
from nndeploy.e_dit.algorithms.eraserdit.video.contracts import (
    EraseRuntimeContext,
    ObjectRuntimeState,
    _is_windowed_runtime_mode,
)
from nndeploy.e_dit.video.io import (
    TensorFrameCache,
    frames_tensor_to_uint8,
)
from nndeploy.e_dit.video.windowing import WindowSpec

from nndeploy.e_dit.video.composite import (
    splice_uint8_patch, splice_bf16_patch, blend_bf16_temporal_overlap, blend_uint8_temporal_overlap, append_frames_tensor_to_cache, blend_temporal_overlap
)

_WINDOW_COMMIT_CONVERSION_CHUNK_SIZE = 4


def commit_window_to_object_output(
    *,
    context: EraseRuntimeContext,
    object_state: ObjectRuntimeState,
    spec: WindowSpec,
    window_batch: Req,
) -> None:
    crop_patch_video = getattr(window_batch, "crop_video_modified", None)
    if crop_patch_video is None:
        raise ValueError("window crop patch is missing")

    crop_bbox = window_batch.crop_bbox
    if crop_bbox is None:
        raise ValueError("window crop bbox is missing")
    x, y, w, h = crop_bbox
    overlap_fuse_mode = str(window_batch.overlap_fuse_mode or "before")

    commit_start = spec.commit_start
    commit_end = spec.commit_end

    input_cache = object_state.input_cache
    output_cache = object_state.output_cache
    stable_start = spec.load_start
    stable_end = commit_end - spec.overlap_right
    stable_length = stable_end - stable_start

    crop_patch = crop_patch_video[:, :, : spec.input_len]
    crop_patch_4d = crop_patch[0].permute(1, 0, 2, 3).contiguous()
    bf16_object_chain = isinstance(input_cache, TensorFrameCache)
    if bf16_object_chain:
        patch_frames = crop_patch_4d.detach().to(
            device="cpu", dtype=torch.bfloat16
        ).contiguous()
        context.record_runtime_transfer(
            "generated_patch_d2h",
            int(patch_frames.numel() * patch_frames.element_size()),
        )
        window_source_frames = None
        source_is_exclusive = True
    else:
        patch_frames = frames_tensor_to_uint8(crop_patch_4d)
        context.record_runtime_transfer("generated_patch_d2h", patch_frames.nbytes)
        window_source_frames = window_batch.extra.pop(
            "window_source_frames_uint8",
            None,
        )
        source_is_exclusive = window_source_frames is not None
    if window_source_frames is None:
        if (
            spec.overlap_left > 0
            and object_state.overlap_cache is not None
            and object_state.overlap_cache.start_index == spec.load_start
            and object_state.overlap_cache.end_index == spec.load_start + spec.overlap_left
        ):
            overlap_frames = object_state.overlap_cache.slice(
                spec.load_start,
                spec.load_start + spec.overlap_left,
            )
            if spec.load_start + spec.overlap_left < spec.load_end:
                new_frames = input_cache.slice(
                    spec.load_start + spec.overlap_left,
                    spec.load_end,
                )
                window_source_frames = (
                    torch.cat([overlap_frames, new_frames], dim=0)
                    if bf16_object_chain
                    else np.concatenate([overlap_frames, new_frames], axis=0)
                )
            else:
                window_source_frames = overlap_frames
        else:
            window_source_frames = input_cache.slice(spec.load_start, spec.load_end)
    if bf16_object_chain:
        valid_source = (
            isinstance(window_source_frames, torch.Tensor)
            and window_source_frames.dtype is torch.bfloat16
            and window_source_frames.ndim == 4
            and window_source_frames.shape[1] == 3
            and window_source_frames.shape[0] == spec.input_len
        )
    else:
        valid_source = (
            isinstance(window_source_frames, np.ndarray)
            and window_source_frames.dtype == np.uint8
            and window_source_frames.ndim == 4
            and window_source_frames.shape[0] == spec.input_len
        )
    if not valid_source:
        raise ValueError(
            "window source buffer does not match the active object-chain contract: "
            f"shape={getattr(window_source_frames, 'shape', None)} "
            f"dtype={getattr(window_source_frames, 'dtype', None)} "
            f"input_len={spec.input_len}"
        )

    original_overlap_frames = None
    if spec.overlap_left > 0:
        original_overlap_frames = (
            window_source_frames[: spec.overlap_left].clone()
            if bf16_object_chain
            else window_source_frames[: spec.overlap_left].copy()
        )

    if bf16_object_chain:
        window_modified_frames = splice_bf16_patch(
            window_source_frames, patch_frames, (x, y, w, h)
        )
        context.record_runtime_transfer("cpu_source_splice", 0)
    else:
        window_modified_frames, splice_bytes = splice_uint8_patch(
            window_source_frames,
            patch_frames,
            (x, y, w, h),
            source_is_exclusive=source_is_exclusive,
        )
        context.record_runtime_transfer("cpu_source_splice", splice_bytes)
    if source_is_exclusive:
        context.record_runtime_transfer(
            "exclusive_buffer_transfer",
            int(window_source_frames.numel() * window_source_frames.element_size())
            if bf16_object_chain
            else int(window_source_frames.nbytes),
        )

    if spec.overlap_left > 0:
        assert original_overlap_frames is not None
        window_modified_frames[: spec.overlap_left] = (
            blend_bf16_temporal_overlap(
                before=original_overlap_frames,
                after=window_modified_frames[: spec.overlap_left],
                mode=overlap_fuse_mode,
            )
            if bf16_object_chain
            else blend_uint8_temporal_overlap(
                before=original_overlap_frames,
                after=window_modified_frames[: spec.overlap_left],
                mode=overlap_fuse_mode,
            )
        )
        context.record_runtime_transfer(
            "overlap_blend",
            int(
                window_modified_frames[: spec.overlap_left].numel()
                * window_modified_frames.element_size()
            )
            if bf16_object_chain
            else int(window_modified_frames[: spec.overlap_left].nbytes),
        )

    passthrough_gap_start = stable_start
    passthrough_gap_end = stable_start
    if output_cache.end_index > stable_start:
        raise ValueError(
            f"Object output cache for object {object_state.object_index} "
            f"advanced beyond stable start {stable_start}: {output_cache.end_index}"
        )
    if output_cache.end_index < stable_start:
        passthrough_gap_start, passthrough_gap_end = append_passthrough_gap(
            output_cache,
            input_cache,
            stable_start,
        )
    passthrough_gap_length = max(0, passthrough_gap_end - passthrough_gap_start)
    if passthrough_gap_length > 0:
        object_state.emit_pop_raw_input(passthrough_gap_start, passthrough_gap_length)
        record_runtime_event(
            context,
            "raw_pop",
            task_state=object_state,
            window_index=spec.window_index,
            start_index=passthrough_gap_start,
            length=passthrough_gap_length,
            source="passthrough_gap",
        )
    if output_cache.end_index != stable_start:
        raise ValueError(
            f"Object output cache for object {object_state.object_index} "
            f"expected stable append index {stable_start}, got {output_cache.end_index}"
        )

    if stable_length > 0:
        stable_frames = window_modified_frames[:stable_length]
        if bf16_object_chain:
            stable_frames = stable_frames.clone()
        append_owned = getattr(output_cache, "append_owned", None)
        if append_owned is None:
            output_cache.append(stable_frames)
            cache_append_bytes = int(stable_frames.nbytes)
        else:
            cache_append_bytes = int(append_owned(stable_frames))
        context.record_runtime_transfer(
            "cache_append",
            cache_append_bytes,
        )
        context.record_runtime_transfer(
            "cache_append_transfer",
            int(stable_frames.numel() * stable_frames.element_size())
            if bf16_object_chain
            else int(stable_frames.nbytes),
        )
        object_state.emit_pop_modified(stable_start, stable_length)
        record_runtime_event(
            context,
            "modified_pop",
            task_state=object_state,
            window_index=spec.window_index,
            start_index=stable_start,
            length=stable_length,
            source="window_commit",
        )

    overlap_frames = None
    if spec.overlap_right > 0:
        overlap_start_offset = stable_end - spec.load_start
        overlap_end_offset = commit_end - spec.load_start
        overlap_frames = window_modified_frames[
            overlap_start_offset:overlap_end_offset
        ]
        overlap_frames = (
            overlap_frames.clone() if bf16_object_chain else overlap_frames.copy()
        )
    set_object_overlap_cache(object_state, stable_end, overlap_frames)
    consume_window_mask(
        context=context,
        spec=spec,
        crop_bbox=crop_bbox,
    )

    object_state.scene_index = spec.scene_index
    object_state.flush_frontier = stable_end
    update_window_state(
        context,
        object_state.object_index,
        spec.window_index,
        status="committed",
        scene_index=spec.scene_index,
        commit_start=commit_start,
        commit_end=commit_end,
        stable_end=stable_end,
        skip=False,
    )
    context.window_history.append(
        {
            "object_index": object_state.object_index,
            "window_index": spec.window_index,
            "scene_index": spec.scene_index,
            "start_index": spec.start_index,
            "end_index": spec.end_index,
            "load_start": spec.load_start,
            "load_end": spec.load_end,
            "deal_start": spec.deal_start,
            "deal_length": spec.deal_length,
            "load_length": spec.load_length,
            "future_length": spec.future_length,
            "stable_start": stable_start,
            "commit_start": commit_start,
            "commit_end": commit_end,
            "stable_end": stable_end,
            "overlap_left": spec.overlap_left,
            "overlap_right": spec.overlap_right,
            "overlap_fuse_mode": overlap_fuse_mode,
            "skip": False,
            "bbox": crop_bbox,
            "prompt": window_batch.prompt,
            "negative_prompt": window_batch.negative_prompt,
            "mask_policy": "consumed_shared_input_stream",
            "output_shape": tuple(crop_patch_video.shape),
            "materialized_output_shape": None,
            "passthrough_gap_start": passthrough_gap_start,
            "passthrough_gap_end": passthrough_gap_end,
            "forward_policy": "stable_output_plus_overlap_cache",
            "input_role": object_state.input_role,
            "output_cache_end": output_cache.end_index,
        }
    )
    record_runtime_event(
        context,
        "window_commit",
        task_state=object_state,
        window_index=spec.window_index,
        stable_start=stable_start,
        stable_end=stable_end,
        passthrough_gap_start=passthrough_gap_start,
        passthrough_gap_end=passthrough_gap_end,
        overlap_right=spec.overlap_right,
        overlap_fuse_mode=overlap_fuse_mode,
    )
    record_task_state_snapshot(
        context,
        object_state,
        phase="committed",
        window_index=spec.window_index,
        stable_end=stable_end,
    )


def record_skipped_object_window(
    *,
    context: EraseRuntimeContext,
    object_state: ObjectRuntimeState,
    spec: WindowSpec,
    reason: str,
    prompt: Any,
    negative_prompt: Any,
) -> None:
    object_state.skip_count += 1
    stable_start = spec.load_start
    stable_end = spec.commit_end - spec.overlap_right
    passthrough_gap_start = stable_start
    passthrough_gap_end = stable_start

    if object_state.output_cache.end_index > stable_start:
        raise ValueError(
            f"Object output cache for object {object_state.object_index} "
            f"advanced beyond stable start {stable_start}: {object_state.output_cache.end_index}"
        )
    if object_state.output_cache.end_index < stable_start:
        passthrough_gap_start, passthrough_gap_end = append_passthrough_gap(
            object_state.output_cache,
            object_state.input_cache,
            stable_start,
        )
    passthrough_gap_length = max(0, passthrough_gap_end - passthrough_gap_start)
    if passthrough_gap_length > 0:
        object_state.emit_pop_raw_input(passthrough_gap_start, passthrough_gap_length)
        record_runtime_event(
            context,
            "raw_pop",
            task_state=object_state,
            window_index=spec.window_index,
            start_index=passthrough_gap_start,
            length=passthrough_gap_length,
            source="skip_passthrough_gap",
        )
    if object_state.output_cache.end_index != stable_start:
        raise ValueError(
            f"Object output cache for object {object_state.object_index} "
            f"expected stable append index {stable_start}, got {object_state.output_cache.end_index}"
        )
    if stable_end > stable_start:
        object_state.output_cache.append(
            object_state.input_cache.slice(stable_start, stable_end)
        )
        object_state.emit_pop_raw_input(stable_start, stable_end - stable_start)
        record_runtime_event(
            context,
            "raw_pop",
            task_state=object_state,
            window_index=spec.window_index,
            start_index=stable_start,
            length=stable_end - stable_start,
            source="window_skip",
        )
    overlap_frames = None
    if spec.overlap_right > 0:
        overlap_frames = object_state.input_cache.slice(stable_end, spec.commit_end)
    set_object_overlap_cache(object_state, stable_end, overlap_frames)
    object_state.scene_index = spec.scene_index
    object_state.flush_frontier = stable_end
    update_window_state(
        context,
        object_state.object_index,
        spec.window_index,
        status="skipped",
        scene_index=spec.scene_index,
        commit_start=spec.commit_start,
        commit_end=spec.commit_end,
        stable_end=stable_end,
        skip=True,
        skip_reason=reason,
        passthrough=True,
    )
    context.window_history.append(
        {
            "object_index": object_state.object_index,
            "window_index": spec.window_index,
            "scene_index": spec.scene_index,
            "start_index": spec.start_index,
            "end_index": spec.end_index,
            "load_start": spec.load_start,
            "load_end": spec.load_end,
            "deal_start": spec.deal_start,
            "deal_length": spec.deal_length,
            "load_length": spec.load_length,
            "future_length": spec.future_length,
            "stable_start": stable_start,
            "commit_start": spec.commit_start,
            "commit_end": spec.commit_end,
            "stable_end": stable_end,
            "overlap_left": spec.overlap_left,
            "overlap_right": spec.overlap_right,
            "overlap_fuse_mode": "before",
            "skip": True,
            "skip_reason": reason,
            "bbox": None,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "mask_policy": "immutable_shared_input_stream",
            "output_shape": None,
            "passthrough_gap_start": passthrough_gap_start,
            "passthrough_gap_end": passthrough_gap_end,
            "forward_policy": "stable_passthrough_plus_overlap_cache",
            "input_role": object_state.input_role,
            "output_cache_end": object_state.output_cache.end_index,
        }
    )
    record_runtime_event(
        context,
        "window_skip",
        task_state=object_state,
        window_index=spec.window_index,
        stable_start=stable_start,
        stable_end=stable_end,
        skip_reason=reason,
    )
    record_task_state_snapshot(
        context,
        object_state,
        phase="skipped",
        window_index=spec.window_index,
        stable_end=stable_end,
        skip_reason=reason,
    )


def commit_window(
    *,
    context: EraseRuntimeContext,
    spec: WindowSpec,
    window_batch: Req,
    object_index: int,
) -> None:
    crop_patch_video = getattr(window_batch, "crop_video_modified", None)
    if crop_patch_video is None:
        raise ValueError("window crop patch is missing")

    commit_start = spec.commit_start
    commit_end = spec.commit_end
    local_start = spec.active_start_offset
    local_end = min(spec.active_end_offset, spec.input_len)
    overlap_fuse_mode = str(window_batch.overlap_fuse_mode or "before")
    crop_bbox = window_batch.crop_bbox
    if crop_bbox is None:
        raise ValueError("window crop bbox is missing")
    x, y, w, h = crop_bbox
    crop_patch = crop_patch_video[:, :, : spec.input_len]
    crop_patch_4d = crop_patch[0].permute(1, 0, 2, 3).contiguous()

    if spec.overlap_left > 0:
        if _is_windowed_runtime_mode(context.runtime_mode):
            if context.video_frame_cache is None:
                raise ValueError(f"{context.runtime_mode} runtime missing frame cache")
            before_overlap_4d = frames_uint8_to_tensor(
                context.video_frame_cache.slice(
                    spec.load_start,
                    spec.load_start + spec.overlap_left,
                )
            ).contiguous()
        else:
            assert context.working_video is not None
            before_overlap = context.working_video[
                :, :, spec.load_start : spec.load_start + spec.overlap_left
            ]
            before_overlap_4d = before_overlap[0].permute(1, 0, 2, 3).contiguous()
        after_overlap_4d = before_overlap_4d.clone()
        overlap_patch_4d = crop_patch_4d[: spec.overlap_left]
        splice_video_inplace(after_overlap_4d, overlap_patch_4d, (x, y))
        fused_overlap_4d = blend_temporal_overlap(
            before=before_overlap_4d,
            after=after_overlap_4d,
            mode=overlap_fuse_mode,
        )
        fused_overlap = fused_overlap_4d.permute(1, 0, 2, 3).unsqueeze(0)
    if _is_windowed_runtime_mode(context.runtime_mode):
        if context.video_frame_cache is None or context.mask_frame_cache is None:
            raise ValueError(f"{context.runtime_mode} runtime missing frame caches")
        if spec.overlap_left > 0:
            context.video_frame_cache.overwrite(
                spec.load_start,
                frames_tensor_to_uint8(fused_overlap_4d),
            )
        commit_base = window_batch.video[:, :, local_start:local_end].clone()
        commit_base_4d = commit_base[0].permute(1, 0, 2, 3).contiguous()
        commit_patch_4d = crop_patch_4d[local_start:local_end]
        splice_video_inplace(commit_base_4d, commit_patch_4d, (x, y))
        context.video_frame_cache.overwrite(
            commit_start,
            frames_tensor_to_uint8(commit_base_4d),
        )
    else:
        assert context.working_video is not None
        assert context.final_video is not None
        if spec.overlap_left > 0:
            context.working_video[
                :, :, spec.load_start : spec.load_start + spec.overlap_left
            ] = fused_overlap.to(
                device=context.working_video.device,
                dtype=context.working_video.dtype,
            )
            context.final_video[
                :, :, spec.load_start : spec.load_start + spec.overlap_left
            ] = fused_overlap.to(
                device=context.final_video.device,
                dtype=context.final_video.dtype,
            )
        commit_patch = crop_patch[:, :, local_start:local_end]
        context.working_video[:, :, commit_start:commit_end, y : y + h, x : x + w] = (
            commit_patch.to(
                device=context.working_video.device,
                dtype=context.working_video.dtype,
            )
        )
        context.final_video[:, :, commit_start:commit_end, y : y + h, x : x + w] = (
            commit_patch.to(
                device=context.final_video.device,
                dtype=context.final_video.dtype,
            )
        )
    consume_window_mask(
        context=context,
        spec=spec,
        crop_bbox=crop_bbox,
    )

    context.window_history.append(
        {
            "object_index": object_index,
            "window_index": spec.window_index,
            "scene_index": spec.scene_index,
            "start_index": spec.start_index,
            "end_index": spec.end_index,
            "load_start": spec.load_start,
            "load_end": spec.load_end,
            "deal_start": spec.deal_start,
            "deal_length": spec.deal_length,
            "load_length": spec.load_length,
            "future_length": spec.future_length,
            "commit_start": commit_start,
            "commit_end": commit_end,
            "overlap_left": spec.overlap_left,
            "overlap_right": spec.overlap_right,
            "overlap_fuse_mode": overlap_fuse_mode,
            "skip": False,
            "bbox": crop_bbox,
            "prompt": window_batch.prompt,
            "negative_prompt": window_batch.negative_prompt,
            "mask_policy": "consumed_shared_input_stream",
            "output_shape": tuple(crop_patch_video.shape),
            "materialized_output_shape": None,
        }
    )


def record_skipped_window(
    *,
    context: EraseRuntimeContext,
    spec: WindowSpec,
    object_index: int,
    reason: str,
    prompt: Any,
    negative_prompt: Any,
) -> None:
    context.window_history.append(
        {
            "object_index": object_index,
            "window_index": spec.window_index,
            "scene_index": spec.scene_index,
            "start_index": spec.start_index,
            "end_index": spec.end_index,
            "load_start": spec.load_start,
            "load_end": spec.load_end,
            "deal_start": spec.deal_start,
            "deal_length": spec.deal_length,
            "load_length": spec.load_length,
            "future_length": spec.future_length,
            "commit_start": spec.commit_start,
            "commit_end": spec.commit_end,
            "overlap_left": spec.overlap_left,
            "overlap_right": spec.overlap_right,
            "overlap_fuse_mode": "before",
            "skip": True,
            "skip_reason": reason,
            "bbox": None,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "output_shape": None,
        }
    )

