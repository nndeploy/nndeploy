"""Frame cache ownership, contiguous appends and range replacement."""
from __future__ import annotations
import numpy as np
import torch
from nndeploy.e_dit.video.io import ArrayFrameCache, ChunkedFrameCache, TensorFrameCache

FrameCache = ArrayFrameCache | ChunkedFrameCache | TensorFrameCache


def _append_owned(
    cache: FrameCache,
    frames: np.ndarray | torch.Tensor,
) -> int:
    append_owned = getattr(cache, "append_owned", None)
    if append_owned is None:
        cache.append(frames)
        return int(frames.nbytes)
    return int(append_owned(frames))


def create_empty_cache_like(
    cache: FrameCache,
    *,
    start_index: int = 0,
) -> FrameCache:
    if isinstance(cache, ArrayFrameCache):
        return ArrayFrameCache(
            start_index=start_index,
            shape_tail=cache.shape_tail,
            dtype=cache.dtype,
        )
    if isinstance(cache, ChunkedFrameCache):
        return ChunkedFrameCache(
            start_index=start_index,
            shape_tail=cache.shape_tail,
            dtype=cache.dtype,
        )
    if isinstance(cache, TensorFrameCache):
        return TensorFrameCache(
            start_index=start_index,
            shape_tail=cache.shape_tail,
            dtype=cache.dtype,
        )
    raise TypeError(f"Unsupported cache type: {type(cache).__name__}")


def clone_cache(
    cache: FrameCache,
) -> FrameCache:
    cloned = create_empty_cache_like(cache, start_index=cache.start_index)
    if cache.num_frames > 0:
        _append_owned(cloned, cache.slice(cache.start_index, cache.end_index))
    return cloned


def append_cache_range(
    target_cache: FrameCache,
    source_cache: FrameCache,
    start_frame: int,
    end_frame: int,
) -> tuple[int, int]:
    if end_frame <= start_frame:
        return 0, 0
    if target_cache.end_index != start_frame:
        raise ValueError(
            f"Cache append expects contiguous write at {start_frame}, "
            f"but target cache currently ends at {target_cache.end_index}"
        )
    frames = source_cache.slice(start_frame, end_frame)
    transferred_bytes = int(frames.nbytes)
    copied_bytes = transferred_bytes + _append_owned(target_cache, frames)
    return copied_bytes, transferred_bytes


def reset_cache_range(
    cache: FrameCache | None,
    source_cache: FrameCache,
    start_frame: int,
    end_frame: int,
) -> FrameCache:
    del cache
    refreshed = create_empty_cache_like(source_cache, start_index=start_frame)
    if end_frame > start_frame:
        _append_owned(refreshed, source_cache.slice(start_frame, end_frame))
    return refreshed


def reset_cache_frames(
    cache: FrameCache | None,
    cache_template: FrameCache,
    start_frame: int,
    frames: np.ndarray | torch.Tensor | None,
) -> FrameCache:
    refreshed = create_empty_cache_like(cache or cache_template, start_index=start_frame)
    if frames is not None and len(frames) > 0:
        refreshed.append(frames)
    return refreshed


def append_passthrough_gap(
    output_cache: FrameCache,
    input_cache: FrameCache,
    end_frame: int,
) -> tuple[int, int]:
    gap_start = output_cache.end_index
    gap_end = min(int(end_frame), input_cache.end_index)
    if gap_end <= gap_start:
        return gap_start, gap_start
    _append_owned(output_cache, input_cache.slice(gap_start, gap_end))
    return gap_start, gap_end
