"""Video patch composition and temporal overlap on explicit tensors/caches."""
from __future__ import annotations
import math
from typing import Any
import numpy as np
import torch
from nndeploy.e_dit.video.io import frames_tensor_to_uint8

_WINDOW_COMMIT_CONVERSION_CHUNK_SIZE = 4


def splice_uint8_patch(
    source_frames: np.ndarray,
    patch_frames: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    source_is_exclusive: bool = False,
) -> tuple[np.ndarray, int]:
    """Splice a THWC uint8 patch without ever materializing a GPU source window."""

    if source_frames.ndim != 4 or source_frames.shape[-1] != 3:
        raise ValueError("source_frames must have shape [F,H,W,3]")
    if patch_frames.ndim != 4 or patch_frames.shape[-1] != 3:
        raise ValueError("patch_frames must have shape [F,H,W,3]")
    x, y, width, height = (int(value) for value in bbox)
    if (
        patch_frames.shape[0] != source_frames.shape[0]
        or patch_frames.shape[1:3] != (height, width)
    ):
        raise ValueError(
            "patch shape does not match bbox/source frames: "
            f"source={source_frames.shape} patch={patch_frames.shape} bbox={bbox}"
        )
    if x < 0 or y < 0 or x + width > source_frames.shape[2] or y + height > source_frames.shape[1]:
        raise ValueError(f"bbox {bbox} exceeds source shape {source_frames.shape}")
    if x == 0 and y == 0 and width == source_frames.shape[2] and height == source_frames.shape[1]:
        return patch_frames, 0
    result = source_frames if source_is_exclusive else source_frames.copy()
    result[:, y : y + height, x : x + width] = patch_frames
    return result, 0 if source_is_exclusive else int(source_frames.nbytes)


def splice_bf16_patch(
    source_frames: torch.Tensor,
    patch_frames: torch.Tensor,
    bbox: tuple[int, int, int, int],
) -> torch.Tensor:
    """Splice an FCHW patch into an exclusive CPU BF16 source buffer."""

    if source_frames.ndim != 4 or source_frames.shape[1] != 3:
        raise ValueError("source_frames must have shape [F,3,H,W]")
    if patch_frames.ndim != 4 or patch_frames.shape[1] != 3:
        raise ValueError("patch_frames must have shape [F,3,H,W]")
    if source_frames.dtype is not torch.bfloat16 or patch_frames.dtype is not torch.bfloat16:
        raise ValueError("BF16 patch splice requires bfloat16 tensors")
    x, y, width, height = (int(value) for value in bbox)
    if patch_frames.shape[0] != source_frames.shape[0] or tuple(
        patch_frames.shape[-2:]
    ) != (height, width):
        raise ValueError("patch shape does not match bbox/source frames")
    if x < 0 or y < 0 or x + width > source_frames.shape[-1] or y + height > source_frames.shape[-2]:
        raise ValueError(f"bbox {bbox} exceeds source shape {tuple(source_frames.shape)}")
    if x == 0 and y == 0 and width == source_frames.shape[-1] and height == source_frames.shape[-2]:
        return patch_frames
    source_frames[:, :, y : y + height, x : x + width].copy_(patch_frames)
    return source_frames


def blend_bf16_temporal_overlap(
    before: torch.Tensor,
    after: torch.Tensor,
    mode: str = "before",
) -> torch.Tensor:
    """Blend equal FCHW BF16 overlap tensors without crossing the uint8 boundary."""

    if before.shape != after.shape:
        raise ValueError("overlap tensors must share shape")
    if before.dtype is not torch.bfloat16 or after.dtype is not torch.bfloat16:
        raise ValueError("BF16 overlap blend requires bfloat16 tensors")
    normalized_mode = (mode or "before").strip().lower()
    if normalized_mode == "before":
        return before.clone()
    if normalized_mode == "after":
        return after.clone()
    overlap_len = int(before.shape[0])
    if overlap_len <= 1:
        return after.clone()
    if normalized_mode == "cosine":
        weights = torch.cos(torch.linspace(0, math.pi / 2, overlap_len))
    elif normalized_mode == "linear":
        weights = torch.linspace(1, 0, overlap_len)
    elif normalized_mode == "sigmoid":
        weights = torch.sigmoid(torch.linspace(6, -6, overlap_len))
    elif normalized_mode.startswith("value:"):
        ratio = float(normalized_mode.split(":", 1)[1].strip())
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("value fuse mode ratio must be within [0,1]")
        weights = torch.full((overlap_len,), ratio)
    else:
        raise ValueError(f"Unsupported overlap fuse mode: {mode}")
    weights = weights.to(dtype=torch.float32).view(-1, 1, 1, 1)
    return (
        weights * before.float() + (1.0 - weights) * after.float()
    ).to(dtype=torch.bfloat16)


def blend_uint8_temporal_overlap(
    before: np.ndarray,
    after: np.ndarray,
    mode: str = "before",
) -> np.ndarray:
    """Blend equal THWC uint8 overlap arrays with the established fuse weights."""

    if before.shape != after.shape:
        raise ValueError(
            f"overlap arrays must share shape, got {before.shape} and {after.shape}"
        )
    if before.dtype != np.uint8 or after.dtype != np.uint8:
        raise ValueError("overlap arrays must use uint8")
    normalized_mode = (mode or "before").strip().lower()
    if normalized_mode == "before":
        return before.copy()
    if normalized_mode == "after":
        return after.copy()
    overlap_len = int(before.shape[0])
    if overlap_len <= 1:
        return after.copy()
    if normalized_mode == "cosine":
        weights = np.cos(np.linspace(0, math.pi / 2, overlap_len, dtype=np.float32))
    elif normalized_mode == "linear":
        weights = np.linspace(1, 0, overlap_len, dtype=np.float32)
    elif normalized_mode == "sigmoid":
        values = np.linspace(6, -6, overlap_len, dtype=np.float32)
        weights = 1.0 / (1.0 + np.exp(-values))
    elif normalized_mode.startswith("value:"):
        ratio = float(normalized_mode.split(":", 1)[1].strip())
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(
                f"value fuse mode ratio must be within [0,1], got {ratio}"
            )
        weights = np.full((overlap_len,), ratio, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported overlap fuse mode: {mode}")
    weights = weights.reshape(-1, 1, 1, 1)
    blended = (
        weights * before.astype(np.float32)
        + (1.0 - weights) * after.astype(np.float32)
    )
    return np.rint(np.clip(blended, 0.0, 255.0)).astype(np.uint8)


def append_frames_tensor_to_cache(
    output_cache: Any,
    frames: torch.Tensor,
    *,
    chunk_size: int = _WINDOW_COMMIT_CONVERSION_CHUNK_SIZE,
) -> None:
    """Convert and append frames without materializing a full-window GPU temporary."""

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    for start in range(0, int(frames.shape[0]), chunk_size):
        output_cache.append(frames_tensor_to_uint8(frames[start : start + chunk_size]))


def blend_temporal_overlap(
    before: torch.Tensor,
    after: torch.Tensor,
    mode: str = "before",
) -> torch.Tensor:
    if before.shape != after.shape:
        raise ValueError(
            f"overlap tensors must share shape, got {tuple(before.shape)} and {tuple(after.shape)}"
        )
    mode = (mode or "before").strip().lower()
    if mode == "before":
        return before
    if mode == "after":
        return after

    overlap_len = before.shape[2]
    if overlap_len <= 1:
        return after if mode != "before" else before

    if mode == "cosine":
        weights = torch.cos(
            torch.linspace(
                0, math.pi / 2, overlap_len, device=before.device, dtype=before.dtype
            )
        )
    elif mode == "linear":
        weights = torch.linspace(
            1, 0, overlap_len, device=before.device, dtype=before.dtype
        )
    elif mode == "sigmoid":
        weights = torch.sigmoid(
            torch.linspace(6, -6, overlap_len, device=before.device, dtype=before.dtype)
        )
    elif mode.startswith("value:"):
        ratio = float(mode.split(":", 1)[1].strip())
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(f"value fuse mode ratio must be within [0,1], got {ratio}")
        weights = torch.full(
            (overlap_len,), ratio, device=before.device, dtype=before.dtype
        )
    else:
        raise ValueError(f"Unsupported overlap fuse mode: {mode}")

    weights = weights.view(1, 1, -1, 1, 1)
    return weights * before + (1.0 - weights) * after
