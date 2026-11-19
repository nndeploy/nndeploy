# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from PIL import Image


_NPY_MAGIC = b"\x93NUMPY"
_ZIP_MAGIC = b"PK\x03\x04"
_NPZ_KEY_PRIORITY = ("masks", "mask", "arr_0")


def _read_prefix(path: str, size: int = 4096) -> bytes:
    with open(path, "rb") as f:
        return f.read(size)


def _detect_mask_type(mask_path: str) -> str:
    prefix = _read_prefix(mask_path)
    stripped = prefix.lstrip()
    suffix = os.path.splitext(mask_path)[1].lower()

    if prefix.startswith(_NPY_MAGIC) or suffix == ".npy":
        return "numpy"
    if prefix.startswith(_ZIP_MAGIC) or suffix == ".npz":
        return "numpy"
    if stripped.startswith((b"{", b"[")) or suffix == ".json":
        return "coco"
    return "video"


def _resize_if_needed(mask: Image.Image, target_size: tuple[int, int] | None) -> Image.Image:
    if target_size is None or mask.size == target_size:
        return mask
    return mask.resize(target_size, Image.Resampling.NEAREST)


def _to_binary_pil(
    frame: np.ndarray | Image.Image,
    target_size: tuple[int, int] | None = None,
) -> Image.Image:
    if isinstance(frame, Image.Image):
        mask = frame.convert("L")
    else:
        arr = np.asarray(frame)
        if arr.ndim == 3:
            if arr.shape[-1] == 1:
                arr = arr[..., 0]
            elif arr.shape[-1] in (3, 4):
                arr = arr[..., :3].max(axis=-1)
            elif arr.shape[0] in (1, 3, 4):
                arr = arr[:3].max(axis=0)
            else:
                raise ValueError(f"Unsupported mask frame shape: {arr.shape}")
        if arr.ndim != 2:
            raise ValueError(f"Unsupported mask frame shape: {arr.shape}")
        mask = Image.fromarray(_binarize_array(arr))

    mask = _resize_if_needed(mask, target_size)
    mask_np = np.asarray(mask.convert("L"))
    return Image.fromarray(((mask_np > 127) * 255).astype(np.uint8))


def _binarize_array(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.bool_:
        binary = arr
    elif np.issubdtype(arr.dtype, np.floating):
        binary = arr > 0.5
    elif np.issubdtype(arr.dtype, np.integer):
        binary = arr > 0
    else:
        raise ValueError(f"Unsupported numpy mask dtype: {arr.dtype}")
    return (binary.astype(np.uint8) * 255)


def _load_numpy_mask_payload(mask_path: str) -> Any:
    data = np.load(mask_path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        try:
            for key in _NPZ_KEY_PRIORITY:
                if key in data.files:
                    return data[key]
            if not data.files:
                raise ValueError(f"NPZ mask file has no arrays: {mask_path}")
            return data[data.files[0]]
        finally:
            data.close()

    if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
        return data.item()
    return data


def _extract_numpy_array(payload: Any) -> np.ndarray:
    if isinstance(payload, dict):
        for key in _NPZ_KEY_PRIORITY:
            if key in payload:
                return np.asarray(payload[key])
        raise ValueError(
            "Unsupported numpy mask dict. Expected one of keys: "
            f"{', '.join(_NPZ_KEY_PRIORITY)}"
        )
    return np.asarray(payload)


def _normalize_numpy_mask_array(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[None, ...]

    if arr.ndim == 3:
        # Prefer the documented video-mask layout: (T, H, W).
        return arr

    if arr.ndim == 4:
        if arr.shape[-1] in (1, 3, 4) and arr.shape[1] > 4:
            return arr
        # Treat (T, N, H, W) as multi-object masks and union objects.
        return arr.max(axis=1)

    raise ValueError(
        "Unsupported numpy mask shape. Expected (T,H,W), (T,H,W,C), "
        f"or (T,N,H,W), got {arr.shape}"
    )


def _load_numpy_mask_array(mask_path: str) -> np.ndarray:
    payload = _load_numpy_mask_payload(mask_path)
    arr = _extract_numpy_array(payload)
    return _normalize_numpy_mask_array(arr)


def _decode_uncompressed_rle(size: list[int], counts: list[int]) -> np.ndarray:
    height, width = size
    flat = np.zeros(height * width, dtype=np.uint8)
    offset = 0
    value = 0
    for count in counts:
        count = int(count)
        if count < 0:
            raise ValueError(f"COCO RLE counts must be non-negative, got {count}")
        end = offset + count
        if end > flat.size:
            raise ValueError(
                f"COCO RLE count sum exceeds mask size: {end} > {flat.size}"
            )
        if value:
            flat[offset:end] = 1
        offset = end
        value = 1 - value

    if offset != flat.size:
        raise ValueError(
            f"COCO RLE count sum does not match mask size: {offset} != {flat.size}"
        )
    return flat.reshape((height, width), order="F")


def _decode_compressed_rle_counts(counts: str | bytes) -> list[int]:
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")

    decoded: list[int] = []
    pos = 0
    while pos < len(counts):
        value = 0
        shift = 0
        more = True
        while more:
            char_value = ord(counts[pos]) - 48
            value |= (char_value & 0x1F) << (5 * shift)
            more = bool(char_value & 0x20)
            pos += 1
            shift += 1
            if not more and (char_value & 0x10):
                value |= -1 << (5 * shift)
        if len(decoded) > 2:
            value += decoded[-2]
        decoded.append(int(value))
    return decoded


def _decode_coco_rle(size: list[int], counts: str | bytes | list[int]) -> np.ndarray:
    if isinstance(counts, list):
        return _decode_uncompressed_rle(size, counts)

    try:
        from pycocotools import mask as mask_util
    except ImportError:
        try:
            return _decode_uncompressed_rle(
                size, _decode_compressed_rle_counts(counts)
            )
        except Exception as fallback_error:
            raise ImportError(
                "pycocotools is required to read this COCO RLE mask JSON file, "
                "and the built-in RLE fallback could not decode it"
            ) from fallback_error

    return mask_util.decode({"size": size, "counts": counts})


def _load_coco_records(mask_path: str) -> list[dict[str, Any]]:
    with open(mask_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return sorted(payload, key=lambda item: int(item.get("frame", 0)))

    if isinstance(payload, dict) and "frames" in payload and isinstance(payload["frames"], list):
        return sorted(payload["frames"], key=lambda item: int(item.get("frame", 0)))

    raise ValueError(
        "Unsupported COCO mask JSON structure. Expected a frame list like "
        "[{'frame': 0, 'size': [H, W], 'counts': [...]}]."
    )


def _load_coco_mask_arrays(mask_path: str) -> list[np.ndarray]:
    records = _load_coco_records(mask_path)
    masks: list[np.ndarray] = []
    for record in records:
        size = record.get("size")
        counts = record.get("counts", [])
        if (
            not isinstance(size, list)
            or len(size) != 2
            or not all(isinstance(v, int) for v in size)
        ):
            raise ValueError(f"Invalid COCO mask size in frame {record.get('frame')}: {size}")

        frame_mask = np.zeros((size[0], size[1]), dtype=np.uint8)
        for item in counts:
            if not isinstance(item, dict) or "mask" not in item:
                raise ValueError(
                    f"Invalid COCO mask count item in frame {record.get('frame')}: {item}"
                )
            mask_value = item["mask"]
            item_size = size
            if isinstance(mask_value, dict):
                item_size = mask_value.get("size", size)
                mask_value = mask_value.get("counts")
            decoded = _decode_coco_rle(item_size, mask_value)
            frame_mask |= (decoded > 0).astype(np.uint8)
        masks.append(frame_mask * 255)
    return masks


def probe_mask_frame_count(mask_path: str) -> int:
    mask_type = _detect_mask_type(mask_path)
    if mask_type == "video":
        from nndeploy.e_dit.algorithms.wan.video.preprocess import (
            probe_video_frame_count,
        )

        return probe_video_frame_count(mask_path)
    if mask_type == "numpy":
        try:
            array = np.load(mask_path, mmap_mode="r", allow_pickle=False)
        except ValueError:
            array = None
        if isinstance(array, np.ndarray):
            return 1 if array.ndim == 2 else int(array.shape[0])
        if array is not None:
            array.close()
        return int(_load_numpy_mask_array(mask_path).shape[0])
    if mask_type == "coco":
        return len(_load_coco_records(mask_path))
    raise ValueError(f"Unsupported mask type: {mask_type}")


def load_mask_frames(
    mask_path: str,
    num_frames: int | None = None,
    target_size: tuple[int, int] | None = None,
) -> list[Image.Image]:
    mask_type = _detect_mask_type(mask_path)
    if mask_type == "video":
        from nndeploy.e_dit.algorithms.wan.video.preprocess import load_video_frames

        frames, _ = load_video_frames(mask_path, num_frames=num_frames)
        return [_to_binary_pil(frame, target_size=target_size) for frame in frames]

    if mask_type == "numpy":
        arr = _load_numpy_mask_array(mask_path)
        if num_frames is not None:
            arr = arr[:num_frames]
        return [_to_binary_pil(frame, target_size=target_size) for frame in arr]

    if mask_type == "coco":
        masks = _load_coco_mask_arrays(mask_path)
        if num_frames is not None:
            masks = masks[:num_frames]
        return [_to_binary_pil(mask, target_size=target_size) for mask in masks]

    raise ValueError(f"Unsupported mask type: {mask_type}")
