# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from nndeploy.e_dit.algorithms.wan.video.contracts import VideoEditWindowSpec
from nndeploy.e_dit.algorithms.wan.video.mask_io import (
    load_mask_frames,
    probe_mask_frame_count,
)
from nndeploy.e_dit.algorithms.wan.video.windowing import VideoEditPassPlan

@dataclass(frozen=True)
class VideoEditSequence:
    """A pass-local pixel sequence with native source-index provenance."""

    frames: tuple[Image.Image, ...]
    masks: tuple[Image.Image, ...]
    global_indices: tuple[int | None, ...]

    def __post_init__(self) -> None:
        if not (
            len(self.frames) == len(self.masks) == len(self.global_indices)
        ):
            raise ValueError(
                "VideoEdit sequence length mismatch: "
                f"frames={len(self.frames)}, masks={len(self.masks)}, "
                f"global_indices={len(self.global_indices)}"
            )


def load_video_frames(
    video_path: str, num_frames: int | None = None
) -> tuple[list[Image.Image], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames: list[Image.Image] = []
    while True:
        if num_frames is not None and len(frames) >= num_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames, fps


def probe_video_frame_count(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if count <= 0:
            count = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                count += 1
    finally:
        cap.release()
    if count <= 0:
        raise RuntimeError(f"No frames found in video file: {video_path}")
    return count


def probe_video_frame_size_and_fps(video_path: str) -> tuple[tuple[int, int], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read a frame from video file: {video_path}")
            height, width = frame.shape[:2]
    finally:
        cap.release()
    return (width, height), fps


def resolve_videoedit_num_frames(
    requested_num_frames: int | None,
    video_input_path: str,
    mask_input_path: str,
) -> int:
    """Validate source alignment, then resolve full or synchronized truncation."""

    video_frames = probe_video_frame_count(video_input_path)
    mask_frames = probe_mask_frame_count(mask_input_path)
    if video_frames != mask_frames:
        raise ValueError(
            "VideoEdit video/mask length mismatch: "
            f"video has {video_frames} frames, mask has {mask_frames}"
        )
    if requested_num_frames is None or requested_num_frames == -1:
        return video_frames
    if requested_num_frames <= 0:
        raise ValueError(
            "num_frames must be a positive integer, -1, or None, "
            f"got {requested_num_frames}"
        )
    return min(requested_num_frames, video_frames)


def get_aligned_size(h: int, w: int, align: int = 16) -> tuple[int, int]:
    return ((h + align - 1) // align) * align, ((w + align - 1) // align) * align


def resize_frames(frames: list[Image.Image], target_h: int, target_w: int) -> list[Image.Image]:
    return [resize_frame(frame, target_h, target_w) for frame in frames]


def resize_frame(frame: Image.Image, target_h: int, target_w: int) -> Image.Image:
    return Image.fromarray(cv2.resize(np.array(frame), (target_w, target_h)))


def _dilate_single(binary: np.ndarray, dilate_px: int) -> np.ndarray:
    if dilate_px <= 0:
        return binary
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
    )
    return cv2.dilate(binary, kernel, iterations=1)


def _scale_single(binary: np.ndarray, scale: float) -> np.ndarray:
    h, w = binary.shape
    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or scale == 1.0:
        return binary
    cx, cy = float(xs.mean()), float(ys.mean())
    matrix = np.array(
        [[scale, 0, cx * (1 - scale)], [0, scale, cy * (1 - scale)]],
        dtype=np.float64,
    )
    return cv2.warpAffine(
        binary,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def expand_mask_frames(
    mask_frames: list[Image.Image],
    dilate_px: int = 15,
    scale: float = 1.2,
    threshold: float = 0.5,
    anchor_idx: int | None = None,
) -> list[Image.Image]:
    """Expand masks, optionally clearing one explicit conditioning anchor."""

    result: list[Image.Image] = []
    for i, mask in enumerate(mask_frames):
        result.append(
            expand_mask_frame(
                mask,
                dilate_px=dilate_px,
                scale=scale,
                threshold=threshold,
                force_zero=i == anchor_idx,
            )
        )
    return result


def expand_mask_frame(
    mask: Image.Image,
    dilate_px: int = 15,
    scale: float = 1.2,
    threshold: float = 0.5,
    force_zero: bool = False,
) -> Image.Image:
    if force_zero:
        return Image.new("L", mask.size, 0)
    gray = np.array(mask.convert("L")).astype(np.float32) / 255.0
    binary = ((gray > threshold) * 255).astype(np.uint8)
    expanded = _scale_single(_dilate_single(binary, dilate_px), scale)
    return Image.fromarray(expanded, mode="L")


def get_mask_bbox(
    mask_frames: list[Image.Image], padding: int = 0
) -> tuple[int, int, int, int] | None:
    all_x_min = all_y_min = float("inf")
    all_x_max = all_y_max = 0
    height = width = None
    for mask in mask_frames:
        mask_np = np.array(mask.convert("L"))
        if height is None:
            height, width = mask_np.shape
        ys, xs = np.where(mask_np > 10)
        if len(ys):
            all_y_min = min(all_y_min, int(ys.min()))
            all_y_max = max(all_y_max, int(ys.max()))
            all_x_min = min(all_x_min, int(xs.min()))
            all_x_max = max(all_x_max, int(xs.max()))
    if all_x_min == float("inf"):
        return None

    crop_w = all_x_max - all_x_min
    crop_h = all_y_max - all_y_min
    cx = (all_x_min + all_x_max) / 2.0
    cy = (all_y_min + all_y_max) / 2.0
    target_w = crop_w + 2 * padding
    target_h = crop_h + 2 * padding
    x_min = int(round(cx - target_w / 2))
    x_max = int(round(cx + target_w / 2))
    y_min = int(round(cy - target_h / 2))
    y_max = int(round(cy + target_h / 2))
    if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
        raise ValueError(
            f"Expanded mask bbox is out of bounds: {(x_min, y_min, x_max, y_max)} "
            f"for frame size {(width, height)}"
        )
    return x_min, y_min, x_max, y_max


def expand_bbox_for_small(
    bbox: tuple[int, int, int, int], height: int, width: int
) -> tuple[int, int, int, int]:
    x_min, y_min, x_max, y_max = bbox
    crop_w = x_max - x_min
    crop_h = y_max - y_min
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    short_side = max(1, min(crop_w, crop_h))
    scale = 480 / short_side
    new_w = min(crop_w * scale, width)
    new_h = min(crop_h * scale, height)
    x_min = int(cx - new_w / 2)
    x_max = int(cx + new_w / 2)
    y_min = int(cy - new_h / 2)
    y_max = int(cy + new_h / 2)
    if x_min < 0:
        x_max -= x_min
        x_min = 0
    if y_min < 0:
        y_max -= y_min
        y_min = 0
    if x_max > width:
        shift = x_max - width
        x_min -= shift
        x_max = width
    if y_max > height:
        shift = y_max - height
        y_min -= shift
        y_max = height
    return max(0, x_min), max(0, y_min), min(width, x_max), min(height, y_max)


def expand_bbox(
    bbox: tuple[int, int, int, int],
    height: int,
    width: int,
    scale: float = 1.6,
) -> tuple[int, int, int, int]:
    """Expand a bbox using the VideoEdit final-size multiplier geometry."""

    if scale <= 0:
        raise ValueError(f"bbox multiplier must be positive, got {scale}")
    x_min, y_min, x_max, y_max = bbox
    crop_w = x_max - x_min
    crop_h = y_max - y_min
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    target_h = crop_h * scale
    target_w = crop_w * scale
    target_area = target_h * target_w

    if target_h > height and target_w > width:
        target_h, target_w = float(height), float(width)
    elif target_h > height:
        target_h = float(height)
        target_w = min(float(width), target_area / height)
    elif target_w > width:
        target_w = float(width)
        target_h = min(float(height), target_area / width)

    target_h_i = round(target_h)
    target_w_i = round(target_w)
    new_x_min = round(cx - target_w / 2)
    new_y_min = round(cy - target_h / 2)
    new_x_max = new_x_min + target_w_i
    new_y_max = new_y_min + target_h_i

    if new_x_min < 0:
        new_x_max -= new_x_min
        new_x_min = 0
    if new_y_min < 0:
        new_y_max -= new_y_min
        new_y_min = 0
    if new_x_max > width:
        new_x_min -= new_x_max - width
        new_x_max = width
    if new_y_max > height:
        new_y_min -= new_y_max - height
        new_y_max = height

    return max(0, new_x_min), max(0, new_y_min), new_x_max, new_y_max


def _finalize_bbox_geometry(
    bbox: tuple[int, int, int, int],
    *,
    height: int,
    width: int,
    align: int,
    bbox_expand_scale: float,
) -> dict[str, int | tuple[int, int, int, int]]:
    algorithm_multiplier = 2 * bbox_expand_scale + 1
    bbox = expand_bbox(bbox, height, width, scale=algorithm_multiplier)
    x_min, y_min, x_max, y_max = bbox
    crop_w, crop_h = x_max - x_min, y_max - y_min
    area_ratio = (crop_w * crop_h) / float(height * width)
    short_side = min(crop_w, crop_h)
    if area_ratio < 0.2 and short_side < 480:
        bbox = expand_bbox_for_small(bbox, height, width)
        x_min, y_min, x_max, y_max = bbox
        crop_w, crop_h = x_max - x_min, y_max - y_min
    aligned_h, aligned_w = get_aligned_size(crop_h, crop_w, align)
    return {
        "bbox": bbox,
        "crop_h": crop_h,
        "crop_w": crop_w,
        "aligned_h": aligned_h,
        "aligned_w": aligned_w,
    }


def crop_frames(
    frames: list[Image.Image], bbox: tuple[int, int, int, int]
) -> list[Image.Image]:
    return [crop_frame(frame, bbox) for frame in frames]


def crop_frame(frame: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    x_min, y_min, x_max, y_max = bbox
    return Image.fromarray(np.array(frame)[y_min:y_max, x_min:x_max])


def create_masked_video(
    video_frames: list[Image.Image],
    mask_frames: list[Image.Image],
    binarize_threshold: int = 128,
    preserve_first_frame: bool = True,
) -> list[Image.Image]:
    masked: list[Image.Image] = []
    for i, (frame, mask) in enumerate(zip(video_frames, mask_frames, strict=True)):
        if preserve_first_frame and i == 0:
            masked.append(frame.copy())
            continue
        frame_np = np.array(frame)
        mask_np = (np.array(mask.convert("L")) > binarize_threshold).astype(np.float32)
        masked.append(Image.fromarray((frame_np * (1 - mask_np[:, :, None])).astype(np.uint8)))
    return masked


def create_mask_video(
    mask_frames: list[Image.Image], preserve_first_frame: bool = True
) -> list[Image.Image]:
    processed: list[Image.Image] = []
    for i, mask in enumerate(mask_frames):
        processed.append(
            Image.new("L", mask.size, 0)
            if preserve_first_frame and i == 0
            else mask.convert("L")
        )
    return processed


def frames_to_tensor(frames: list[Image.Image], normalize: bool = True) -> torch.Tensor:
    tensors = []
    for frame in frames:
        tensor = torch.from_numpy(np.array(frame)).float()
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(-1)
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor / 127.5 - 1.0 if normalize else tensor / 255.0
        tensors.append(tensor)
    return torch.stack(tensors)


def scan_global_bbox(
    input_video: str,
    mask_video: str,
    num_frames: int | None = None,
    reference_image: str | None = None,
    bbox_padding: int = 0,
    dilate_px: int = 0,
    mask_scale: float = 1.0,
    bbox_expand_scale: float = 0.3,
    align: int = 16,
) -> dict:
    """Scan source masks in native ``0..N-1`` coordinates; reference is out-of-band."""

    from nndeploy.e_dit.algorithms.wan.video.stream_decoder import (
        SequentialMaskDecoder,
    )

    frame_size, fps = probe_video_frame_size_and_fps(input_video)
    width, height = frame_size
    video_frame_count = probe_video_frame_count(input_video)
    mask_frame_count = probe_mask_frame_count(mask_video)
    if video_frame_count != mask_frame_count:
        raise ValueError(
            "VideoEdit video/mask length mismatch: "
            f"video has {video_frame_count} frames, mask has {mask_frame_count}"
        )
    frame_limit = None if num_frames is None or num_frames < 0 else num_frames
    raw_num_frames = (
        video_frame_count
        if frame_limit is None
        else min(video_frame_count, frame_limit)
    )
    if raw_num_frames <= 0:
        raise RuntimeError("No frames available for VideoEdit bbox scan")
    all_x_min = all_y_min = float("inf")
    all_x_max = all_y_max = 0
    loaded_raw_frames = 0
    decoder = SequentialMaskDecoder(mask_video, target_size=frame_size)
    try:
        for raw_idx in range(raw_num_frames):
            raw_mask = decoder.read_next()
            if raw_mask is None:
                break
            loaded_raw_frames += 1
            expanded_mask = expand_mask_frame(
                raw_mask,
                dilate_px=dilate_px,
                scale=mask_scale,
                force_zero=False,
            )
            mask_np = np.array(expanded_mask.convert("L"))
            ys, xs = np.where(mask_np > 10)
            if len(ys):
                all_y_min = min(all_y_min, int(ys.min()))
                all_y_max = max(all_y_max, int(ys.max()))
                all_x_min = min(all_x_min, int(xs.min()))
                all_x_max = max(all_x_max, int(xs.max()))
    finally:
        decoder.close()
    if loaded_raw_frames <= 0:
        raise RuntimeError("No mask frames loaded during VideoEdit bbox scan")
    if all_x_min == float("inf"):
        raise RuntimeError("No mask region detected")

    crop_w = all_x_max - all_x_min
    crop_h = all_y_max - all_y_min
    cx = (all_x_min + all_x_max) / 2.0
    cy = (all_y_min + all_y_max) / 2.0
    target_w = crop_w + 2 * bbox_padding
    target_h = crop_h + 2 * bbox_padding
    x_min = int(round(cx - target_w / 2))
    x_max = int(round(cx + target_w / 2))
    y_min = int(round(cy - target_h / 2))
    y_max = int(round(cy + target_h / 2))
    if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
        raise ValueError(
            f"Expanded mask bbox is out of bounds: {(x_min, y_min, x_max, y_max)} "
            f"for frame size {(width, height)}"
        )
    bbox = (x_min, y_min, x_max, y_max)

    geometry = _finalize_bbox_geometry(
        bbox,
        height=height,
        width=width,
        align=align,
        bbox_expand_scale=bbox_expand_scale,
    )
    geometry.update(
        {
            "fps": fps,
            "num_frames": loaded_raw_frames,
            "frame_size": frame_size,
            "frame_width": width,
            "frame_height": height,
        }
    )
    return geometry


def prepare_global_inputs(
    input_video: str,
    mask_video: str,
    num_frames: int | None = None,
    reference_image: str | None = None,
    bbox_padding: int = 0,
    dilate_px: int = 0,
    mask_scale: float = 1.0,
    bbox_expand_scale: float = 0.3,
    align: int = 16,
    debug_dir: str | None = None,
    scanned_geometry: dict | None = None,
) -> dict:
    """Preprocess aligned source frames/masks and return the reference separately."""

    video_frame_count = probe_video_frame_count(input_video)
    mask_frame_count = probe_mask_frame_count(mask_video)
    if video_frame_count != mask_frame_count:
        raise ValueError(
            "VideoEdit video/mask length mismatch: "
            f"video has {video_frame_count} frames, mask has {mask_frame_count}"
        )
    frame_limit = None if num_frames is None or num_frames < 0 else num_frames
    original_frames, fps = load_video_frames(input_video, frame_limit)
    if not original_frames:
        raise RuntimeError("No frames loaded from input video")
    raw_mask_frames = load_mask_frames(
        mask_video,
        num_frames=frame_limit,
        target_size=original_frames[0].size,
    )
    if not raw_mask_frames:
        raise RuntimeError("No frames loaded from input or mask video")
    if len(original_frames) != len(raw_mask_frames):
        raise ValueError(
            "VideoEdit decoded video/mask length mismatch after synchronized "
            f"truncation: video={len(original_frames)}, mask={len(raw_mask_frames)}"
        )
    n = len(original_frames)

    reference_frame: Image.Image | None = None
    if reference_image:
        with Image.open(reference_image) as image:
            reference_frame = image.convert("RGB")
        if reference_frame.size != original_frames[0].size:
            reference_frame = reference_frame.resize(
                original_frames[0].size, Image.Resampling.BICUBIC
            )
    dilated_masks = expand_mask_frames(
        raw_mask_frames,
        dilate_px=dilate_px,
        scale=mask_scale,
        anchor_idx=None,
    )
    height, width = original_frames[0].height, original_frames[0].width
    if scanned_geometry is None:
        bbox = get_mask_bbox(dilated_masks, padding=bbox_padding)
        if bbox is None:
            raise RuntimeError("No mask region detected")
        geometry = _finalize_bbox_geometry(
            bbox,
            height=height,
            width=width,
            align=align,
            bbox_expand_scale=bbox_expand_scale,
        )
    else:
        bbox = tuple(scanned_geometry["bbox"])
        crop_h = int(scanned_geometry.get("crop_h", bbox[3] - bbox[1]))
        crop_w = int(scanned_geometry.get("crop_w", bbox[2] - bbox[0]))
        aligned_h = int(scanned_geometry.get("aligned_h", get_aligned_size(crop_h, crop_w, align)[0]))
        aligned_w = int(scanned_geometry.get("aligned_w", get_aligned_size(crop_h, crop_w, align)[1]))
        geometry = {
            "bbox": bbox,
            "crop_h": crop_h,
            "crop_w": crop_w,
            "aligned_h": aligned_h,
            "aligned_w": aligned_w,
        }
        fps = float(scanned_geometry.get("fps", fps))

    bbox = geometry["bbox"]
    crop_h = int(geometry["crop_h"])
    crop_w = int(geometry["crop_w"])
    aligned_h = int(geometry["aligned_h"])
    aligned_w = int(geometry["aligned_w"])
    cropped_video = crop_frames(original_frames, bbox)
    dilated_cropped_masks = crop_frames(dilated_masks, bbox)
    resized_video = resize_frames(cropped_video, aligned_h, aligned_w)
    resized_masks = resize_frames(dilated_cropped_masks, aligned_h, aligned_w)
    resized_reference = None
    if reference_frame is not None:
        cropped_reference = crop_frame(reference_frame, bbox)
        resized_reference = resize_frame(cropped_reference, aligned_h, aligned_w)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        resized_video[0].save(os.path.join(debug_dir, "global_resized_000.png"))
        resized_masks[0].save(os.path.join(debug_dir, "global_mask_000.png"))
        if resized_reference is not None:
            resized_reference.save(
                os.path.join(debug_dir, "global_resized_reference.png")
            )
    return {
        "original_frames": original_frames,
        "dilated_cropped_masks": dilated_cropped_masks,
        "resized_video": resized_video,
        "resized_masks": resized_masks,
        "resized_reference": resized_reference,
        "bbox": bbox,
        "crop_h": crop_h,
        "crop_w": crop_w,
        "aligned_h": aligned_h,
        "aligned_w": aligned_w,
        "num_frames": n,
        "fps": fps,
    }


def build_videoedit_bridge(
    long_output_frames: Sequence[Image.Image | None],
    bridge_length: int,
) -> tuple[Image.Image, ...]:
    """Build ``long_output[1:1+b][::-1]`` without hiding positional holes."""

    if bridge_length < 1:
        raise ValueError(f"bridge_length must be positive, got {bridge_length}")
    bridge_slice = list(long_output_frames[1 : 1 + bridge_length])
    if len(bridge_slice) != bridge_length or any(
        frame is None for frame in bridge_slice
    ):
        contiguous = 0
        for frame in long_output_frames[1:]:
            if frame is None:
                break
            contiguous += 1
        raise RuntimeError(
            "VideoEdit bridge has holes: "
            f"need {bridge_length} contiguous frames after reference, "
            f"found {contiguous}"
        )
    bridge: list[Image.Image] = []
    for frame in reversed(bridge_slice):
        assert frame is not None
        bridge.append(frame.copy())
    return tuple(bridge)


def materialize_videoedit_pass(
    plan: VideoEditPassPlan,
    *,
    source_frames: Sequence[Image.Image],
    source_masks: Sequence[Image.Image],
    reference_frame: Image.Image | None = None,
    bridge_frames: Sequence[Image.Image] | None = None,
) -> VideoEditSequence:
    """Materialize a pass while keeping conditioning frames out of global indices."""

    if len(source_frames) != len(source_masks):
        raise ValueError(
            "VideoEdit source frame/mask length mismatch: "
            f"frames={len(source_frames)}, masks={len(source_masks)}"
        )
    for index in plan.source_indices:
        if not 0 <= index < len(source_frames):
            raise IndexError(
                f"VideoEdit source index {index} out of range [0,{len(source_frames)})"
            )

    if plan.prefix_kind == "reference":
        if reference_frame is None:
            raise ValueError("Long VideoEdit pass requires an edited reference frame")
        if plan.prefix_length != 1:
            raise ValueError(
                "Reference conditioning prefix must contain exactly one frame, "
                f"got {plan.prefix_length}"
            )
        prefix_frames = (reference_frame.copy(),)
    else:
        if bridge_frames is None:
            raise ValueError("Short VideoEdit pass requires bridge frames")
        if len(bridge_frames) != plan.prefix_length:
            raise ValueError(
                "VideoEdit bridge length mismatch: "
                f"expected {plan.prefix_length}, got {len(bridge_frames)}"
            )
        prefix_frames = tuple(frame.copy() for frame in bridge_frames)

    source_sequence_frames = tuple(
        source_frames[index].copy() for index in plan.source_indices
    )
    source_sequence_masks = tuple(
        source_masks[index].convert("L").copy() for index in plan.source_indices
    )
    if source_sequence_frames:
        expected_size = source_sequence_frames[0].size
    else:
        expected_size = prefix_frames[0].size
    if any(frame.size != expected_size for frame in prefix_frames):
        raise ValueError("VideoEdit conditioning/source frame sizes must match")
    if any(mask.size != expected_size for mask in source_sequence_masks):
        raise ValueError("VideoEdit source frame/mask sizes must match")

    black_prefix_masks = tuple(
        Image.new("L", expected_size, 0) for _ in range(plan.prefix_length)
    )
    return VideoEditSequence(
        frames=prefix_frames + source_sequence_frames,
        masks=black_prefix_masks + source_sequence_masks,
        global_indices=plan.sequence_indices,
    )


def apply_videoedit_overlap(
    window: VideoEditSequence,
    spec: VideoEditWindowSpec,
    previous_output_frames: Sequence[Image.Image] | None,
) -> VideoEditSequence:
    """Replace a later window's complete overlap with the previous output."""

    overlap = spec.overlap_mask_zero_count
    if overlap <= 0:
        return window
    if spec.window_index <= 0 or spec.reference_prev_local_idx is None:
        raise ValueError("Only a later VideoEdit window may propagate overlap")
    if previous_output_frames is None:
        raise ValueError(
            f"VideoEdit window {spec.window_index} requires previous output frames"
        )
    previous_start = spec.reference_prev_local_idx
    propagated = list(
        previous_output_frames[previous_start : previous_start + overlap]
    )
    if len(propagated) != overlap:
        raise ValueError(
            "Previous VideoEdit output does not contain the full overlap slice: "
            f"need [{previous_start}:{previous_start + overlap}], "
            f"got {len(previous_output_frames)} frames"
        )

    frames = [frame.copy() for frame in window.frames]
    masks = [mask.copy() for mask in window.masks]
    target_size = frames[0].size
    for local_idx, propagated_frame in enumerate(propagated):
        frame = propagated_frame
        if frame.size != target_size:
            frame = frame.resize(target_size, Image.Resampling.BICUBIC)
        frames[local_idx] = frame.copy()
        masks[local_idx] = Image.new("L", target_size, 0)
    return VideoEditSequence(
        frames=tuple(frames),
        masks=tuple(masks),
        global_indices=window.global_indices,
    )


def materialize_videoedit_window(
    sequence: VideoEditSequence,
    spec: VideoEditWindowSpec,
    *,
    previous_output_frames: Sequence[Image.Image] | None = None,
) -> VideoEditSequence:
    """Select, reverse-mirror pad, and propagate overlap for one strict window."""

    if not spec.input_indices:
        raise ValueError("VideoEdit window input_indices must not be empty")
    if any(index < 0 or index >= len(sequence.frames) for index in spec.input_indices):
        raise IndexError("VideoEdit window contains an invalid pass-local input index")

    frames = tuple(sequence.frames[index].copy() for index in spec.input_indices)
    masks = tuple(sequence.masks[index].copy() for index in spec.input_indices)
    global_indices = tuple(
        sequence.global_indices[spec.start_index + local_idx]
        if spec.start_index + local_idx < len(sequence.global_indices)
        else None
        for local_idx in range(len(spec.input_indices))
    )
    raw_window = VideoEditSequence(
        frames=frames,
        masks=masks,
        global_indices=global_indices,
    )
    return apply_videoedit_overlap(raw_window, spec, previous_output_frames)


def prepare_window_inputs(
    window_video: list[Image.Image],
    window_masks: list[Image.Image],
    device: str | torch.device,
    dtype: torch.dtype,
    mask_downsample_mode: str = "nearest",
    preserve_first_frame: bool = True,
    return_intermediates: bool = True,
) -> dict:
    if len(window_video) != len(window_masks):
        raise ValueError("Window video and mask length mismatch")
    if mask_downsample_mode != "nearest":
        raise ValueError(
            "mask_downsample_mode must be nearest, "
            f"got {mask_downsample_mode!r}"
        )
    masked_video = create_masked_video(
        window_video, window_masks, preserve_first_frame=preserve_first_frame
    )
    processed_masks = create_mask_video(
        window_masks, preserve_first_frame=preserve_first_frame
    )
    masked_video_tensor = frames_to_tensor(masked_video, normalize=True).to(device=device, dtype=dtype)
    mask_video_tensor = frames_to_tensor(processed_masks, normalize=False).to(device=device, dtype=torch.float32)
    first_frame_mask = mask_video_tensor[0:1].repeat(4, 1, 1, 1)
    expanded_masks = torch.cat([first_frame_mask, mask_video_tensor[1:]], dim=0)
    cond_masks = F.interpolate(
        expanded_masks, scale_factor=1 / 8, mode="nearest"
    )
    cond_masks = (cond_masks < 0.5).float()
    num_mask_frames, _, latent_height, latent_width = cond_masks.shape
    if num_mask_frames % 4 != 0:
        raise ValueError(f"Packed mask frame count must be divisible by 4: {num_mask_frames}")
    cond_masks = cond_masks.view(1, num_mask_frames // 4, 4, latent_height, latent_width)
    cond_masks = cond_masks.transpose(1, 2).contiguous().to(device=device, dtype=dtype)
    return {
        "masked_video_tensor": masked_video_tensor,
        "mask_video_tensor": (
            mask_video_tensor.to(device=device, dtype=dtype) if return_intermediates else None
        ),
        "cond_masks": cond_masks,
        "video_tensor": (
            frames_to_tensor(window_video, normalize=True).to(device=device, dtype=dtype)
            if return_intermediates else None
        ),
        "num_frames": len(window_video),
    }
