# SPDX-License-Identifier: Apache-2.0
import cv2
import numpy as np
from PIL import Image

from nndeploy.e_dit.algorithms.wan.video.preprocess import resize_frames
from nndeploy.e_dit.algorithms.wan.video.composite import paste_back_frame as composite_frame


def _ensure_2d(mask: np.ndarray) -> np.ndarray:
    return mask[:, :, 0] if mask.ndim == 3 else mask


def _to_float_mask(mask) -> np.ndarray:
    if isinstance(mask, Image.Image):
        arr = np.array(mask.convert("L")).astype(np.float32) / 255.0
    else:
        arr = np.array(mask).astype(np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
    return _ensure_2d(arr)


def _edge_feather_blend(
    orig_crop: np.ndarray,
    gen: np.ndarray,
    mask_bin: np.ndarray,
    feather_px: int = 12,
) -> np.ndarray:
    k = feather_px * 2 + 1
    feather = cv2.GaussianBlur(
        mask_bin.astype(np.float32), (k, k), sigmaX=max(feather_px / 2.0, 0.1)
    )
    feather = np.clip(feather, 0, 1)[:, :, None]
    return np.clip(gen * feather + orig_crop * (1 - feather), 0, 255)


def paste_back(
    original_frames: list[Image.Image],
    generated_frames: list[Image.Image],
    mask_frames: list,
    bbox: tuple[int, int, int, int],
    crop_h: int,
    crop_w: int,
    feather_px: int = 12,
    adain_boundary_dilate: int = 15,
) -> list[Image.Image]:
    gen_resized = resize_frames(generated_frames, crop_h, crop_w)
    result_frames: list[Image.Image] = []
    for orig, gen, mask in zip(original_frames, gen_resized, mask_frames, strict=False):
        result_frames.append(
            paste_back_frame(
                original_frame=orig,
                generated_frame=gen,
                mask_frame=mask,
                bbox=bbox,
                feather_px=feather_px,
                adain_boundary_dilate=adain_boundary_dilate,
            )
        )
    return result_frames


def paste_back_frame(
    original_frame: Image.Image,
    generated_frame: Image.Image,
    mask_frame,
    bbox: tuple[int, int, int, int],
    feather_px: int = 12,
    adain_boundary_dilate: int = 0,
    crop_edge_feather: int = 0,
) -> Image.Image:
    return Image.fromarray(composite_frame(
        original_frame, generated_frame, mask_frame, bbox,
        generated_frame.height, generated_frame.width,
        feather_px=feather_px, adain_boundary_dilate=adain_boundary_dilate,
        crop_edge_feather=crop_edge_feather,
    ))
