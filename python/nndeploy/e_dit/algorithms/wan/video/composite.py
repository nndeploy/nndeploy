# SPDX-License-Identifier: Apache-2.0
# Adapted from VideoEdit-diffusers streaming inference utilities (2026-09-23).
"""
Post-processing tool for inference: AdaIN boundary color correction + narrow-band edge feather paste back.

Strategy:
  - Inside mask: Use generated content directly (no fusion with original to avoid motion ghosting)
  - Narrow outer edge of mask (feather_px width): Smooth transition to eliminate seam artifacts
  - AdaIN: Color alignment on edge ring to eliminate color differences; always applied, no-op when adain_boundary_dilate=0
"""

import cv2
import numpy as np
from PIL import Image


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


def _adain_boundary(
    gen: np.ndarray,
    orig: np.ndarray,
    mask_bin: np.ndarray,
    boundary_dilate: int = 15,
    feather_ksize: int = 41,
    eps: float = 1e-5,
) -> np.ndarray:
    mask_uint8 = (mask_bin > 0.5).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask_d = cv2.dilate(mask_uint8, kernel, iterations=boundary_dilate)
    boundary = (mask_d - mask_uint8).astype(np.float32)

    if boundary.sum() < 10:
        return gen

    out = gen.copy().astype(np.float32)
    for c in range(3):
        g = gen[:, :, c].astype(np.float32)
        o = orig[:, :, c].astype(np.float32)

        g_sum = mask_bin.sum()
        if g_sum < 1:
            continue
        g_mean = (g * mask_bin).sum() / (g_sum + eps)
        g_std = np.sqrt(((g - g_mean) ** 2 * mask_bin).sum() / (g_sum + eps) + eps)

        b_sum = boundary.sum()
        o_mean = (o * boundary).sum() / (b_sum + eps)
        o_std = np.sqrt(((o - o_mean) ** 2 * boundary).sum() / (b_sum + eps) + eps)

        g_std = max(g_std, 1.0)
        corrected = (g - g_mean) / g_std * o_std + o_mean

        k = feather_ksize if feather_ksize % 2 == 1 else feather_ksize + 1
        weight = cv2.GaussianBlur(boundary, (k, k), sigmaX=k / 3.0)
        out[:, :, c] = g * (1 - weight) + corrected * weight

    return np.clip(out, 0, 255)


def _edge_feather_blend(
    orig_crop: np.ndarray,
    gen: np.ndarray,
    mask_bin: np.ndarray,
    feather_px: int = 12,
) -> np.ndarray:
    k = feather_px * 2 + 1
    feather = cv2.GaussianBlur(
        mask_bin.astype(np.float32), (k, k), sigmaX=feather_px / 2.0
    )
    feather = np.clip(feather, 0, 1)[:, :, None]
    blended = gen * feather + orig_crop * (1 - feather)
    return np.clip(blended, 0, 255)


def color_correct_is_noop() -> bool:
    """Whether ``color_correct=True`` currently differs from ``color_correct=False``.

    It does not: the flag has been dead in :func:`paste_back` since the first commit,
    so the ``*_color.mp4`` output is byte-identical to the plain one. Streaming
    inference queries this to composite each frame once instead of twice, which keeps
    the outputs exactly as they are today without paying for the duplicate work. When
    a real correction is implemented, return False here and the second pass comes
    back automatically.
    """
    return True


def _crop_edge_ramp(h: int, w: int, ramp_px: int) -> np.ndarray:
    """Weight that falls linearly from 1 to 0 over `ramp_px` at each crop edge.

    Multiplied into the mask feather, this makes generated content fade out before it
    reaches the crop rectangle's border, so the border cannot show even when the
    object sits close to it. Off by default (``ramp_px=0``): with enough margin the
    feather has already decayed to zero at the border and the paste is exact.
    """
    if ramp_px <= 0:
        return np.ones((h, w), dtype=np.float32)
    ramp = np.ones((h, w), dtype=np.float32)
    n = min(ramp_px, h // 2, w // 2)
    if n <= 0:
        return ramp
    line = (np.arange(n, dtype=np.float32) + 0.5) / n
    ramp[:n, :] *= line[:, None]
    ramp[-n:, :] *= line[::-1][:, None]
    ramp[:, :n] *= line[None, :]
    ramp[:, -n:] *= line[::-1][None, :]
    return ramp


def paste_back_frame(
    original: Image.Image | np.ndarray,
    generated: Image.Image | np.ndarray,
    mask,
    bbox: tuple[int, int, int, int],
    crop_h: int,
    crop_w: int,
    feather_px: int = 12,
    adain_boundary_dilate: int = 15,
    color_correct: bool = False,
    crop_edge_feather: int = 0,
) -> np.ndarray:
    """Composite one generated crop back into one full-resolution frame.

    Split out of :func:`paste_back` so chunked inference can composite and write each
    frame as it is produced. Every step here is per-frame — the AdaIN statistics come
    from that frame's own mask and boundary ring — so a frame's result does not depend
    on how the sequence is batched.

    ``color_correct`` is accepted and ignored, mirroring :func:`paste_back`, where the
    parameter has never been wired to anything. Callers pass it to produce a
    "colour-corrected" output that is in fact identical to the plain one. Kept as-is
    rather than given an invented implementation; see :func:`color_correct_is_noop`.

    Returns a uint8 RGB array.
    """
    orig_np = np.asarray(original).astype(np.float32)
    # Resize to the crop size in uint8 first and only then to the in-frame size in
    # float, matching paste_back's resize_frames() + cv2.resize() pair exactly. The
    # intermediate quantisation is visible (1/255) if skipped, which would stop the
    # streaming path from being a bit-exact refactor of the batched one.
    gen_u8 = np.asarray(generated, dtype=np.uint8)
    if gen_u8.shape[:2] != (crop_h, crop_w):
        gen_u8 = cv2.resize(gen_u8, (crop_w, crop_h))
    gen_np = gen_u8.astype(np.float32)
    mask_np = _to_float_mask(mask)

    x_min, y_min, _, _ = bbox
    h_full, w_full = orig_np.shape[:2]
    y_end = min(y_min + crop_h, h_full)
    x_end = min(x_min + crop_w, w_full)
    h = y_end - y_min
    w = x_end - x_min
    if h <= 0 or w <= 0:
        return orig_np.astype(np.uint8)

    gen_np = cv2.resize(gen_np, (w, h))
    mask_np = cv2.resize(mask_np, (w, h))
    mask_bin = (mask_np > 0.5).astype(np.float32)

    orig_crop = orig_np[y_min:y_end, x_min:x_end]

    gen_np = _adain_boundary(
        gen_np,
        orig_crop,
        mask_bin,
        boundary_dilate=adain_boundary_dilate,
    )

    k = feather_px * 2 + 1
    feather = cv2.GaussianBlur(
        mask_bin.astype(np.float32), (k, k), sigmaX=feather_px / 2.0
    )
    feather = np.clip(feather, 0, 1) * _crop_edge_ramp(h, w, crop_edge_feather)
    blended = np.clip(
        gen_np * feather[:, :, None] + orig_crop * (1 - feather[:, :, None]), 0, 255
    )

    result_np = orig_np.copy()
    result_np[y_min:y_end, x_min:x_end] = blended
    return result_np.astype(np.uint8)
