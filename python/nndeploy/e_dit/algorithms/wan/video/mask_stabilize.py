# SPDX-License-Identifier: Apache-2.0
# Adapted from VideoEdit-diffusers streaming inference utilities (2026-09-23).
"""
Mask temporal stabilization utilities for rigid object editing.

When the camera moves, per-frame object tracking produces jittering masks that
force rigid objects to deform frame-by-frame. These utilities stabilize masks
across time to preserve object rigidity.
"""

import numpy as np
from PIL import Image


def stabilize_mask_union(
    mask_frames: list[Image.Image],
) -> list[Image.Image]:
    """
    Stabilize masks by taking the union across all frames.

    All frames receive the same mask (the spatial union), so rigid objects can
    maintain fixed shape while translating within the unified mask region.

    Args:
        mask_frames: List of PIL Image masks (L mode, white = masked region)

    Returns:
        List of PIL Image masks, all identical (the union)
    """
    if not mask_frames:
        return []

    # Convert all masks to numpy and take max (union)
    mask_arrays = [np.array(m.convert("L")) for m in mask_frames]
    union_mask = np.maximum.reduce(mask_arrays)

    # Return the same union mask for all frames
    union_pil = Image.fromarray(union_mask, mode="L")
    return [union_pil.copy() for _ in range(len(mask_frames))]


# ─────────────────────────── shape-preserving stabilization ───────────────────


def _to_binary(mask: Image.Image) -> np.ndarray:
    """PIL L mask -> uint8 {0,1} (1 = foreground)."""
    return (np.array(mask.convert("L")) > 10).astype(np.uint8)


def _centroid(bin_mask: np.ndarray):
    """Centroid (cx, cy) of a binary mask, or None if empty."""
    ys, xs = np.where(bin_mask > 0)
    if len(ys) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _shift_binary(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate a {0,1} mask by (dx, dy) pixels, zero-filling out-of-bounds."""
    dx = int(round(dx))
    dy = int(round(dy))
    if dx == 0 and dy == 0:
        return mask
    h, w = mask.shape
    out = np.zeros_like(mask)
    src_x0, src_x1 = max(0, -dx), min(w, w - dx)
    dst_x0, dst_x1 = max(0, dx), min(w, w + dx)
    src_y0, src_y1 = max(0, -dy), min(h, h - dy)
    dst_y0, dst_y1 = max(0, dy), min(h, h + dy)
    if src_x1 > src_x0 and src_y1 > src_y0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return out


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union > 0 else 0.0


def _moving_average(values, half_win: int) -> np.ndarray:
    """Centered box moving average with clamped edges."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return values
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half_win)
        hi = min(n, i + half_win + 1)
        out[i] = values[lo:hi].mean()
    return out


def _bidirectional_ma(values, half_win: int) -> np.ndarray:
    """Zero-phase (forward then backward) moving average."""
    return _moving_average(_moving_average(values, half_win)[::-1], half_win)[::-1]


def _fill_nan(arr: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN entries; hold nearest valid value at the ends."""
    arr = np.asarray(arr, dtype=np.float64)
    mask = ~np.isnan(arr)
    if not mask.any():
        return np.zeros_like(arr)
    return np.interp(np.arange(len(arr)), np.arange(len(arr))[mask], arr[mask])


def _find_present_runs(present: np.ndarray) -> list[list[int]]:
    """Maximal runs of True in a boolean array, as [start, end] inclusive."""
    runs = []
    n = len(present)
    i = 0
    while i < n:
        if present[i]:
            j = i
            while j < n and present[j]:
                j += 1
            runs.append([i, j - 1])
            i = j
        else:
            i += 1
    return runs


def _find_segments(present: np.ndarray, gap_bridge: int) -> list[list[int]]:
    """Present runs, merged across absence gaps of length <= gap_bridge."""
    runs = _find_present_runs(present)
    if gap_bridge <= 0 or len(runs) <= 1:
        return runs
    merged = [runs[0]]
    for r in runs[1:]:
        gap = r[0] - merged[-1][1] - 1
        if gap <= gap_bridge:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return merged


def _split_on_jumps(
    bins: list[np.ndarray],
    segments: list[list[int]],
    jump_iou_threshold: float,
) -> list[list[int]]:
    """Split segments further wherever consecutive present frames barely overlap."""
    out: list[list[int]] = []
    for a, b in segments:
        sub_start = a
        for i in range(a, b):
            if _mask_iou(bins[i], bins[i + 1]) <= jump_iou_threshold:
                out.append([sub_start, i])
                sub_start = i + 1
        out.append([sub_start, b])
    return out


def stabilize_mask_shape(
    mask_frames: list[Image.Image],
    window_size: int = 5,
    presence_threshold: int = 10,
    gap_bridge: int = 0,
    jump_iou_threshold: float = 0.0,
    smooth_position: bool = True,
) -> list[Image.Image]:
    """
    Stabilize a mask sequence by smoothing the mask itself (not its bbox).

    The sequence is first split into *present* segments — maximal runs of frames
    whose mask has enough foreground pixels — optionally bridging tiny absence
    gaps (``gap_bridge``) and further splitting wherever consecutive present
    frames barely overlap (``jump_iou_threshold``). Smoothing is applied *within*
    each segment only, so a mask that disappears/reappears, or jumps to an
    unrelated region, is never blended across the boundary.

    Within a segment, each frame's mask is smoothed by a motion-compensated
    temporal vote:

      1. per-frame centroid (the bbox/centroid is only an alignment anchor);
      2. zero-phase (forward+backward) moving average of the centroid trajectory,
         which removes positional jitter with no lag;
      3. for each frame, the masks in its temporal window are translated so their
         centroids coincide, then voted pixel-wise; the winning silhouette is
         placed back at the smoothed centroid.

    Unlike a bbox-based smoother, this preserves the actual object silhouette
    (a consensus of real masks) instead of collapsing it to a filled rectangle,
    while still removing frame-to-frame jitter and jumps.

    Args:
        mask_frames: List of PIL masks (L mode, white = masked region).
        window_size: Temporal window (frames) for the vote & centroid smoothing.
        presence_threshold: Min foreground pixels for a frame to count as present.
        gap_bridge: Max consecutive absent frames that still count as one segment
            (0 = strict: any absence splits). Bridged gaps are filled by the vote.
        jump_iou_threshold: Split a segment where consecutive present frames have
            IoU <= this value (0 = split only on zero overlap; <0 = never split).
        smooth_position: If True, also smooth the centroid trajectory (place the
            voted shape at the smoothed centroid). If False, keep raw centroids.

    Returns:
        List of PIL masks (L mode), same length as the input.
    """
    if not mask_frames:
        return []

    n = len(mask_frames)
    h, w = mask_frames[0].size[::-1]  # PIL size is (w, h)
    bins = [_to_binary(m) for m in mask_frames]
    present = np.array([int(b.sum()) > presence_threshold for b in bins])

    segments = _find_segments(present, gap_bridge)
    if jump_iou_threshold >= 0.0:
        segments = _split_on_jumps(bins, segments, jump_iou_threshold)

    result = [np.zeros((h, w), dtype=np.uint8) for _ in range(n)]
    half_win = window_size // 2

    for a, b in segments:
        idxs = list(range(a, b + 1))
        seg_len = len(idxs)

        # Centroid per frame (auxiliary alignment anchor), NaN-filled across any
        # bridged empty frames so the vote can reconstruct them.
        cx = np.full(seg_len, np.nan)
        cy = np.full(seg_len, np.nan)
        for m, i in enumerate(idxs):
            c = _centroid(bins[i])
            if c is not None:
                cx[m], cy[m] = c
        cx = _fill_nan(cx)
        cy = _fill_nan(cy)

        # Zero-phase smoothing of the centroid trajectory (position jitter).
        sx = _bidirectional_ma(cx, half_win) if smooth_position else cx
        sy = _bidirectional_ma(cy, half_win) if smooth_position else cy

        for k, i in enumerate(idxs):
            stack = []
            for m, j in enumerate(idxs):
                if abs(m - k) > half_win:
                    continue
                # Align mask j onto frame i's centroid, then vote pixel-wise.
                stack.append(_shift_binary(bins[j], cx[k] - cx[m], cy[k] - cy[m]))
            if not stack:
                continue
            voted = (np.median(np.stack(stack, axis=0), axis=0) > 0.5).astype(np.uint8)
            # Place the winning silhouette at the smoothed centroid.
            voted = _shift_binary(voted, sx[k] - cx[k], sy[k] - cy[k])
            result[i] = voted

    return [Image.fromarray((b * 255).astype(np.uint8), mode="L") for b in result]


# ─────────────────────────── chunk-scoped stabilization ──────────────────────


def make_window_stabilizer(
    union: bool = False,
    shape: bool = False,
    window_size: int = 5,
):
    """Build a stabilizer to run per inference chunk, or None if both are off.

    Stabilizing per chunk rather than over the whole video matters most for
    ``union``: a union over every frame of a long shot covers the object's entire
    trajectory, so the "rigid object keeps its shape" gain is paid for with a mask
    that swallows the whole path. A chunk-scoped union only spans one window's worth
    of motion. ``shape`` is already temporally local (its vote window is
    ``window_size``), so for it the change only affects chunk boundaries.

    The returned callable also enforces one invariant the global path never had to:
    **an all-black input mask stays all black.** A chunk's leading frames can be
    conditioning slots — the reference image, bridge frames, frames carried over from
    the previous chunk — whose black mask is what keeps them untouched. Letting the
    vote or the union hand them a mask would erase the very content they exist to
    anchor.
    """
    if not (union or shape):
        return None
    if union and shape:
        raise ValueError("[MaskStabilize] pass at most one of union / shape")

    def stabilize(masks: list[Image.Image]) -> list[Image.Image]:
        if not masks:
            return masks
        blank = [i for i, m in enumerate(masks) if np.array(m.convert("L")).max() <= 10]
        out = (
            stabilize_mask_union(masks)
            if union
            else stabilize_mask_shape(masks, window_size=window_size)
        )
        for i in blank:
            w, h = masks[i].size
            out[i] = Image.new("L", (w, h), 0)
        return out

    return stabilize
