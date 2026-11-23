# SPDX-License-Identifier: Apache-2.0
# Adapted from VideoEdit-diffusers streaming inference utilities (2026-09-23).
"""Chunk planning for streaming long-video inference.

Inference is organised as *clips* (the long clip and, for a non-zero reference
frame, the short clip) each cut into overlapping *chunks*. Everything here is
decided before a single pixel is generated:

  1. :func:`plan_clips` lays out both clips' frame sequences and their chunks,
     including which global frames each chunk reads and which it owns (writes out).
  2. :func:`scan_mask_bboxes` streams the mask video once and records the per-frame
     bounding box of the processed (dilated + scaled) mask.
  3. :func:`assign_chunk_bboxes` turns those into one crop box per chunk.

Per-chunk crop boxes are the point of the exercise: a single global box has to
cover the object's whole trajectory, which on a long shot wastes most of the
inference resolution on background the model never edits.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from nndeploy.e_dit.algorithms.wan.video.preprocess import (
    expand_bbox,
    expand_bbox_for_small,
    expand_mask_frames,
    get_aligned_size,
)
from nndeploy.e_dit.algorithms.wan.video.windowing import (
    build_videoedit_pass_window_specs,
    plan_videoedit_passes,
)
from nndeploy.e_dit.algorithms.wan.video.windowing import (
    shrink_videoedit_bridge as shrink_bridge,
)


def plan_long_pass(total, k):
    plan = plan_videoedit_passes(total, k)
    return (
        list(plan.long.source_indices),
        plan.long.direction,
        (list(plan.short.source_indices) if plan.short else []),
    )


def plan_window_starts(total, infer_len, overlap):
    return [
        spec.start_index
        for spec in build_videoedit_pass_window_specs(
            [None] * total, infer_len, overlap
        )
    ]


BBox = tuple[int, int, int, int]


# ──────────────────────────────────────────────────────────────────────────────
# Plan structures
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ChunkPlan:
    """One sliding window of one clip."""

    w_idx: int
    seq_start: int
    # Global frame index per window position, in inference order. ``None`` marks a
    # conditioning-only slot: the reference image, a bridge frame, or an overlap frame
    # carried over from the previous chunk. Length is the window's valid (unpadded)
    # length; the tail is mirror-padded downstream.
    context: list[int | None]
    # Half-open window-local range this chunk writes out. Earlier positions are either
    # conditioning slots or frames the previous chunk already owned.
    owned: tuple[int, int]
    # Global source span to decode for this chunk, or None when it reads no source.
    read_range: tuple[int, int] | None

    # Filled in by assign_chunk_bboxes.
    bbox: BBox | None = None
    crop_h: int = 0
    crop_w: int = 0
    aligned_h: int = 0
    aligned_w: int = 0

    @property
    def valid_len(self) -> int:
        return len(self.context)

    def owned_global(self) -> list[int]:
        return [g for g in self.context[self.owned[0] : self.owned[1]] if g is not None]


@dataclass
class ClipPlan:
    """One pass over a contiguous span of the video."""

    label: str
    direction: str
    # Global index per sequence position, in inference order (``None`` = conditioning).
    seq_idx: list[int | None]
    chunks: list[ChunkPlan] = field(default_factory=list)
    # Number of leading conditioning slots (1 reference frame, or the bridge length).
    lead: int = 0

    @property
    def reverse(self) -> bool:
        """True when this clip's inference order runs backwards in global time."""
        real = [g for g in self.seq_idx if g is not None]
        return len(real) > 1 and real[-1] < real[0]

    def first_global(self) -> int:
        """Smallest global index this clip actually writes out."""
        owned = [g for c in self.chunks for g in c.owned_global()]
        if not owned:
            raise RuntimeError(f"[ChunkPlan] clip {self.label} writes no frames")
        return min(owned)

    def read_ranges(self) -> list[tuple[int, int]]:
        return [c.read_range for c in self.chunks if c.read_range is not None]


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: clip / chunk layout
# ──────────────────────────────────────────────────────────────────────────────


def _cut_chunks(
    seq_idx: list[int | None], infer_len: int, overlap: int, chunk_limit: int | None
) -> list[ChunkPlan]:
    starts = plan_window_starts(len(seq_idx), infer_len, overlap)
    if chunk_limit is not None and chunk_limit > 0:
        starts = starts[:chunk_limit]

    chunks = []
    for w_idx, start in enumerate(starts):
        context = seq_idx[start : start + infer_len]
        # Window 0 owns everything from its first frame; later windows re-generate the
        # overlap region for continuity but the previous chunk already wrote it out.
        take_start = 0 if w_idx == 0 else overlap
        take_end = len(context)
        real = [g for g in context if g is not None]
        chunks.append(
            ChunkPlan(
                w_idx=w_idx,
                seq_start=start,
                context=context,
                owned=(take_start, max(take_start, take_end)),
                read_range=(min(real), max(real) + 1) if real else None,
            )
        )
    return chunks


def plan_clips(
    total_frames: int,
    ref_frame_idx: int,
    infer_len: int,
    overlap: int,
    bridge_overlap: int,
    chunk_limit: int | None = None,
) -> list[ClipPlan]:
    """Lay out the long clip and (when the reference is not frame 0) the short clip.

    Clips are returned in *inference* order — the long clip first, since the short
    clip is seeded from its output.

    The bridge length is fixed here rather than after the long clip runs: the long
    clip's window 0 deterministically covers ``min(infer_len, 1 + len(long_idx))``
    sequence positions, so how many generated frames are available to bridge with is
    known upfront.
    """
    long_idx, direction, short_idx = plan_long_pass(total_frames, ref_frame_idx)

    long_seq: list[int | None] = [None] + list(long_idx)
    long_clip = ClipPlan(label="long", direction=direction, seq_idx=long_seq, lead=1)
    long_clip.chunks = _cut_chunks(long_seq, infer_len, overlap, chunk_limit)
    if not long_clip.chunks or long_clip.chunks[0].w_idx != 0:
        raise ValueError(
            f"--chunks={chunk_limit} excludes the long clip's window 0, which holds the "
            f"reference frame and produces the bridge for the short clip"
        )

    clips = [long_clip]
    if short_idx:
        # Window 0 of the long clip emits `valid_len - 1` generated frames after the
        # reference; those, reversed, seed the short clip.
        available = min(infer_len, len(long_seq)) - 1
        bridge_len = shrink_bridge(bridge_overlap, available)
        short_seq: list[int | None] = [None] * bridge_len + list(short_idx)
        short_dir = "forward" if direction == "backward" else "backward"
        short_clip = ClipPlan(
            label="short", direction=short_dir, seq_idx=short_seq, lead=bridge_len
        )
        short_clip.chunks = _cut_chunks(short_seq, infer_len, overlap, chunk_limit)
        clips.append(short_clip)
    return clips


def merge_order(clips: list[ClipPlan]) -> list[tuple[str, bool]]:
    """``(label, reverse)`` per clip, ordered by the first global frame each writes."""
    return [(c.label, c.reverse) for c in sorted(clips, key=lambda c: c.first_global())]


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: mask prescan
# ──────────────────────────────────────────────────────────────────────────────


def scan_mask_bboxes(
    mask_reader,
    dilate_px: int,
    mask_scale: float,
    block: int = 32,
    frame_limit: int | None = None,
) -> list[BBox | None]:
    """Per-frame bounding box of the processed mask, over the whole mask video.

    The masks are run through the real :func:`expand_mask_frames` (binarise, dilate,
    scale-about-centroid) rather than transforming the raw box analytically — the
    scale step pivots on the *dilated* mask's centroid, which is not recoverable from
    a bounding box. Only the four resulting integers per frame are kept, so memory
    stays flat no matter how long the video is.

    Frames are converted RGB -> ``PIL.convert("L")``, matching how the rest of the
    pipeline reads masks, so binarisation lands on exactly the same pixels.
    """
    total = (
        mask_reader.num_frame
        if frame_limit is None
        else min(mask_reader.num_frame, frame_limit)
    )
    boxes: list[BBox | None] = []
    lo = 0
    while lo < total:
        hi = min(total, lo + block)
        frames = [
            Image.fromarray(f).convert("L") for f in mask_reader.read_range(lo, hi)
        ]
        processed = expand_mask_frames(
            frames, dilate_px=dilate_px, scale=mask_scale, anchor_idx=None
        )
        for mask in processed:
            arr = np.array(mask)
            ys, xs = np.where(arr > 10)
            boxes.append(
                (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                if len(ys)
                else None
            )
        lo = hi
    return boxes


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: per-chunk crop boxes
# ──────────────────────────────────────────────────────────────────────────────


def _union(boxes: list[BBox]) -> BBox:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _pad_bbox(bbox: BBox, H: int, W: int, padding: int) -> BBox:
    """Symmetrically grow `bbox` by `padding` px per side, about its centre.

    Mirrors ``get_mask_bbox``'s ``padding`` handling, which the whole-video path
    applies before the expand step — including raising rather than clamping, so an
    over-large padding is reported instead of silently producing a different box.
    """
    if padding <= 0:
        return bbox
    x_min, y_min, x_max, y_max = bbox
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    target_w = (x_max - x_min) + 2 * padding
    target_h = (y_max - y_min) + 2 * padding
    out = (
        int(round(cx - target_w / 2)),
        int(round(cy - target_h / 2)),
        int(round(cx + target_w / 2)),
        int(round(cy + target_h / 2)),
    )
    if out[0] < 0 or out[1] < 0 or out[2] > W or out[3] > H:
        raise ValueError(
            f"[ChunkPlan] Out of bounds caused by bbox_padding={padding}!\n"
            f"Frame size: (H={H}, W={W})\n"
            f"Original bbox: {bbox}\nPadded bbox: {out}\n"
            f"👉 Please reduce --bbox_padding!"
        )
    return out


def _recentre(
    bbox: BBox,
    H: int,
    W: int,
    target_w: int,
    target_h: int,
    keep: BBox | None = None,
) -> BBox:
    """Centre a box in frame, sliding it to contain the inclusive mask box `keep`.

    Alignment can shrink the expanded box below the frame size (1080 to 1072,
    for example). Preserve a mask touching the frame edge whenever it fits.
    """
    target_w = min(target_w, W)
    target_h = min(target_h, H)
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    x0 = int(round(cx - target_w / 2.0))
    y0 = int(round(cy - target_h / 2.0))
    if keep is not None:
        # Mask maxima are inclusive; crop maxima are exclusive.
        if keep[2] - keep[0] + 1 <= target_w:
            x0 = min(max(x0, keep[2] + 1 - target_w), keep[0])
        if keep[3] - keep[1] + 1 <= target_h:
            y0 = min(max(y0, keep[3] + 1 - target_h), keep[1])
    x0 = max(0, min(x0, W - target_w))
    y0 = max(0, min(y0, H - target_h))
    return (x0, y0, x0 + target_w, y0 + target_h)


def _grow_expanded(bbox: BBox, H: int, W: int, expand_scale: float) -> BBox:
    """The existing expand pipeline: scale out, then the small-object rescue."""
    out = expand_bbox(bbox, H, W, scale=expand_scale)
    x0, y0, x1, y1 = out
    crop_w, crop_h = x1 - x0, y1 - y0
    if (crop_w * crop_h) / float(H * W) < 0.2 and min(crop_w, crop_h) < 480:
        out = expand_bbox_for_small(out, H, W)
    return out


def _check_margin(
    tight: BBox, crop: BBox, H: int, W: int, feather_px: int, label: str
) -> None:
    """Warn when the feather ring would reach the crop edge.

    ``paste_back`` blends with a Gaussian of radius ``feather_px`` around the mask and
    then hard-writes the crop rectangle back into the frame. That is seamless only
    because the feather has decayed to zero by the time it reaches the rectangle's
    edge; if it has not, the rectangle's border becomes visible — and with per-chunk
    boxes it would visibly jump between chunks. A side sitting on the frame border has
    no margin to give and is not a defect.
    """
    need = feather_px + 2
    sides = (
        ("left", tight[0] - crop[0], crop[0] <= 0),
        ("top", tight[1] - crop[1], crop[1] <= 0),
        ("right", crop[2] - tight[2], crop[2] >= W),
        ("bottom", crop[3] - tight[3], crop[3] >= H),
    )
    tight_sides = [
        f"{name}={gap}px"
        for name, gap, at_border in sides
        if gap < need and not at_border
    ]
    if tight_sides:
        warnings.warn(
            f"[ChunkPlan] {label}: mask is within {need}px of the crop edge "
            f"({', '.join(tight_sides)}); the paste-back rectangle may show. "
            f"Raise --bbox_expand_scale, or use --crop_edge_feather.",
            stacklevel=2,
        )


def assign_chunk_bboxes(
    clips: list[ClipPlan],
    frame_bboxes: list[BBox | None],
    H: int,
    W: int,
    mode: str = "fixed_size",
    expand_scale: float = 1.6,
    align: int = 16,
    feather_px: int = 8,
    bbox_padding: int = 0,
) -> None:
    """Give every chunk its crop box, in place.

    Modes:

    * ``fixed_size`` (default) — each chunk's box is centred on its own mask union,
      but every chunk uses the same box *size* (the largest any chunk needs, aligned
      up). The model then sees one resolution throughout, as it did in training, and
      chunk boundaries differ only by a translation instead of a rescale.
    * ``tight`` — each chunk gets its own size as well. Best resolution use, but the
      object is generated at a different scale in each chunk, which can show as a
      sharpness step at the boundary, and the crop-only output stops being a single
      constant-size video.
    * ``global`` — one box for every chunk, from the union over all frames. Reproduces
      the pre-streaming geometry exactly; kept for A/B comparison.

    ``fixed_size`` and ``tight`` normally make the crop box a multiple of ``align``.
    If that would clip the mask, keep the full frame extent in that direction and
    resize to the aligned size. ``global`` keeps the old crop-then-resize behavior.

    ``bbox_padding`` grows the mask box before the expand step, matching where
    ``get_mask_bbox`` applies it on the whole-video path.
    """
    if mode not in ("fixed_size", "tight", "global"):
        raise ValueError(f"unknown chunk_bbox_mode={mode!r}")

    all_boxes = [b for b in frame_bboxes if b is not None]
    if not all_boxes:
        raise RuntimeError("[ChunkPlan] No mask region detected in any frame!")

    if mode == "global":
        tight = _pad_bbox(_union(all_boxes), H, W, bbox_padding)
        crop = _grow_expanded(tight, H, W, expand_scale)
        crop_w, crop_h = crop[2] - crop[0], crop[3] - crop[1]
        aligned_h, aligned_w = get_aligned_size(crop_h, crop_w, align)
        _check_margin(tight, crop, H, W, feather_px, "global bbox")
        for clip in clips:
            for chunk in clip.chunks:
                chunk.bbox = crop
                chunk.crop_h, chunk.crop_w = crop_h, crop_w
                chunk.aligned_h, chunk.aligned_w = aligned_h, aligned_w
        return

    # Per-chunk tight union over the frames the chunk touches (its overlap frames
    # included, so a box is never sized for less than the window it conditions on).
    tights: dict[tuple[str, int], BBox] = {}
    grown: dict[tuple[str, int], BBox] = {}
    for clip in clips:
        for chunk in clip.chunks:
            boxes = [
                frame_bboxes[g]
                for g in chunk.context
                if g is not None and frame_bboxes[g] is not None
            ]
            # An all-empty window still needs a box; fall back to the global union.
            tight = _pad_bbox(
                _union(boxes) if boxes else _union(all_boxes), H, W, bbox_padding
            )
            key = (clip.label, chunk.w_idx)
            tights[key] = tight
            grown[key] = _grow_expanded(tight, H, W, expand_scale)

    if mode == "fixed_size":
        target_w = max(b[2] - b[0] for b in grown.values())
        target_h = max(b[3] - b[1] for b in grown.values())
        target_h, target_w = get_aligned_size(target_h, target_w, align)
        target_w = min(target_w, (W // align) * align)
        target_h = min(target_h, (H // align) * align)

    for clip in clips:
        for chunk in clip.chunks:
            key = (clip.label, chunk.w_idx)
            if mode == "fixed_size":
                box = _recentre(
                    grown[key], H, W, target_w, target_h, keep=tights[key]
                )
            else:
                g = grown[key]
                h, w = get_aligned_size(g[3] - g[1], g[2] - g[0], align)
                box = _recentre(
                    g,
                    H,
                    W,
                    min(w, (W // align) * align),
                    min(h, (H // align) * align),
                    keep=tights[key],
                )
            tight = tights[key]
            # A mask spanning more than the aligned-down frame size cannot fit.
            # Keep that direction's full extent and resize rather than cut it off.
            x0, y0, x1, y1 = box
            if tight[0] < x0 or tight[2] >= x1:
                x0, x1 = 0, W
            if tight[1] < y0 or tight[3] >= y1:
                y0, y1 = 0, H
            resampled = (x0, y0, x1, y1) != box
            box = (x0, y0, x1, y1)
            _check_margin(
                tight, box, H, W, feather_px, f"{clip.label} chunk {chunk.w_idx}"
            )
            chunk.bbox = box
            chunk.crop_w, chunk.crop_h = box[2] - box[0], box[3] - box[1]
            if resampled:
                chunk.aligned_h, chunk.aligned_w = get_aligned_size(
                    chunk.crop_h, chunk.crop_w, align
                )
            else:
                chunk.aligned_w, chunk.aligned_h = chunk.crop_w, chunk.crop_h

    if mode == "tight":
        # Each chunk generating at its own resolution is the point of `tight`, but a
        # wide spread means the same object is synthesised at very different scales,
        # which tends to show as a sharpness step where chunks meet.
        areas = [
            (c.crop_w * c.crop_h, c.crop_w, c.crop_h)
            for clip in clips
            for c in clip.chunks
        ]
        lo, hi = min(areas), max(areas)
        if hi[0] > 2.0 * lo[0]:
            warnings.warn(
                f"[ChunkPlan] tight crop sizes vary a lot across chunks "
                f"({lo[1]}x{lo[2]} .. {hi[1]}x{hi[2]}, {hi[0] / lo[0]:.1f}x area): the "
                f"object is generated at different scales, which can show at chunk "
                f"boundaries. Consider --chunk_bbox_mode fixed_size.",
                stacklevel=2,
            )
