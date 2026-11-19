# SPDX-License-Identifier: Apache-2.0
"""Feather weights and weighted accumulation for VSR tile blending.

Ports of ``infer/utils/tiling.py`` (``_axis_weight`` / ``build_blend_mask_3d``)
and ``infer/stream.py`` (``_temporal_weight``). Validated against the reference
by ``verify.compare_ops``.

Blend convention, unchanged from the reference: weight is 1.0 in a window's
non-overlapping core, ramps linearly inside an overlap so the two neighbours sum
to 1 across the seam, and is **not** ramped at the outer border of the whole
volume (there is no neighbour there, and ramping would risk a zero-weight
division). Overlap is applied on all three axes.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

#: Guards the division when a position somehow accumulates no weight at all.
WEIGHT_EPS = 1e-8


def axis_weight(
    extent: int,
    overlap: int,
    ramp_low: bool,
    ramp_high: bool,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """1D feather weight of length ``extent``.

    ``ramp_low`` / ``ramp_high`` say whether this window has a neighbour on that
    side; a border window leaves that side at 1.0.
    """
    w = torch.ones(extent, device=device, dtype=dtype)
    if overlap <= 0:
        return w

    o = min(overlap, extent)
    ramp = torch.linspace(0.0, 1.0, o, device=device, dtype=dtype)
    if ramp_low:
        w[:o] = torch.minimum(w[:o], ramp)
    if ramp_high:
        w[-o:] = torch.minimum(w[-o:], ramp.flip(0))
    return w


def build_blend_mask_3d(
    t: int,
    h: int,
    w: int,
    t_overlap: int,
    s_overlap: int,
    ramp_t: Tuple[bool, bool],
    ramp_h: Tuple[bool, bool],
    ramp_w: Tuple[bool, bool],
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Separable 3D blend mask, shape ``[1, 1, t, h, w]``.

    Channels share one mask.
    """
    wt = axis_weight(t, t_overlap, ramp_t[0], ramp_t[1], device=device, dtype=dtype)
    wh = axis_weight(h, s_overlap, ramp_h[0], ramp_h[1], device=device, dtype=dtype)
    ww = axis_weight(w, s_overlap, ramp_w[0], ramp_w[1], device=device, dtype=dtype)
    mask = wt.view(t, 1, 1) * wh.view(1, h, 1) * ww.view(1, 1, w)
    return mask.view(1, 1, t, h, w)


def temporal_weight(
    tile_t: int,
    ov_prev: int,
    ov_next: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Per-frame feather weight of one temporal window, shape ``[tile_t]``.

    Ramps 0->1 over the *actual* overlap with the previous window and 1->0 over
    the actual overlap with the next; 1.0 in between and at a clip border.

    An overlap of exactly one frame gets **no** ramp: a lone
    ``linspace(0, 1, 1)`` sample is 0.0, which would zero that frame out from
    both sides.
    """
    w = torch.ones(tile_t, dtype=dtype)
    if ov_prev > 1:
        o = min(ov_prev, tile_t)
        w[:o] = torch.minimum(w[:o], torch.linspace(0.0, 1.0, o, dtype=dtype))
    if ov_next > 1:
        o = min(ov_next, tile_t)
        w[-o:] = torch.minimum(w[-o:], torch.linspace(0.0, 1.0, o, dtype=dtype).flip(0))
    return w


def normalize(accum: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted average: ``accum / weight`` with a floor on the divisor.

    Frames covered by three or more windows are averaged here rather than faded
    twice -- see ``requirements.md`` §7-4. The reference relies on this same
    normalisation, and the two disagree when a frame is covered by >= 3 windows.
    """
    return accum / weight.clamp_min(WEIGHT_EPS)


def save_tile_video(path: str, tile: torch.Tensor, fps: float) -> None:
    """Debug aid: write one restored tile ``[1, C, t, h, w]`` as an mp4.

    The tile is written *before* blending, cropping and colour correction, so it
    is a raw model output, not what the deliverable contains.
    """
    import imageio as _imageio

    video = tile.float().squeeze(0).permute(1, 2, 3, 0)          # [t, h, w, C]
    video = (video * 0.5 + 0.5).clamp(0, 1)
    array = (video * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    writer = _imageio.get_writer(
        path, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=None
    )
    for frame in array:
        writer.append_data(frame)
    writer.close()


@torch.no_grad()
def tiled_restore_rect(
    frames: torch.Tensor,
    restore_window_fn,
    *,
    tile_t: int,
    tile_h: int,
    tile_w: int,
    t_overlap: int,
    s_overlap: int,
    pad_mode: str = "replicate",
    save_dir: Optional[str] = None,
    tile_fps: float = 24.0,
    chunk_label: Optional[str] = None,
    show_progress: bool = True,
    restore_windows_fn=None,
    check_interrupt=None,
) -> torch.Tensor:
    """Restore ``[B, C, T, H, W]`` through overlapping rectangular 3D tiles.

    Port of the reference's ``tiled_restore_rect`` (tile_h != tile_w is
    supported because the training crops are non-square).

    Args:
        restore_window_fn: maps ``[B, C, tile_t, tile_h, tile_w]`` -> same shape.
            It moves the window to the compute device itself; the result is
            brought back to ``frames.device`` for accumulation.
        restore_windows_fn: optional ordered iterator mapping for parallel workers.
            It consumes padded windows and yields outputs in exactly input order.
        save_dir: debug only -- each restored tile lands there as an mp4, before
            blending.
        chunk_label: overrides the per-chunk progress label. The streaming caller
            passes one temporal window per call, so "chunk 1/1" would be
            meaningless there.
    """
    from nndeploy.e_dit.algorithms.vsr.geometry import compute_tile_positions

    B, C, T, H, W = frames.shape
    device = frames.device
    out_dtype = frames.dtype

    if save_dir is not None:
        import os

        os.makedirs(save_dir, exist_ok=True)

    accum = torch.zeros((B, C, T, H, W), device=device, dtype=torch.float32)
    weight = torch.zeros((1, 1, T, H, W), device=device, dtype=torch.float32)

    t_pos = compute_tile_positions(T, tile_t, t_overlap)
    h_pos = compute_tile_positions(H, tile_h, s_overlap)
    w_pos = compute_tile_positions(W, tile_w, s_overlap)
    n_patches = len(h_pos) * len(w_pos)

    for ti, (ts, te) in enumerate(t_pos):
        label = chunk_label or f"temporal chunk {ti + 1}/{len(t_pos)}"
        if show_progress:
            print(f"[tiling] {label} frames [{ts}:{te}) | {n_patches} patches "
                  f"({len(h_pos)}x{len(w_pos)})")
        pbar = _progress(n_patches, label if chunk_label else f"chunk {ti + 1}/{len(t_pos)}",
                         show_progress)
        def windows(ts=ts, te=te):
            for hs, he in h_pos:
                for ws, we in w_pos:
                    if check_interrupt is not None:
                        check_interrupt()
                    window = frames[:, :, ts:te, hs:he, ws:we]
                    pad = (0, tile_w - (we - ws), 0, tile_h - (he - hs),
                           0, tile_t - (te - ts))
                    yield F.pad(window, pad, mode=pad_mode) if any(pad) else window

        results = (restore_windows_fn(windows()) if restore_windows_fn is not None
                   else map(restore_window_fn, windows()))
        for hi, (hs, he) in enumerate(h_pos):
            for wi, (ws, we) in enumerate(w_pos):
                ct, ch, cw = te - ts, he - hs, we - ws
                restored = next(results)
                restored = restored[:, :, :ct, :ch, :cw].to(device=device, dtype=torch.float32)

                if save_dir is not None:
                    import os

                    save_tile_video(
                        os.path.join(save_dir, f"tile_t{ti:03d}_h{hi:03d}_w{wi:03d}.mp4"),
                        restored, tile_fps,
                    )

                # Ramps only on sides that have a neighbour; the outer border of
                # the whole volume is left at 1.0.
                mask = build_blend_mask_3d(
                    ct, ch, cw, t_overlap, s_overlap,
                    ramp_t=(ti > 0, ti < len(t_pos) - 1),
                    ramp_h=(hi > 0, hi < len(h_pos) - 1),
                    ramp_w=(wi > 0, wi < len(w_pos) - 1),
                    device=device, dtype=torch.float32,
                )

                accum[:, :, ts:te, hs:he, ws:we] += restored * mask
                weight[:, :, ts:te, hs:he, ws:we] += mask
                if pbar is not None:
                    pbar.update(1)
        if pbar is not None:
            pbar.close()

    out = normalize(accum, weight)
    return out.to(out_dtype)


def _progress(total: int, desc: str, enabled: bool):
    """tqdm if it is installed, otherwise silent (progress is a nicety here)."""
    if not enabled:
        return None
    try:
        from tqdm import tqdm

        return tqdm(total=total, desc=f"  {desc}", unit="patch", leave=False)
    except Exception:
        return None
