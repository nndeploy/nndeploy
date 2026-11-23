# SPDX-License-Identifier: Apache-2.0
"""Pixel-space geometry for VSR: resolution, resize, padding, tile positions.

These are ports of ``infer/utils/video_io.py`` and ``infer/utils/tiling.py`` in
the reference implementation. They are pure functions of their arguments, so
semantic identity is checked directly against the reference by
``verify.compare_ops`` rather than inferred from matching outputs -- see
``docs_always/add_new_mode/add_vsr/requirements.md`` §4.1.3 sampling point 1.

Two conventions worth stating because they are easy to get backwards:

* ``target_resolution`` strings are **H x W**, matching the reference. The
  usual ``W x H`` reading is wrong here.
* Frame tensors are ``[B, C, T, H, W]`` with pixels in ``[-1, 1]``.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F

#: H and W are padded up to a multiple of this. It is the VAE's spatial
#: compression (16) times the DiT's patch size (2).
ALIGN = 32

#: Frames per resize chunk. The reference caps this so a 4K volume does not
#: need one giant contiguous interpolation buffer.
RESIZE_CHUNK = 4


def parse_resolution(text: str) -> Tuple[int, int]:
    """Parse an ``HxW`` string into ``(height, width)``."""
    h, w = text.lower().split("x")
    return int(h), int(w)


def resize_to_long_edge(h: int, w: int, long_edge: int) -> Tuple[int, int]:
    """Aspect-preserving target size so ``max(h, w)`` becomes ``long_edge``.

    Rounds to even, matching the reference (which relies on Python's
    round-half-to-even).
    """
    scale = long_edge / max(h, w)
    new_h = max(2, int(round(h * scale / 2)) * 2)
    new_w = max(2, int(round(w * scale / 2)) * 2)
    return new_h, new_w


def resize_video(frames: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Bicubic-resize ``[B, C, T, H, W]`` to ``(target_h, target_w)``.

    Runs on CPU in fp32 and in chunks of ``RESIZE_CHUNK`` frames: CPU has no
    bicubic kernel for bf16/fp16, and one contiguous 4K buffer can OOM. The
    model-dtype cast happens later, on the GPU.
    """
    B, C, T, H, W = frames.shape
    if H == target_h and W == target_w:
        return frames

    device = frames.device
    frames_cpu = frames.cpu() if device.type != "cpu" else frames

    chunk_size = max(1, min(RESIZE_CHUNK, T))
    out_chunks = []
    for t_start in range(0, T, chunk_size):
        t_end = min(t_start + chunk_size, T)
        chunk = frames_cpu[:, :, t_start:t_end, :, :]
        bt = chunk.shape[2]
        chunk = chunk.permute(0, 2, 1, 3, 4).reshape(B * bt, C, H, W)
        chunk = F.interpolate(chunk, size=(target_h, target_w), mode="bicubic",
                              align_corners=False)
        chunk = chunk.reshape(B, bt, C, target_h, target_w).permute(0, 2, 1, 3, 4)
        out_chunks.append(chunk)

    out = torch.cat(out_chunks, dim=2).contiguous()
    return out.to(device) if device.type != "cpu" else out


def pad_hw_to_multiple(h: int, w: int, align: int = ALIGN) -> Tuple[int, int, int, int]:
    """``(h, w)`` -> ``(padded_h, padded_w, pad_h, pad_w)``.

    Padding is right/bottom only, so a plain crop undoes it with no shift.
    """
    pad_h = (align - h % align) % align
    pad_w = (align - w % align) % align
    return h + pad_h, w + pad_w, pad_h, pad_w


def reflect_pad_time(video: torch.Tensor, target_t: int) -> torch.Tensor:
    """Extend a clip shorter than ``target_t`` frames by reflecting in time.

    The reference's cycle is ``0..T-1`` then back down ``T-2..1``, so index 0
    and ``T-1`` are not repeated at the seam. A single-frame clip repeats that
    one frame.
    """
    T = video.shape[2]
    if T >= target_t:
        return video
    if T == 1:
        idx = [0] * target_t
    else:
        cycle = list(range(T)) + list(range(T - 2, 0, -1))
        idx = [cycle[i % len(cycle)] for i in range(target_t)]
    return video[:, :, idx, :, :]


def compute_tile_positions(length: int, tile_size: int, overlap: int) -> List[Tuple[int, int]]:
    """Half-open ``(start, end)`` windows covering ``[0, length)`` on one axis.

    Stride is ``tile_size - overlap``. The final window is shifted *back* so it
    keeps the full ``tile_size`` whenever ``length >= tile_size`` -- which makes
    the last overlap larger than requested. Callers must read the actual overlap
    back out of these positions rather than assuming ``overlap``.
    """
    if length <= tile_size:
        return [(0, length)]

    stride = max(1, tile_size - overlap)
    positions: List[Tuple[int, int]] = []
    start = 0
    while start < length:
        end = min(start + tile_size, length)
        if end - start < tile_size:
            start = max(0, length - tile_size)
            end = length
        positions.append((start, end))
        if end >= length:
            break
        start += stride

    return positions
