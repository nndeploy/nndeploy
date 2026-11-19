"""EraserDiT mask morphology and temporal compression.

Verbatim port of ``utils/pre.py:243-310`` (``VideoInpaintPre.mask_video_nchw``)
from the frozen baseline at commit ``9944867``.  Two semantics are easy to get
wrong and are deliberately preserved:

* The dilation kernel is a **cross** (one full row plus one full column), applied
  ``dilate_iter`` times with an approximate re-binarisation every time the
  accumulated magnitude would exceed 64000 (``enable_approximate``).
* Mask "on" values are **not 1.0**.  Wrapped latent frames are binarised to
  0/255 and then pushed through a 3-channel grey kernel whose blue coefficient is
  *not* divided by 255, giving ~0.988; the first latent frame keeps the raw
  uint8 0/1 values and comes out at ~0.0039.  Both values must be reproduced
  exactly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nndeploy.e_dit.video.morphology import build_dilate_kernel, cross_dilate

__all__ = [
    "binarize_and_dilate",
    "compress_mask_temporal",
    "gray_normalize_mask",
    "build_dilate_kernel",
    "eraser_dit_mask_channel",
    "ERASERDIT_MASK_CHANNELS",
]

# The baseline mask is an RGB stream; ``mask_video_nchw`` keys the dilation kernel
# on ``mask_align.shape[1]`` and the grey kernel expects three input channels.
ERASERDIT_MASK_CHANNELS = 3

_GRAY_COEFFICIENTS = (0.299 / 255, 0.587 / 255, 0.0004)


def _gray_kernel(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[[[_GRAY_COEFFICIENTS[0]]], [[_GRAY_COEFFICIENTS[1]]], [[_GRAY_COEFFICIENTS[2]]]]],
        dtype=torch.float32,
        device=device,
        requires_grad=False,
    )


def compress_mask_temporal(mask: torch.Tensor, *, head_batch: bool) -> torch.Tensor:
    """LTX causal 8x temporal compression of a ``uint8`` 0/1 mask.

    ``head_batch`` mirrors the baseline's first-window branch: the first frame is
    emitted un-compressed (keeping its 0/1 values) and the remainder is grouped by
    eight with a max (expressed as ``sum >= 1``) followed by a 0/255 binarise.
    """
    if mask.dtype != torch.uint8:
        raise ValueError("compress_mask_temporal expects the uint8 dilation output")
    n_frame, channel, height, width = mask.shape

    if head_batch:
        shift_n_frame = 1
        wrapping_batch_size = (n_frame - 1) // 8
        left_batch = (n_frame - 1) % 8
    else:
        shift_n_frame = 0
        wrapping_batch_size = n_frame // 8
        left_batch = n_frame % 8

    parts: list[torch.Tensor] = []
    if head_batch:
        parts.append(mask[:1])

    wrapped = torch.sum(
        mask[shift_n_frame : shift_n_frame + wrapping_batch_size * 8].reshape(
            wrapping_batch_size, 8, channel, height, width
        ),
        dim=1,
        keepdim=False,
    )
    parts.append(torch.where(wrapped >= 1, 255, 0))

    if left_batch > 0:
        left = torch.sum(mask[-left_batch:], dim=0, keepdim=True)
        parts.append(torch.where(left >= 1, 255, 0))

    return torch.cat(parts, dim=0, out=None).to(torch.float32)


def gray_normalize_mask(mask_latents: torch.Tensor) -> torch.Tensor:
    """Three-channel grey collapse and normalisation (``0.299/255``, ``0.587/255``, ``0.0004``)."""
    if mask_latents.shape[1] != ERASERDIT_MASK_CHANNELS:
        raise ValueError(
            f"grey kernel needs {ERASERDIT_MASK_CHANNELS} channels, "
            f"got {mask_latents.shape[1]}"
        )
    return F.conv2d(
        mask_latents, _gray_kernel(mask_latents.device), stride=1, padding=0
    )


def eraser_dit_mask_channel(
    mask_nchw: torch.Tensor,
    *,
    head_batch: bool,
    ksize: tuple[int, int] = (9, 9),
    dilate_iter: int = 9,
    threshold: float = 0.039,
    enable_approximate: bool = True,
) -> torch.Tensor:
    """Full mask branch of ``mask_video_nchw``: dilate, compress, grey-normalise.

    ``mask_nchw`` is ``[N, C, H, W]`` at frame resolution with raw 0..255 values.
    Returns ``[L, 1, H, W]`` float32 -- the tensor the baseline feeds to the model
    as its single ``mask_values`` channel.
    """
    dilated = binarize_and_dilate(
        mask_nchw,
        ksize=ksize,
        dilate_iter=dilate_iter,
        threshold=threshold,
        enable_approximate=enable_approximate,
    )
    compressed = compress_mask_temporal(dilated, head_batch=head_batch)
    return gray_normalize_mask(compressed)


def binarize_and_dilate(mask, *, ksize=(9, 9), dilate_iter=9, threshold=0.039,
                        enable_approximate=True):
    return cross_dilate(mask, ksize=ksize, dilate_iter=dilate_iter,
                        threshold=threshold, enable_approximate=enable_approximate)
