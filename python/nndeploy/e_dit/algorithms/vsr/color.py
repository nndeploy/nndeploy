# SPDX-License-Identifier: Apache-2.0
"""Per-channel colour correction for VSR.

Ports of ``infer/utils/video_io.py`` (``color_stats`` / ``match_color`` /
``match_color_to_stats``) and ``infer/stream.py`` (``scan_color_reference``).

AdaIN-style global correction: the output's per-channel mean/std are pushed
towards a reference's. It is applied **per chunk and before temporal blending**,
over the whole chunk including the overlap regions that will later be blended
away -- see ``requirements.md`` §7-5. Both of those details change the output.

A trap worth naming: the two halves of the reference use *different* standard
deviations. ``color_stats`` calls ``torch.std``, which is the **unbiased**
(sample) estimator, while ``scan_color_reference`` accumulates ``E[x^2]-E[x]^2``,
which is the **population** one. They differ by ``sqrt(n/(n-1))``, which is
invisible for large ``n`` but real, and the reference glues the two together.
This port reproduces that exactly rather than "fixing" it.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

#: Keeps the division finite where the measured std is ~0.
COLOR_EPS = 1e-5


def color_stats(video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-channel ``(mean, std)`` of ``[B, C, T, H, W]``, shape ``[1, C, 1, 1, 1]``.

    ``std`` is PyTorch's default, i.e. the unbiased estimator. Matching the
    reference matters more than consistency -- see the module docstring.
    """
    dims = (0, 2, 3, 4)
    return video.mean(dim=dims, keepdim=True), video.std(dim=dims, keepdim=True)


def match_color_to_stats(
    out: torch.Tensor,
    ref_mean: torch.Tensor,
    ref_std: torch.Tensor,
    eps: float = COLOR_EPS,
) -> torch.Tensor:
    """AdaIN-style correction against *pre-computed* reference statistics.

    Split out from :func:`match_color` so the streaming path can apply one fixed
    global reference to every chunk without ever holding the whole source.
    """
    out_mean, out_std = color_stats(out)
    return (out - out_mean) / (out_std + eps) * ref_std.to(out) + ref_mean.to(out)


def match_color(out: torch.Tensor, ref: torch.Tensor, eps: float = COLOR_EPS) -> torch.Tensor:
    """Correction against another tensor's own statistics (``color_ref=chunk``)."""
    ref_mean, ref_std = color_stats(ref)
    return match_color_to_stats(out, ref_mean, ref_std, eps)


@torch.no_grad()
def scan_color_reference(
    path,
    total_frames: int,
    target_h: int,
    target_w: int,
    max_samples: int = 64,
    batch: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Global per-channel ``(mean, std)`` of the *resized* source, in O(1) memory.

    The ``color_ref=global`` pre-pass: evenly spaced frames are decoded, resized
    to the output geometry and folded into running sums. Sampling is safe
    because consecutive frames are highly correlated; ``max_samples=0`` walks
    every frame.

    This recovers the *reference* half of the whole-volume path's colour
    semantics, but not the measured half: the output statistics being corrected
    can only come from the chunk in hand. So ``global`` is **not** numerically
    identical to correcting the whole restored volume at once.

    Frame indices come from ``torch.linspace`` rounded to int, deduplicated and
    sorted -- reproduced here because the sample set is part of the output.

    Returns two ``[1, C, 1, 1, 1]`` float32 tensors.
    """
    import decord

    decord.bridge.set_bridge("torch")

    vr = decord.VideoReader(uri=str(path))
    try:
        if max_samples and max_samples < total_frames:
            idx = torch.linspace(0, total_frames - 1, max_samples)
            idx = sorted({int(v) for v in idx.round().tolist()})
        else:
            idx = list(range(total_frames))

        # float64 accumulators: a 4K frame contributes ~2.5e7 samples per
        # channel and fp32 sums lose precision well before the end.
        total = torch.zeros(3, dtype=torch.float64)
        total_sq = torch.zeros(3, dtype=torch.float64)
        count = 0

        for start in range(0, len(idx), batch):
            picks = idx[start:start + batch]
            x = vr.get_batch(picks).float() / 255.0
            x = (x - 0.5) / 0.5
            x = x.permute(3, 0, 1, 2).unsqueeze(0)
            # Resize before measuring: the whole-volume path computes its stats
            # on the resized volume, and resampling shifts the std slightly.
            from nndeploy.e_dit.algorithms.vsr.geometry import resize_video

            x = resize_video(x, target_h, target_w).double()
            total += x.sum(dim=(0, 2, 3, 4))
            total_sq += (x * x).sum(dim=(0, 2, 3, 4))
            count += x.shape[2] * target_h * target_w
            del x
    finally:
        del vr

    mean = total / count
    # Population std, not torch.std's unbiased estimator. See module docstring.
    std = (total_sq / count - mean * mean).clamp_min(0).sqrt()
    return mean.float().view(1, 3, 1, 1, 1), std.float().view(1, 3, 1, 1, 1)
