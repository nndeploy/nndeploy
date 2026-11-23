# SPDX-License-Identifier: Apache-2.0
"""Video decode/encode for VSR.

Ports of ``infer/utils/video_io.py``'s I/O half. Decoding stays on **decord**,
matching the reference: the decoder is part of the input path
(``requirements.md`` §4.1.3 sampling point 1), and swapping it for cv2 would
change the pixels before the model ever runs.

The one non-obvious rule here is in :func:`to_uint8_hwc` -- see its docstring.
"""

from __future__ import annotations

from pathlib import Path

import torch

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def probe_video(path) -> tuple[float, int, int, int]:
    """Read ``(fps, T, H, W)`` without decoding the whole file.

    The streaming path needs the geometry up front to plan windows but must not
    materialise the volume.
    """
    import decord

    vr = decord.VideoReader(uri=str(path))
    fps = float(vr.get_avg_fps())
    T = len(vr)
    H, W = int(vr[0].shape[0]), int(vr[0].shape[1])
    del vr
    return fps, T, H, W


def open_video_writer(path, fps: float, crf: int = 5):
    """Open an x264 mp4 writer for incremental ``append_data`` calls.

    The caller owns the handle and must ``close()`` it. Settings match the
    reference exactly -- codec, pixel format, ``crf`` and ``macro_block_size``
    are all part of the delivered bytes.
    """
    import imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        str(path), fps=fps, codec="libx264", pixelformat="yuv420p",
        macro_block_size=None, ffmpeg_params=["-crf", str(crf)],
    )


def to_uint8_hwc(video_bcthw: torch.Tensor):
    """``[1, C, T, H, W]`` in ``[-1, 1]`` -> ``[T, H, W, C]`` uint8 numpy on CPU.

    **The final cast truncates towards zero; it does not round.** ``torch``'s
    float -> uint8 conversion is a C-style cast, so ``0.6 -> 0``,
    ``1.6 -> 1`` and ``254.9 -> 254`` (rounding would give ``1``, ``2``, ``255``).
    Rounding here costs roughly half a grey level on about half the pixels --
    ``mae ~= 0.5``, which is ~20% of the ``max_mae = 2.5`` budget -- and shows
    up as a uniform small bias rather than as noise. See ``requirements.md`` §7-7.
    """
    if video_bcthw.dim() == 5:
        video_bcthw = video_bcthw.squeeze(0)
    out = video_bcthw.permute(1, 2, 3, 0).float()
    out = (out * 0.5 + 0.5).clamp(0, 1)
    return (out * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()


class WindowReader:
    """Single-threaded decord reader for successive frame windows.

    Decord readers are single-thread affine: sharing one across threads either
    crashes or returns wrong frames, so each worker owns its own. The streaming
    core creates one inside its reader thread for exactly that reason.
    """

    def __init__(self, path):
        import decord

        decord.bridge.set_bridge("torch")
        self._vr = decord.VideoReader(uri=str(path))

    def __len__(self) -> int:
        return len(self._vr)

    def read(self, start: int, end: int) -> torch.Tensor:
        """Decode ``[start, end)`` as ``[1, C, n, H, W]`` float32 in ``[-1, 1]``.

        Stays fp32: CPU has no bicubic or replicate-pad kernel for bf16/fp16.
        The cast to the model dtype happens on the GPU.
        """
        x = self._vr.get_batch(list(range(start, end)))  # [n, H, W, 3] uint8
        x = x.float() / 255.0
        x = (x - 0.5) / 0.5
        return x.permute(3, 0, 1, 2).unsqueeze(0)

    def close(self) -> None:
        del self._vr
