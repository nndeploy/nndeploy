# SPDX-License-Identifier: Apache-2.0
import os

import imageio
import numpy as np
from PIL import Image


def save_video_frames(
    frames: list[Image.Image] | list[np.ndarray],
    path: str,
    fps: float,
    quality: int | float | None = None,
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    arrays = [np.array(frame) if isinstance(frame, Image.Image) else frame for frame in frames]
    mimsave_kwargs = {
        "fps": fps,
        "codec": "libx264",
        "quality": quality if quality is not None else 8,
    }
    if os.path.splitext(path)[1].lower() == ".mp4":
        mimsave_kwargs["format"] = "mp4"
    imageio.mimsave(path, arrays, **mimsave_kwargs)
    return path
