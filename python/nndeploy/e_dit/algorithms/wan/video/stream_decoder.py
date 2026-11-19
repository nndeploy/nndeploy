# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import cv2
from PIL import Image

from nndeploy.e_dit.algorithms.wan.video.mask_io import (
    _detect_mask_type,
    _load_coco_mask_arrays,
    _load_numpy_mask_array,
    _to_binary_pil,
)


class SequentialVideoDecoder:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {video_path}")

    def read_next(self) -> Image.Image | None:
        ok, frame = self._cap.read()
        if not ok:
            return None
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class SequentialMaskDecoder:
    def __init__(
        self,
        mask_path: str,
        target_size: tuple[int, int] | None = None,
    ):
        self.mask_path = mask_path
        self.target_size = target_size
        self.mask_type = _detect_mask_type(mask_path)
        self._next_index = 0
        self._video_decoder: SequentialVideoDecoder | None = None
        self._array_payload = None

        if self.mask_type == "video":
            self._video_decoder = SequentialVideoDecoder(mask_path)
        elif self.mask_type == "numpy":
            self._array_payload = _load_numpy_mask_array(mask_path)
        elif self.mask_type == "coco":
            self._array_payload = _load_coco_mask_arrays(mask_path)
        else:
            raise ValueError(f"Unsupported mask type: {self.mask_type}")

    def read_next(self) -> Image.Image | None:
        if self.mask_type == "video":
            assert self._video_decoder is not None
            frame = self._video_decoder.read_next()
            if frame is None:
                return None
            return _to_binary_pil(frame, target_size=self.target_size)

        assert self._array_payload is not None
        if self._next_index >= len(self._array_payload):
            return None
        frame = self._array_payload[self._next_index]
        self._next_index += 1
        return _to_binary_pil(frame, target_size=self.target_size)

    def close(self) -> None:
        if self._video_decoder is not None:
            self._video_decoder.close()
            self._video_decoder = None
