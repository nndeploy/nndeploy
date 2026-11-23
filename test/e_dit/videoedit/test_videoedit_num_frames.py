# Adapted from SGLang multimodal_gen tests (Apache-2.0).
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from nndeploy.e_dit.algorithms.wan.videoedit_sampling import (
    WanVideoEditSamplingParams,
)
from nndeploy.e_dit.algorithms.wan.video.preprocess import (
    resolve_videoedit_num_frames,
)


def _write_test_video(path: Path, frame_count: int) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (8, 8),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not open test video writer")
    try:
        for i in range(frame_count):
            frame = np.full((8, 8, 3), i % 255, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


class TestVideoEditNumFrames(unittest.TestCase):
    def test_positive_num_frames_validates_inputs_and_clamps_to_available(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video = Path(temp_dir) / "video.avi"
            mask = Path(temp_dir) / "mask.avi"
            _write_test_video(video, 12)
            _write_test_video(mask, 12)

            self.assertEqual(
                resolve_videoedit_num_frames(7, str(video), str(mask)), 7
            )
            self.assertEqual(
                resolve_videoedit_num_frames(81, str(video), str(mask)), 12
            )

    def test_minus_one_resolves_to_all_available_frames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video = Path(temp_dir) / "video.avi"
            mask = Path(temp_dir) / "mask.avi"
            _write_test_video(video, 12)
            _write_test_video(mask, 12)

            self.assertEqual(resolve_videoedit_num_frames(-1, str(video), str(mask)), 12)
            self.assertEqual(resolve_videoedit_num_frames(None, str(video), str(mask)), 12)

    def test_mismatched_video_and_mask_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video = Path(temp_dir) / "video.avi"
            mask = Path(temp_dir) / "mask.avi"
            _write_test_video(video, 12)
            _write_test_video(mask, 7)

            with self.assertRaisesRegex(ValueError, "length mismatch"):
                resolve_videoedit_num_frames(-1, str(video), str(mask))

    def test_minus_one_missing_video_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            resolve_videoedit_num_frames(-1, "missing_video.avi", "missing_mask.avi")

    def test_sampling_params_defaults_to_full_video(self):
        self.assertIsNone(WanVideoEditSamplingParams().num_frames)
        self.assertIsNone(WanVideoEditSamplingParams(num_frames=-1).num_frames)


if __name__ == "__main__":
    unittest.main()
