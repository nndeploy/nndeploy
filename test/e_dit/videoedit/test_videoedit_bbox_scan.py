# Adapted from SGLang multimodal_gen tests (Apache-2.0).
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from nndeploy.e_dit.algorithms.wan.video.preprocess import (
    expand_bbox,
    prepare_global_inputs,
    scan_global_bbox,
)


def _write_test_video(path: Path, frame_count: int, size: tuple[int, int]) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError("Could not open test video writer")
    try:
        width, height = size
        for i in range(frame_count):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :, 0] = (i * 10) % 255
            frame[:, :, 1] = 40
            frame[:, :, 2] = 80
            writer.write(frame)
    finally:
        writer.release()


class TestVideoEditBBoxScan(unittest.TestCase):
    def test_bbox_expansion_matches_videoedit_multiplier_geometry(self):
        # Public VideoEdit requests express expansion per side (0.3), while the
        # reference algorithm receives the equivalent final-size multiplier
        # (1 + 2 * 0.3 = 1.6).
        multiplier = 1.6
        self.assertEqual(
            expand_bbox((381, 269, 1492, 736), 1080, 1920, scale=multiplier),
            (48, 129, 1826, 876),
        )

        # Preserve target dimensions by shifting a box that meets an edge.
        self.assertEqual(
            expand_bbox((0, 100, 200, 300), 600, 800, scale=multiplier),
            (0, 40, 320, 360),
        )

        # If one scaled dimension exceeds the frame, compensate in the other
        # dimension to preserve as much of the target area as possible.
        self.assertEqual(
            expand_bbox((300, 100, 500, 500), 500, 800, scale=multiplier),
            (195, 0, 605, 500),
        )

    def test_bbox_scan_matches_eager_prepare_geometry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "video.avi"
            mask_path = Path(temp_dir) / "mask.npy"
            _write_test_video(video_path, frame_count=4, size=(32, 24))

            masks = np.zeros((4, 24, 32), dtype=np.uint8)
            masks[:, 5:20, 4:28] = 1
            np.save(mask_path, masks)

            scanned = scan_global_bbox(
                str(video_path),
                str(mask_path),
                num_frames=4,
                dilate_px=0,
                mask_scale=1.0,
            )
            eager = prepare_global_inputs(
                str(video_path),
                str(mask_path),
                num_frames=4,
                dilate_px=0,
                mask_scale=1.0,
            )

            self.assertEqual(scanned["bbox"], eager["bbox"])
            self.assertEqual(scanned["crop_h"], eager["crop_h"])
            self.assertEqual(scanned["crop_w"], eager["crop_w"])
            self.assertEqual(scanned["aligned_h"], eager["aligned_h"])
            self.assertEqual(scanned["aligned_w"], eager["aligned_w"])
            self.assertEqual(scanned["num_frames"], eager["num_frames"])
            self.assertAlmostEqual(scanned["fps"], eager["fps"])

    def test_prepare_global_inputs_accepts_scanned_geometry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "video.avi"
            mask_path = Path(temp_dir) / "mask.npy"
            _write_test_video(video_path, frame_count=3, size=(32, 24))

            masks = np.zeros((3, 24, 32), dtype=np.uint8)
            masks[:, 4:20, 8:24] = 1
            np.save(mask_path, masks)

            scanned = scan_global_bbox(
                str(video_path),
                str(mask_path),
                num_frames=3,
                dilate_px=0,
                mask_scale=1.0,
            )
            prepared = prepare_global_inputs(
                str(video_path),
                str(mask_path),
                num_frames=3,
                dilate_px=0,
                mask_scale=1.0,
                scanned_geometry=scanned,
            )
            baseline = prepare_global_inputs(
                str(video_path),
                str(mask_path),
                num_frames=3,
                dilate_px=0,
                mask_scale=1.0,
            )

            self.assertEqual(prepared["bbox"], baseline["bbox"])
            self.assertEqual(prepared["aligned_h"], baseline["aligned_h"])
            self.assertEqual(prepared["aligned_w"], baseline["aligned_w"])
            self.assertEqual(prepared["num_frames"], baseline["num_frames"])
            self.assertEqual(len(prepared["resized_video"]), len(baseline["resized_video"]))
            self.assertEqual(len(prepared["resized_masks"]), len(baseline["resized_masks"]))
            self.assertTrue(
                np.array_equal(
                    np.asarray(prepared["resized_video"][0]),
                    np.asarray(baseline["resized_video"][0]),
                )
            )
            self.assertTrue(
                np.array_equal(
                    np.asarray(prepared["resized_masks"][0]),
                    np.asarray(baseline["resized_masks"][0]),
                )
            )


if __name__ == "__main__":
    unittest.main()
