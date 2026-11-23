# Adapted from SGLang multimodal_gen tests (Apache-2.0).
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from nndeploy.e_dit.algorithms.wan.video.mask_io import (
    load_mask_frames,
    probe_mask_frame_count,
)
from nndeploy.e_dit.algorithms.wan.video.preprocess import (
    prepare_global_inputs,
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


class TestVideoEditMaskSources(unittest.TestCase):
    def assert_binary_frames(self, frames):
        for frame in frames:
            self.assertEqual(frame.mode, "L")
            values = set(np.unique(np.asarray(frame)).tolist())
            self.assertTrue(values.issubset({0, 255}), values)

    def test_load_npy_mask_frames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mask_path = Path(temp_dir) / "mask.npy"
            masks = np.zeros((3, 4, 5), dtype=np.uint8)
            masks[1, 1:3, 2:4] = 1
            np.save(mask_path, masks)

            self.assertEqual(probe_mask_frame_count(str(mask_path)), 3)
            frames = load_mask_frames(str(mask_path), target_size=(10, 8))

            self.assertEqual(len(frames), 3)
            self.assertEqual(frames[0].size, (10, 8))
            self.assert_binary_frames(frames)
            self.assertGreater(np.asarray(frames[1]).sum(), 0)

    def test_load_npz_mask_frames_from_masks_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mask_path = Path(temp_dir) / "mask.npz"
            masks = np.zeros((4, 6, 7, 1), dtype=np.float32)
            masks[2, 2:5, 3:6, 0] = 1.0
            np.savez(mask_path, masks=masks)

            self.assertEqual(probe_mask_frame_count(str(mask_path)), 4)
            frames = load_mask_frames(str(mask_path), num_frames=3)

            self.assertEqual(len(frames), 3)
            self.assert_binary_frames(frames)
            self.assertGreater(np.asarray(frames[2]).sum(), 0)

    def test_num_frames_minus_one_rejects_shorter_numpy_mask(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "video.avi"
            mask_path = Path(temp_dir) / "mask.npy"
            _write_test_video(video_path, 8)
            np.save(mask_path, np.zeros((5, 8, 8), dtype=np.uint8))

            with self.assertRaisesRegex(ValueError, "length mismatch"):
                resolve_videoedit_num_frames(
                    -1, str(video_path), str(mask_path)
                )

    def test_explicit_limit_still_rejects_original_length_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "video.avi"
            mask_path = Path(temp_dir) / "mask.npy"
            _write_test_video(video_path, 8)
            np.save(mask_path, np.ones((5, 8, 8), dtype=np.uint8))

            with self.assertRaisesRegex(ValueError, "length mismatch"):
                prepare_global_inputs(
                    str(video_path),
                    str(mask_path),
                    num_frames=3,
                    dilate_px=0,
                    mask_scale=1.0,
                )

    def test_prepare_global_inputs_keeps_reference_out_of_band(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "video.avi"
            mask_path = Path(temp_dir) / "mask.npy"
            reference_path = Path(temp_dir) / "reference.png"
            _write_test_video(video_path, 3)
            masks = np.ones((3, 8, 8), dtype=np.uint8)
            np.save(mask_path, masks)
            Image.new("RGB", (4, 4), (200, 10, 20)).save(reference_path)

            data = prepare_global_inputs(
                str(video_path),
                str(mask_path),
                num_frames=3,
                reference_image=str(reference_path),
                dilate_px=0,
                mask_scale=1.0,
            )

            self.assertEqual(data["num_frames"], 3)
            self.assertEqual(data["original_frames"][0].size, (8, 8))
            self.assertEqual(np.asarray(data["original_frames"][0])[0, 0, 0], 0)
            self.assertGreater(int(np.asarray(data["dilated_cropped_masks"][0]).sum()), 0)
            self.assertGreater(int(np.asarray(data["resized_masks"][0]).sum()), 0)
            self.assertIsNotNone(data["resized_reference"])
            self.assertEqual(
                int(np.asarray(data["resized_reference"])[0, 0, 0]), 200
            )

    def test_load_current_coco_rle_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mask_path = Path(temp_dir) / "mask.json"
            first = np.zeros((5, 6), dtype=np.uint8)
            second_a = np.zeros((5, 6), dtype=np.uint8)
            second_b = np.zeros((5, 6), dtype=np.uint8)
            second_a[1:3, 2:4] = 1
            second_b[3:5, 0:2] = 1

            def encode(mask: np.ndarray) -> str:
                flat = mask.reshape(-1, order="F")
                counts = []
                last = 0
                run = 0
                for value in flat:
                    value = int(value > 0)
                    if value == last:
                        run += 1
                    else:
                        counts.append(run)
                        run = 1
                        last = value
                counts.append(run)

                encoded = []
                for i, count in enumerate(counts):
                    value = count
                    if i > 2:
                        value -= counts[i - 2]
                    more = True
                    while more:
                        char_value = value & 0x1F
                        value >>= 5
                        more = (
                            value != -1 if char_value & 0x10 else value != 0
                        )
                        if more:
                            char_value |= 0x20
                        encoded.append(chr(char_value + 48))
                return "".join(encoded)

            payload = [
                {
                    "frame": 0,
                    "size": [5, 6],
                    "counts": [{"object_id": 1, "mask": encode(first)}],
                },
                {
                    "frame": 1,
                    "size": [5, 6],
                    "counts": [
                        {"object_id": 1, "mask": encode(second_a)},
                        {"object_id": 2, "mask": encode(second_b)},
                    ],
                },
            ]
            mask_path.write_text(json.dumps(payload), encoding="utf-8")

            self.assertEqual(probe_mask_frame_count(str(mask_path)), 2)
            frames = load_mask_frames(str(mask_path))

            self.assertEqual(len(frames), 2)
            self.assert_binary_frames(frames)
            self.assertEqual(int(np.asarray(frames[0]).sum()), 0)
            self.assertEqual(int(np.asarray(frames[1]).sum()), 8 * 255)


if __name__ == "__main__":
    unittest.main()
