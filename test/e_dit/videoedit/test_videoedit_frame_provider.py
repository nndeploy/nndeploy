# Adapted from SGLang multimodal_gen tests (Apache-2.0).
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from nndeploy.e_dit.algorithms.wan.video.frame_provider import (
    WindowFrameProvider,
)
from nndeploy.e_dit.algorithms.wan.video.postprocess import paste_back
from nndeploy.e_dit.algorithms.wan.video.preprocess import (
    prepare_global_inputs,
    scan_global_bbox,
)
from nndeploy.e_dit.algorithms.wan.video.windowing import (
    build_videoedit_pass_window_specs,
    plan_videoedit_passes,
)


def _write_rgb_video(path: Path, frames: list[np.ndarray], fps: float = 5.0) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not open test video writer")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _make_test_frames(frame_count: int, size: tuple[int, int]) -> list[np.ndarray]:
    width, height = size
    frames = []
    for i in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 25) % 255
        frame[:, :, 1] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
        frame[:, :, 2] = np.linspace(255, 0, height, dtype=np.uint8)[:, None]
        frames.append(frame)
    return frames


def _make_mask_frames(frame_count: int, size: tuple[int, int]) -> list[np.ndarray]:
    width, height = size
    masks = []
    for i in range(frame_count):
        mask = np.zeros((height, width, 3), dtype=np.uint8)
        x0 = 4 + i
        x1 = min(width, x0 + 10)
        mask[6:18, x0:x1, :] = 255
        masks.append(mask)
    return masks


class TestVideoEditFrameProvider(unittest.TestCase):
    def test_bbox_scan_matches_eager_for_video_mask(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "video.avi"
            mask_path = Path(temp_dir) / "mask.avi"
            frames = _make_test_frames(frame_count=5, size=(32, 24))
            masks = _make_mask_frames(frame_count=5, size=(32, 24))
            _write_rgb_video(video_path, frames)
            _write_rgb_video(mask_path, masks)

            scanned = scan_global_bbox(
                str(video_path),
                str(mask_path),
                num_frames=5,
                dilate_px=0,
                mask_scale=1.0,
            )
            eager = prepare_global_inputs(
                str(video_path),
                str(mask_path),
                num_frames=5,
                dilate_px=0,
                mask_scale=1.0,
            )

            self.assertEqual(scanned["bbox"], eager["bbox"])
            self.assertEqual(scanned["aligned_h"], eager["aligned_h"])
            self.assertEqual(scanned["aligned_w"], eager["aligned_w"])

    def test_materialize_window_matches_eager_without_prefetch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "video.avi"
            mask_path = Path(temp_dir) / "mask.avi"
            frames = _make_test_frames(frame_count=6, size=(32, 24))
            masks = _make_mask_frames(frame_count=6, size=(32, 24))
            _write_rgb_video(video_path, frames)
            _write_rgb_video(mask_path, masks)

            eager = prepare_global_inputs(
                str(video_path),
                str(mask_path),
                num_frames=6,
                dilate_px=0,
                mask_scale=1.0,
            )
            scanned = scan_global_bbox(
                str(video_path),
                str(mask_path),
                num_frames=6,
                dilate_px=0,
                mask_scale=1.0,
            )
            provider = WindowFrameProvider.from_scanned_geometry(
                video_input_path=str(video_path),
                mask_input_path=str(mask_path),
                reference_image_path=None,
                scanned_geometry=scanned,
                dilate_px=0,
                mask_scale=1.0,
                infer_len=4,
                enable_prefetch=False,
            )
            try:
                frames_out, masks_out = provider.materialize_window([0, 1, 2, 3])
                reflect_frames, reflect_masks = provider.materialize_window([4, 5, 4, 3])
            finally:
                provider.close()

            expected_indices = [0, 1, 2, 3]
            for out, idx in zip(frames_out, expected_indices, strict=True):
                self.assertTrue(
                    np.array_equal(np.asarray(out), np.asarray(eager["resized_video"][idx]))
                )
            for out, idx in zip(masks_out, expected_indices, strict=True):
                self.assertTrue(
                    np.array_equal(np.asarray(out), np.asarray(eager["resized_masks"][idx]))
                )
            reflected_indices = [4, 5, 4, 3]
            for out, idx in zip(reflect_frames, reflected_indices, strict=True):
                self.assertTrue(
                    np.array_equal(np.asarray(out), np.asarray(eager["resized_video"][idx]))
                )
            for out, idx in zip(reflect_masks, reflected_indices, strict=True):
                self.assertTrue(
                    np.array_equal(np.asarray(out), np.asarray(eager["resized_masks"][idx]))
                )

    def test_prefetch_and_stream_paste_back_match_eager(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "video.avi"
            mask_path = Path(temp_dir) / "mask.avi"
            frames = _make_test_frames(frame_count=5, size=(32, 24))
            masks = _make_mask_frames(frame_count=5, size=(32, 24))
            _write_rgb_video(video_path, frames)
            _write_rgb_video(mask_path, masks)

            eager = prepare_global_inputs(
                str(video_path),
                str(mask_path),
                num_frames=5,
                dilate_px=0,
                mask_scale=1.0,
            )
            scanned = scan_global_bbox(
                str(video_path),
                str(mask_path),
                num_frames=5,
                dilate_px=0,
                mask_scale=1.0,
            )
            provider = WindowFrameProvider.from_scanned_geometry(
                video_input_path=str(video_path),
                mask_input_path=str(mask_path),
                reference_image_path=None,
                scanned_geometry=scanned,
                dilate_px=0,
                mask_scale=1.0,
                infer_len=4,
                enable_prefetch=True,
            )
            try:
                frames_out, masks_out = provider.materialize_window([0, 1, 2, 3])
                expected = paste_back(
                    original_frames=eager["original_frames"],
                    generated_frames=eager["resized_video"],
                    mask_frames=eager["dilated_cropped_masks"],
                    bbox=eager["bbox"],
                    crop_h=eager["crop_h"],
                    crop_w=eager["crop_w"],
                    feather_px=12,
                    adain_boundary_dilate=15,
                )
                streamed = provider.paste_back_frames(
                    eager["resized_video"],
                    feather_px=12,
                    adain_boundary_dilate=15,
                )
                thread = provider._prefetch_thread
            finally:
                provider.close()

            self.assertEqual(len(frames_out), 4)
            self.assertEqual(len(masks_out), 4)
            for exp, out in zip(expected, streamed, strict=True):
                self.assertTrue(np.array_equal(np.asarray(exp), np.asarray(out)))
            if thread is not None:
                self.assertFalse(thread.is_alive())

    def test_reference_stays_out_of_band_and_matches_eager(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "video.avi"
            mask_path = Path(temp_dir) / "mask.avi"
            reference_path = Path(temp_dir) / "reference.png"
            frames = _make_test_frames(frame_count=5, size=(32, 24))
            masks = _make_mask_frames(frame_count=5, size=(32, 24))
            _write_rgb_video(video_path, frames)
            _write_rgb_video(mask_path, masks)
            Image.new("RGB", (32, 24), (210, 40, 30)).save(reference_path)

            eager = prepare_global_inputs(
                str(video_path),
                str(mask_path),
                num_frames=5,
                reference_image=str(reference_path),
                dilate_px=0,
                mask_scale=1.0,
            )
            scanned = scan_global_bbox(
                str(video_path),
                str(mask_path),
                num_frames=5,
                reference_image=str(reference_path),
                dilate_px=0,
                mask_scale=1.0,
            )
            provider = WindowFrameProvider.from_scanned_geometry(
                video_input_path=str(video_path),
                mask_input_path=str(mask_path),
                reference_image_path=str(reference_path),
                scanned_geometry=scanned,
                dilate_px=0,
                mask_scale=1.0,
                infer_len=4,
                enable_prefetch=False,
            )
            try:
                frames_out, masks_out = provider.materialize_window([0, 1, 2])
                resized_reference = provider.get_resized_reference_frame()

                long_plan = plan_videoedit_passes(5, 0, bridge_overlap=5).long
                first_spec = build_videoedit_pass_window_specs(
                    long_plan.sequence_indices,
                    infer_len=5,
                    overlap=0,
                )[0]
                pass_window = provider.materialize_pass_window(
                    long_plan, first_spec
                )
                overlap_specs = build_videoedit_pass_window_specs(
                    long_plan.sequence_indices,
                    infer_len=5,
                    overlap=1,
                )
                overlap_previous = [
                    Image.new(
                        "RGB",
                        (scanned["aligned_w"], scanned["aligned_h"]),
                        (230 + i, 230 + i, 230 + i),
                    )
                    for i in range(5)
                ]
                overlap_window = provider.materialize_pass_window(
                    long_plan,
                    overlap_specs[1],
                    previous_output_frames=overlap_previous,
                )

                backward_plans = plan_videoedit_passes(
                    5, 3, bridge_overlap=5
                )
                backward_spec = build_videoedit_pass_window_specs(
                    backward_plans.long.sequence_indices,
                    infer_len=5,
                    overlap=0,
                )[0]
                backward_window = provider.materialize_pass_window(
                    backward_plans.long, backward_spec
                )
                assert backward_plans.short is not None
                short_spec = build_videoedit_pass_window_specs(
                    backward_plans.short.sequence_indices,
                    infer_len=5,
                    overlap=0,
                )[0]
                short_window = provider.materialize_pass_window(
                    backward_plans.short,
                    short_spec,
                    bridge_frames=[
                        Image.new(
                            "RGB",
                            (scanned["aligned_w"], scanned["aligned_h"]),
                            (222, 222, 222),
                        )
                    ],
                )
                eager_paste = paste_back(
                    original_frames=eager["original_frames"],
                    generated_frames=eager["resized_video"],
                    mask_frames=eager["dilated_cropped_masks"],
                    bbox=eager["bbox"],
                    crop_h=eager["crop_h"],
                    crop_w=eager["crop_w"],
                    feather_px=12,
                    adain_boundary_dilate=15,
                )
                stream_paste = provider.paste_back_frames(
                    eager["resized_video"],
                    feather_px=12,
                    adain_boundary_dilate=15,
                )
            finally:
                provider.close()

            for out, expected in zip(frames_out, eager["resized_video"][:3], strict=True):
                self.assertTrue(np.array_equal(np.asarray(out), np.asarray(expected)))
            for out, expected in zip(masks_out, eager["resized_masks"][:3], strict=True):
                self.assertTrue(np.array_equal(np.asarray(out), np.asarray(expected)))
            self.assertGreater(int(np.asarray(masks_out[0]).sum()), 0)
            self.assertIsNotNone(resized_reference)
            assert resized_reference is not None
            self.assertTrue(
                np.array_equal(
                    np.asarray(resized_reference),
                    np.asarray(eager["resized_reference"]),
                )
            )
            self.assertEqual(pass_window.global_indices, (None, 0, 1, 2, 3))
            self.assertEqual(int(np.asarray(pass_window.masks[0]).sum()), 0)
            self.assertGreater(int(np.asarray(pass_window.masks[1]).sum()), 0)
            self.assertEqual(overlap_window.global_indices, (3, 4, None, None, None))
            self.assertEqual(int(np.asarray(overlap_window.frames[0])[0, 0, 0]), 234)
            self.assertEqual(int(np.asarray(overlap_window.masks[0]).sum()), 0)
            self.assertGreater(int(np.asarray(overlap_window.masks[1]).sum()), 0)
            self.assertEqual(overlap_specs[1].commit_local_to_global, {1: 4})
            self.assertEqual(backward_window.global_indices, (None, 3, 2, 1, 0))
            self.assertEqual(int(np.asarray(backward_window.masks[0]).sum()), 0)
            self.assertTrue(
                np.array_equal(
                    np.asarray(backward_window.masks[1]),
                    np.asarray(eager["resized_masks"][3]),
                )
            )
            self.assertEqual(short_window.global_indices, (None, 4, None, None, None))
            self.assertEqual(int(np.asarray(short_window.masks[0]).sum()), 0)
            self.assertGreater(int(np.asarray(short_window.masks[1]).sum()), 0)
            self.assertEqual(int(np.asarray(short_window.frames[0])[0, 0, 0]), 222)
            for expected, actual in zip(
                eager_paste, stream_paste, strict=True
            ):
                self.assertTrue(
                    np.array_equal(np.asarray(expected), np.asarray(actual))
                )


if __name__ == "__main__":
    unittest.main()
