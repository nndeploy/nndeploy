# Adapted from SGLang multimodal_gen tests (Apache-2.0).
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from nndeploy.e_dit.algorithms.wan.video.ffmpeg_io import (
    _build_ffmpeg_cmd,
    probe_video_profile,
    save_video_frames_like_reference,
)


def _write_reference_video(path: Path, frame_count: int = 3) -> bool:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (8, 8),
    )
    if not writer.isOpened():
        return False
    try:
        for i in range(frame_count):
            frame = np.full((8, 8, 3), i * 40, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return True


def _make_rgb_frames(frame_count: int = 12, size: int = 16) -> list[np.ndarray]:
    base = np.arange(size * size * 3, dtype=np.uint16).reshape(size, size, 3)
    return [((base + i * 17) % 256).astype(np.uint8) for i in range(frame_count)]


def _write_reference_mov(path: Path, frames: list[np.ndarray], fps: int = 6) -> bool:
    height, width = frames[0].shape[:2]
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-b:v",
        "120k",
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_trc",
        "bt709",
        "-color_primaries",
        "bt709",
        str(path),
    ]
    result = subprocess.run(
        cmd,
        input=b"".join(frame.tobytes() for frame in frames),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return result.returncode == 0


def _write_profiled_mp4(path: Path, frames: list[np.ndarray], fps: int = 6) -> bool:
    height, width = frames[0].shape[:2]
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-vf",
        (
            "setparams=range=tv:color_primaries=bt709:"
            "color_trc=bt709:colorspace=bt709,setsar=1/1"
        ),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-b:v",
        "120k",
        str(path),
    ]
    result = subprocess.run(
        cmd,
        input=b"".join(frame.tobytes() for frame in frames),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return result.returncode == 0


class TestVideoEditFfmpegIO(unittest.TestCase):
    def setUp(self):
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg/ffprobe is not available")

    def test_save_video_frames_like_reference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ref_path = Path(temp_dir) / "reference.mp4"
            out_path = Path(temp_dir) / "output.mp4"
            if not _write_reference_video(ref_path):
                self.skipTest("OpenCV could not create reference mp4")

            ref_profile = probe_video_profile(str(ref_path))
            frames = [
                Image.fromarray(np.full((8, 8, 3), i * 30, dtype=np.uint8))
                for i in range(3)
            ]
            save_video_frames_like_reference(
                frames,
                str(out_path),
                refer_file=str(ref_path),
                fps=ref_profile["fps"],
            )
            out_profile = probe_video_profile(str(out_path))

            self.assertEqual(out_profile["codec_name"], ref_profile["codec_name"])
            self.assertEqual(round(out_profile["fps"]), round(ref_profile["fps"]))
            self.assertEqual(out_profile["width"], 8)
            self.assertEqual(out_profile["height"], 8)

    def test_build_ffmpeg_cmd_uses_reference_bitrate_and_color_range(self):
        profile = {
            "codec_name": "h264",
            "pix_fmt": "yuv420p",
            "bit_rate": 123456,
            "color_range": "tv",
            "color_space": "bt709",
            "color_transfer": "bt709",
            "color_primaries": "bt709",
        }

        cmd = _build_ffmpeg_cmd(
            "output.mov",
            width=16,
            height=16,
            fps=6,
            profile=profile,
            quality=None,
            loglevel="error",
        )

        self.assertNotIn("-crf", cmd)
        self.assertEqual(cmd[cmd.index("-b:v") + 1], "123456")
        self.assertEqual(cmd[cmd.index("-pix_fmt", 10) + 1], "yuv420p")
        self.assertEqual(cmd[cmd.index("-color_range") + 1], "tv")
        self.assertEqual(cmd[cmd.index("-colorspace") + 1], "bt709")
        self.assertEqual(cmd[cmd.index("-color_trc") + 1], "bt709")
        self.assertEqual(cmd[cmd.index("-color_primaries") + 1], "bt709")

    def test_build_ffmpeg_cmd_can_match_original_videoedit_writer(self):
        profile = {
            "codec_name": "h264",
            "pix_fmt": "yuv444p",
            "bit_rate": 123456,
            "color_range": "tv",
            "color_space": "bt709",
            "color_transfer": "bt709",
            "color_primaries": "bt709",
        }

        cmd = _build_ffmpeg_cmd(
            "output.mp4",
            width=17,
            height=15,
            fps=50,
            profile=profile,
            quality=None,
            loglevel="error",
            bit_rate=10_000_000,
            copy_color_metadata=False,
        )

        self.assertEqual(cmd[cmd.index("-c:v") + 1], "libx264")
        self.assertEqual(cmd[cmd.index("-b:v") + 1], "10000000")
        self.assertNotIn("-color_range", cmd)
        self.assertNotIn("-colorspace", cmd)
        self.assertNotIn("-color_trc", cmd)
        self.assertNotIn("-color_primaries", cmd)

    def test_save_odd_sized_frames_preserves_crop_geometry(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ref_path = Path(temp_dir) / "reference.mov"
            out_path = Path(temp_dir) / "output.mov"
            frames = _make_rgb_frames()
            if not _write_reference_mov(ref_path, frames):
                self.skipTest("ffmpeg could not create reference mov")

            odd_frames = [
                Image.fromarray(np.full((16, 17, 3), i * 30, dtype=np.uint8))
                for i in range(3)
            ]
            ref_profile = probe_video_profile(str(ref_path))
            save_video_frames_like_reference(
                odd_frames,
                str(out_path),
                refer_file=str(ref_path),
                fps=ref_profile["fps"],
            )
            out_profile = probe_video_profile(str(out_path))

            self.assertEqual(out_profile["codec_name"], ref_profile["codec_name"])
            self.assertEqual(out_profile["pix_fmt"], "yuv444p")
            self.assertEqual(out_profile["width"], 17)
            self.assertEqual(out_profile["height"], 16)

    def test_save_mov_frames_like_reference_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ref_path = Path(temp_dir) / "reference.mov"
            out_path = Path(temp_dir) / "output.mov"
            frames = _make_rgb_frames()
            if not _write_reference_mov(ref_path, frames):
                self.skipTest("ffmpeg could not create reference mov")

            ref_profile = probe_video_profile(str(ref_path))
            save_video_frames_like_reference(
                [Image.fromarray(frame) for frame in frames],
                str(out_path),
                refer_file=str(ref_path),
                fps=ref_profile["fps"],
            )
            out_profile = probe_video_profile(str(out_path))

            self.assertIn("mov", out_profile["format_name"])
            self.assertEqual(out_profile["codec_name"], ref_profile["codec_name"])
            self.assertEqual(out_profile["pix_fmt"], ref_profile["pix_fmt"])
            for field in (
                "color_range",
                "color_space",
                "color_transfer",
                "color_primaries",
            ):
                if ref_profile.get(field) and ref_profile[field] != "unknown":
                    self.assertEqual(out_profile[field], ref_profile[field])
            if ref_profile.get("bit_rate") and out_profile.get("bit_rate"):
                delta = abs(out_profile["bit_rate"] - ref_profile["bit_rate"])
                self.assertLessEqual(delta / ref_profile["bit_rate"], 0.75)

    def test_save_mp4_frames_preserves_aspect_and_color_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            ref_path = Path(temp_dir) / "reference.mp4"
            out_path = Path(temp_dir) / "output.mp4"
            base = np.arange(12 * 16 * 3, dtype=np.uint16).reshape(12, 16, 3)
            frames = [((base + i * 17) % 256).astype(np.uint8) for i in range(12)]
            if not _write_profiled_mp4(ref_path, frames):
                self.skipTest("ffmpeg could not create profiled reference mp4")

            ref_profile = probe_video_profile(str(ref_path))
            expected_profile = {
                "sample_aspect_ratio": "1:1",
                "display_aspect_ratio": "4:3",
                "color_range": "tv",
                "color_space": "bt709",
                "color_transfer": "bt709",
                "color_primaries": "bt709",
            }
            for field, expected in expected_profile.items():
                self.assertEqual(ref_profile[field], expected)

            save_video_frames_like_reference(
                [Image.fromarray(frame) for frame in frames],
                str(out_path),
                refer_file=str(ref_path),
                fps=ref_profile["fps"],
            )
            out_profile = probe_video_profile(str(out_path))

            for field, expected in expected_profile.items():
                self.assertEqual(out_profile[field], expected)


if __name__ == "__main__":
    unittest.main()
