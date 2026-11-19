# Adapted from SGLang multimodal_gen tests (Apache-2.0).
import json
import subprocess

import numpy as np
import pytest
from nndeploy.e_dit.algorithms.wan.video.ffmpeg_io import (
    save_video_frames_like_reference,
)


def test_saved_edit_preserves_source_audio_packets(tmp_path):
    source = tmp_path / "source.mp4"
    output = tmp_path / "edited.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=32x32:r=5:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
    )
    save_video_frames_like_reference(
        [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(5)],
        str(output),
        str(source),
        preserve_audio=True,
    )

    def audio_packets(path):
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_packets",
                "-show_data_hash",
                "sha256",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return [p["data_hash"] for p in json.loads(result.stdout)["packets"]]

    assert audio_packets(output) == audio_packets(source)


@pytest.mark.parametrize("audio_duration", [None, 0.3, 2.0])
def test_missing_or_short_audio_does_not_truncate_video(tmp_path, audio_duration):
    source, output = tmp_path / "source.mp4", tmp_path / "edited.mp4"
    command = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=32x32:r=5:d=1"]
    if audio_duration:
        command += [
            "-f",
            "lavfi",
            "-i",
            f"sine=duration={audio_duration}",
            "-c:a",
            "aac",
        ]
    subprocess.run(command + ["-c:v", "libx264", str(source)], check=True)
    save_video_frames_like_reference(
        [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(5)],
        str(output),
        str(source),
        preserve_audio=True,
    )
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout)["streams"]
    assert (
        int(next(s for s in streams if s["codec_type"] == "video")["nb_read_frames"])
        == 5
    )
    assert any(s["codec_type"] == "audio" for s in streams) == bool(audio_duration)


def test_source_audio_start_offset_is_preserved(tmp_path):
    source, output = tmp_path / "offset.mp4", tmp_path / "edited.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=s=32x32:r=5:d=1",
            "-itsoffset",
            "0.3",
            "-f",
            "lavfi",
            "-i",
            "sine=duration=0.5",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
    )
    save_video_frames_like_reference(
        [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(5)],
        str(output),
        str(source),
        preserve_audio=True,
    )

    def start(path):
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=start_time",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(json.loads(result.stdout)["streams"][0]["start_time"])

    assert start(output) == pytest.approx(start(source), abs=0.001)


def test_vfr_audio_is_not_silently_desynchronized(tmp_path):
    source, output = tmp_path / "vfr.mp4", tmp_path / "edited.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=s=32x32:r=5:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=duration=1.5",
            "-vf",
            "setpts=if(lt(N\\,3)\\,N/(5*TB)\\,(N+2)/(5*TB))",
            "-vsync",
            "vfr",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
    )
    with pytest.raises(ValueError, match="constant-frame-rate"):
        save_video_frames_like_reference(
            [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(5)],
            str(output),
            str(source),
            preserve_audio=True,
        )
