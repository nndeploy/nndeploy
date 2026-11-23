# SPDX-License-Identifier: Apache-2.0
"""Real media tests; use an identity restorer to isolate output handling."""

import json
import os
import shlex
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch
from nndeploy.e_dit.algorithms.vsr.stream import stream_restore

pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="requires ffmpeg and ffprobe",
)


def ffmpeg(*args):
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-y", *map(str, args)],
        check=True,
        capture_output=True,
    ).stdout


def probe(path):
    return json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ]
        )
    )


def source_video(tmp_path, *, codec="aac", audio_duration=2, offset=0, tracks=1):
    path = tmp_path / "source.mkv"
    args = ["-f", "lavfi", "-i", "testsrc2=size=32x32:rate=10:duration=2"]
    for i in range(tracks):
        args += [
            "-itsoffset",
            str(offset),
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={440 + i * 440}:duration={audio_duration}",
        ]
    args += ["-map", "0:v:0"]
    for i in range(tracks):
        args += [
            "-map",
            f"{i + 1}:a:0",
            f"-metadata:s:a:{i}",
            "language=eng" if i == 0 else "language=zho",
            f"-disposition:a:{i}",
            "default" if i == tracks - 1 else "0",
        ]
    args += ["-c:v", "libx264", "-c:a", codec, path]
    ffmpeg(*args)
    return path


def restore(source, output, **kwargs):
    restorer = SimpleNamespace(
        tile_t=5,
        t_overlap=0,
        tile_h=32,
        tile_w=32,
        s_overlap=0,
        device=torch.device("cpu"),
        restore_window=lambda x: x,
    )
    return stream_restore(
        restorer,
        source,
        output,
        target_h=32,
        target_w=32,
        color_ref="none",
        show_progress=False,
        **kwargs,
    )


def test_default_preserves_audio_without_changing_video(tmp_path):
    source = source_video(tmp_path)
    output = tmp_path / "result.mp4"
    assert restore(source, output) == 20
    streams = probe(output)["streams"]
    assert [s["codec_type"] for s in streams] == ["video", "audio"]
    assert streams[1]["codec_name"] == "aac"
    silent = tmp_path / "silent.mp4"
    restore(source, silent, preserve_audio=False)
    assert [s["codec_type"] for s in probe(silent)["streams"]] == ["video"]

    def pixels(path):
        return ffmpeg("-i", path, "-map", "0:v:0", "-f", "framemd5", "-")

    assert pixels(output) == pixels(silent)

    def audio_packets(path):
        return json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_packets",
                    "-show_data_hash",
                    "sha256",
                    "-show_entries",
                    "packet=data_hash",
                    "-of",
                    "json",
                    str(path),
                ]
            )
        )["packets"]

    before, after = audio_packets(source), audio_packets(output)
    assert after == before[: len(after)]
    assert len(after) >= len(before) - 2  # trim only packets beyond the video end


@pytest.mark.parametrize("tracks,codec", [(0, "aac"), (2, "aac"), (2, "pcm_s16le")])
def test_audio_tracks_and_container_fallback(tmp_path, tracks, codec, caplog):
    source = source_video(tmp_path, tracks=tracks, codec=codec)
    output = tmp_path / "result.mp4"
    restore(source, output)
    audio = [s for s in probe(output)["streams"] if s["codec_type"] == "audio"]
    assert len(audio) == tracks
    if tracks:
        assert [s["codec_name"] for s in audio] == ["aac", "aac"]
        assert [s["tags"]["language"] for s in audio] == ["eng", "zho"]
        assert [s["disposition"]["default"] for s in audio] == [0, 1]
    if codec == "pcm_s16le":
        assert "transcoding to AAC" in caplog.text


@pytest.mark.parametrize("duration,offset", [(0.5, 0.5), (4, 0), (2, -0.5)])
def test_audio_timeline_does_not_truncate_video(tmp_path, duration, offset):
    source = source_video(
        tmp_path, codec="pcm_s16le", audio_duration=duration, offset=offset
    )
    output = tmp_path / "result.mp4"
    restore(source, output)
    streams = probe(output)["streams"]
    video, audio = streams
    assert int(video["nb_frames"]) == 20
    assert float(video["duration"]) == pytest.approx(2, abs=0.01)
    source_streams = probe(source)["streams"]
    relative_start = float(source_streams[1]["start_time"]) - float(
        source_streams[0]["start_time"]
    )
    assert float(audio["start_time"]) == pytest.approx(max(0, relative_start), abs=0.05)
    assert float(audio["start_time"]) + float(audio["duration"]) <= 2.05


def test_failed_or_cancelled_restore_never_publishes_partial_file(tmp_path):
    source = source_video(tmp_path)
    output = tmp_path / "result.mp4"
    output.write_bytes(b"previous result")

    def cancel():
        raise TimeoutError("cancelled")

    with pytest.raises(TimeoutError, match="cancelled"):
        restore(source, output, check_interrupt=cancel)
    assert output.read_bytes() == b"previous result"
    assert not list(tmp_path.glob(".vsr-*"))


@pytest.mark.parametrize("cancel", [True, False])
def test_mux_cancel_and_failure_keep_previous_output(tmp_path, monkeypatch, cancel):
    from nndeploy.e_dit.algorithms.vsr.audio import video_output

    source = source_video(tmp_path)
    video = tmp_path / "video.mp4"
    ffmpeg("-i", source, "-map", "0:v:0", "-c", "copy", video)
    output = tmp_path / "result.mp4"
    output.write_bytes(b"previous result")
    # Substitute only the external executable, keeping real process lifecycle,
    # probing, staging and publication. The long-running child must be reaped.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "ffmpeg"
    marker = tmp_path / "pid"
    executable.write_text(
        "#!/bin/sh\n"
        + (
            f"echo $$ > {shlex.quote(str(marker))}\nexec sleep 30\n"
            if cancel
            else "echo 'Permission denied' >&2\nexit 1\n"
        )
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])

    def check():
        if marker.exists():
            raise TimeoutError("cancel during mux")

    with pytest.raises(
        TimeoutError if cancel else RuntimeError,
        match="cancel during mux" if cancel else "Permission denied",
    ), video_output(source, output, check_interrupt=check) as staging:
        shutil.copyfile(video, staging)
    assert output.read_bytes() == b"previous result"
    assert not list(tmp_path.glob(".vsr-*"))
    if cancel:
        with pytest.raises(ProcessLookupError):
            os.kill(int(marker.read_text()), 0)


def test_missing_probe_fails_before_inference_but_audio_can_be_disabled(
    tmp_path, monkeypatch
):
    source = source_video(tmp_path)
    output = tmp_path / "result.mp4"
    with monkeypatch.context() as environment:
        environment.setenv("PATH", str(tmp_path))
        with pytest.raises(RuntimeError, match="requires ffprobe"):
            restore(source, output)
        assert not output.exists()
        restore(source, output, preserve_audio=False)
    assert [s["codec_type"] for s in probe(output)["streams"]] == ["video"]
