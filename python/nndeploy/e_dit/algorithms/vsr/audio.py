# SPDX-License-Identifier: Apache-2.0
"""Publish a complete VSR video with the source audio, without re-encoding video."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


def _run(command, check_interrupt):
    """Poll subprocesses so request cancellation also covers probing and muxing."""
    if check_interrupt:
        check_interrupt()
    with subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        try:
            while True:
                try:
                    stdout, stderr = process.communicate(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    if check_interrupt:
                        check_interrupt()
            if check_interrupt:
                check_interrupt()
            if process.returncode:
                raise subprocess.CalledProcessError(
                    process.returncode,
                    command,
                    stdout,
                    stderr,
                )
            return stdout
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()


def _probe(path, check_interrupt):
    if not shutil.which("ffprobe"):
        raise RuntimeError("VSR audio preservation requires ffprobe on PATH")
    try:
        return json.loads(
            _run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_streams",
                    "-show_format",
                    "-of",
                    "json",
                    str(path),
                ],
                check_interrupt,
            )
        )
    except subprocess.CalledProcessError as error:
        message = error.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"VSR media probe failed for {path}: {message}") from error


def _mux(video, source, output, source_info, check_interrupt):
    audio = [s for s in source_info["streams"] if s["codec_type"] == "audio"]
    source_video = next(s for s in source_info["streams"] if s["codec_type"] == "video")
    start = float(
        source_video.get("start_time", source_info["format"].get("start_time", 0))
    )
    video_info = _probe(video, check_interrupt)
    duration = video_info["format"]["duration"]
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-copyts",
        "-i",
        str(video),
        "-itsoffset",
        str(-start),
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "1:a",
        "-c",
        "copy",
        "-t",
        duration,
        "-avoid_negative_ts",
        "disabled",
    ]
    # Explicit dispositions prevent ffmpeg from inventing a default audio track.
    for i, stream in enumerate(audio):
        flags = "+".join(k for k, v in stream.get("disposition", {}).items() if v)
        command += [f"-disposition:a:{i}", flags or "0"]
        for name in ("language", "title"):
            value = stream.get("tags", {}).get(name)
            if value is not None:
                command += [f"-metadata:s:a:{i}", f"{name}={value}"]
    transcoded = set()
    while True:
        try:
            _run([*command, str(output)], check_interrupt)
            return
        except subprocess.CalledProcessError as error:
            message = error.stderr.decode(errors="replace")
            # Only container/codec incompatibility permits a lossy fallback.
            match = re.search(r"Could not find tag for codec (\w+)", message)
            if not match:
                match = re.search(r"(\w+) in MP4 support is experimental", message)
            codec = match.group(1) if match else None
            indices = [
                i
                for i, s in enumerate(audio)
                if s.get("codec_name") == codec and i not in transcoded
            ]
            if not indices:
                raise RuntimeError(
                    f"VSR audio mux failed: {message.strip()}"
                ) from error
            for i in indices:
                command += [f"-c:a:{i}", "aac", f"-b:a:{i}", "192k"]
                transcoded.add(i)
                logger.warning(
                    "VSR audio track %s (%s) is incompatible with %s; transcoding to AAC",
                    i,
                    codec,
                    output.suffix,
                )


@contextmanager
def video_output(input_path, output_path, *, preserve_audio=True, check_interrupt=None):
    """Yield a video-only staging path; publish atomically after successful mux.

    The source must remain available until this context exits. The output video
    defines the duration: short audio never truncates video. VFR timing remains
    subject to the existing average-FPS video encoder.
    """
    source, output = Path(input_path).resolve(), Path(output_path).resolve()
    if source == output or (output.exists() and os.path.samefile(source, output)):
        raise ValueError("VSR input and output must be different files")
    if check_interrupt:
        check_interrupt()
    info = _probe(source, check_interrupt) if preserve_audio else None
    has_audio = info and any(s["codec_type"] == "audio" for s in info["streams"])
    if has_audio and not shutil.which("ffmpeg"):
        raise RuntimeError("VSR audio preservation requires ffmpeg on PATH")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".vsr-", dir=output.parent) as directory:
        video = Path(directory) / ("video" + output.suffix)
        yield video
        final = video
        if has_audio:
            final = Path(directory) / ("muxed" + output.suffix)
            _mux(video, source, final, info, check_interrupt)
        if check_interrupt:
            check_interrupt()
        os.replace(final, output)
