# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from fractions import Fraction
from typing import Any

import numpy as np
from PIL import Image


_CODEC_ENCODER_MAP = {
    "h264": "libx264",
    "hevc": "libx265",
    "h265": "libx265",
    "mpeg4": "mpeg4",
    "mjpeg": "mjpeg",
    "prores": "prores_ks",
    "vp8": "libvpx",
    "vp9": "libvpx-vp9",
}


def _parse_fps(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    try:
        return float(Fraction(value))
    except Exception:
        return None


def probe_video_profile(video_path: str) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-select_streams",
        "v:0",
        "-show_streams",
        "-show_format",
        "-print_format",
        "json",
        video_path,
    ]
    result = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    payload = json.loads(result.stdout)
    video_streams = [
        stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"
    ]
    if not video_streams:
        raise ValueError(f"No video stream found in reference file: {video_path}")

    stream = video_streams[0]
    fmt = payload.get("format", {})
    fps = _parse_fps(stream.get("avg_frame_rate")) or _parse_fps(
        stream.get("r_frame_rate")
    )
    bit_rate = stream.get("bit_rate") or fmt.get("bit_rate")

    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "format_name": fmt.get("format_name"),
        "format_long_name": fmt.get("format_long_name"),
        "codec_name": stream.get("codec_name"),
        "codec_tag_string": stream.get("codec_tag_string"),
        "pix_fmt": stream.get("pix_fmt"),
        "sample_aspect_ratio": stream.get("sample_aspect_ratio"),
        "display_aspect_ratio": stream.get("display_aspect_ratio"),
        "bit_rate": int(bit_rate) if bit_rate else None,
        "bits_per_raw_sample": stream.get("bits_per_raw_sample"),
        "bits_per_sample": stream.get("bits_per_sample"),
        "color_range": stream.get("color_range"),
        "color_space": stream.get("color_space"),
        "color_transfer": stream.get("color_transfer"),
        "color_primaries": stream.get("color_primaries"),
        "field_order": stream.get("field_order"),
        "profile": stream.get("profile"),
    }


def _encoder_for_codec(codec_name: str | None) -> str:
    if not codec_name:
        return "libx264"
    return _CODEC_ENCODER_MAP.get(codec_name.lower(), "libx264")


def _quality_to_crf(quality: int | float) -> int:
    quality = max(0.0, min(10.0, float(quality)))
    return int(round(51 - quality * 5.1))


def _aspect_ratio_for_filter(value: str | None) -> str | None:
    if not value or value in {"N/A", "unknown", "0:1", "0/1"}:
        return None
    try:
        ratio = Fraction(value.replace(":", "/"))
    except (ValueError, ZeroDivisionError):
        return None
    if ratio <= 0:
        return None
    return f"{ratio.numerator}/{ratio.denominator}"


def _as_rgb24_array(frame: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(frame, Image.Image):
        arr = np.asarray(frame.convert("RGB"))
    else:
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(f"Unsupported video frame shape: {arr.shape}")

    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return np.ascontiguousarray(arr)


def _build_ffmpeg_cmd(
    output_path: str,
    width: int,
    height: int,
    fps: float,
    profile: dict[str, Any],
    quality: int | float | None,
    loglevel: str,
    bit_rate: int | None = None,
    copy_color_metadata: bool = True,
) -> list[str]:
    encoder = _encoder_for_codec(profile.get("codec_name"))
    pix_fmt = profile.get("pix_fmt") or "yuv420p"
    has_odd_dimension = bool(width % 2 or height % 2)
    preserve_odd_geometry = has_odd_dimension and encoder in {"libx264", "libx265"}
    if preserve_odd_geometry:
        # 4:2:0 requires even dimensions.  VideoEdit crop sidecars are an
        # algorithm boundary, so keep their exact bbox geometry and switch to
        # a 4:4:4 pixel format instead of silently padding one pixel.
        pix_fmt = "yuv444p"
        padded_width, padded_height = width, height
    else:
        padded_width = width + (width % 2)
        padded_height = height + (height % 2)
    video_filters = []
    if padded_width != width or padded_height != height:
        video_filters.append(f"pad={padded_width}:{padded_height}:0:0")

    if copy_color_metadata:
        setparams = []
        for profile_field, filter_option in (
            ("color_range", "range"),
            ("color_primaries", "color_primaries"),
            ("color_transfer", "color_trc"),
            ("color_space", "colorspace"),
        ):
            value = profile.get(profile_field)
            if value and value != "unknown":
                setparams.append(f"{filter_option}={value}")
        if setparams:
            video_filters.append(f"setparams={':'.join(setparams)}")

        sample_aspect_ratio = _aspect_ratio_for_filter(
            profile.get("sample_aspect_ratio")
        )
        if sample_aspect_ratio:
            video_filters.append(f"setsar={sample_aspect_ratio}")

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        loglevel,
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
    ]

    if video_filters:
        cmd.extend(["-vf", ",".join(video_filters)])

    cmd.extend([
        "-c:v",
        encoder,
        "-pix_fmt",
        pix_fmt,
        "-r",
        str(fps),
    ])

    if quality is not None and encoder in {"libx264", "libx265"}:
        cmd.extend(["-crf", str(_quality_to_crf(quality))])
    elif bit_rate is not None:
        cmd.extend(["-b:v", str(bit_rate)])
    elif profile.get("bit_rate"):
        cmd.extend(["-b:v", str(profile["bit_rate"])])

    if copy_color_metadata:
        if profile.get("color_range") and profile["color_range"] != "unknown":
            cmd.extend(["-color_range", profile["color_range"]])
        if profile.get("color_space") and profile["color_space"] != "unknown":
            cmd.extend(["-colorspace", profile["color_space"]])
        if profile.get("color_transfer") and profile["color_transfer"] != "unknown":
            cmd.extend(["-color_trc", profile["color_transfer"]])
        if profile.get("color_primaries") and profile["color_primaries"] != "unknown":
            cmd.extend(["-color_primaries", profile["color_primaries"]])

        field_order = profile.get("field_order")
        if field_order in {"progressive", "tt", "bb", "tb", "bt"}:
            cmd.extend(["-field_order", field_order])

    cmd.append(output_path)
    return cmd


def save_video_frames_like_reference(
    frames: list[Image.Image] | list[np.ndarray],
    output_path: str,
    refer_file: str,
    fps: float | None = None,
    quality: int | float | None = None,
    loglevel: str = "warning",
    bit_rate: int | None = None,
    copy_color_metadata: bool = True,
    preserve_audio: bool = False,
) -> str:
    if not frames:
        raise ValueError("No video frames to save")

    profile = probe_video_profile(refer_file)
    first = _as_rgb24_array(frames[0])
    height, width = first.shape[:2]
    output_fps = float(fps or profile.get("fps") or 24.0)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cmd = _build_ffmpeg_cmd(
        output_path=output_path,
        width=width,
        height=height,
        fps=output_fps,
        profile=profile,
        quality=quality,
        loglevel=loglevel,
        bit_rate=bit_rate,
        copy_color_metadata=copy_color_metadata,
    )

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None

    def _read_stderr() -> str:
        stderr_bytes = process.stderr.read() if process.stderr is not None else b""
        if process.stderr is not None:
            process.stderr.close()
        return stderr_bytes.decode("utf-8", errors="replace")

    def _raise_ffmpeg_error(message: str, return_code: int, stderr: str) -> None:
        details = stderr.strip() or "no ffmpeg stderr"
        raise RuntimeError(
            f"{message} (return code {return_code}): {details}. "
            f"Command: {' '.join(cmd)}"
        )

    try:
        process.stdin.write(first.tobytes())
        for frame in frames[1:]:
            arr = _as_rgb24_array(frame)
            if arr.shape[:2] != (height, width):
                raise ValueError(
                    "All frames must have the same size, got "
                    f"{arr.shape[:2]} and expected {(height, width)}"
                )
            process.stdin.write(arr.tobytes())
        process.stdin.close()
        stderr = _read_stderr()
        return_code = process.wait()
    except BrokenPipeError as exc:
        try:
            process.stdin.close()
        except Exception:
            pass
        stderr = _read_stderr()
        return_code = process.wait()
        _raise_ffmpeg_error("ffmpeg exited early while saving video", return_code, stderr)
    except Exception:
        process.kill()
        process.wait()
        if process.stderr is not None:
            process.stderr.close()
        raise

    if return_code != 0:
        _raise_ffmpeg_error("ffmpeg failed while saving video", return_code, stderr)
    if preserve_audio:
        mux_source_audio(output_path, refer_file, len(frames) / output_fps)
    return output_path


def mux_source_audio(output_path: str, source_path: str, duration: float) -> bool:
    """Copy original audio tracks onto a finished, chronological edit atomically.

    No shortest-stream truncation: a short audio track must not remove video frames.
    Incompatible containers fail explicitly instead of silently dropping audio.
    """
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", source_path],
        check=True, capture_output=True, text=True,
    )
    streams = json.loads(result.stdout)["streams"]
    if not any(s.get("codec_type") == "audio" for s in streams):
        return False
    video = next(s for s in streams if s.get("codec_type") == "video")
    # The RGB encoder has a constant frame cadence. Refuse to silently put
    # unchanged audio on a VFR timeline whose individual frame times moved.
    validate_audio_timeline(source_path, video)
    start = float(video.get("start_time", 0))
    with tempfile.TemporaryDirectory(
        prefix=".videoedit-audio-", dir=os.path.dirname(os.path.abspath(output_path))
    ) as work:
        merged = os.path.join(work, "muxed" + os.path.splitext(output_path)[1])
        command = [
            "ffmpeg", "-v", "error", "-y", "-copyts", "-i", output_path,
            "-itsoffset", str(-start), "-i", source_path,
            "-map", "0:v:0", "-map", "1:a", "-c", "copy",
            "-t", str(duration), "-avoid_negative_ts", "disabled", merged,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Could not preserve source audio: {result.stderr}")
        os.replace(merged, output_path)
    return True


def validate_audio_timeline(source_path: str, video: dict) -> None:
    fps = _parse_fps(video.get("avg_frame_rate"))
    if not fps:
        raise ValueError("Cannot preserve audio without a valid video frame rate")
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", source_path,
    ], check=True, capture_output=True, text=True)
    frames = json.loads(result.stdout)["frames"]
    timestamps = [float(frame["best_effort_timestamp_time"]) for frame in frames]
    if any(abs(t - timestamps[0] - i / fps) > 0.001 for i, t in enumerate(timestamps)):
        raise ValueError("Preserving source audio currently requires constant-frame-rate video; normalize VFR input first")
