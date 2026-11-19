"""Tensor-level video helpers for the local erase stages."""

from __future__ import annotations

from nndeploy.e_dit.utils.env import get_env

import math
import os
import queue
import subprocess
import tempfile
import threading
from collections import deque
from fractions import Fraction
from pathlib import Path
from typing import Any

import ffmpeg
import numpy as np
import torch
import torch.nn.functional as F
import shutil

from nndeploy.e_dit.video.encoding import VideoEncodingProfile


def _free_bytes(path: str) -> int:
    stats = os.statvfs(path)
    return int(stats.f_bavail * stats.f_frsize)


def select_runtime_workdir(preferred: str | None = None) -> str:
    candidates: list[str] = []
    env_value = get_env("NNDEPLOY_DIT_RUNTIME_TMPDIR")
    if preferred:
        candidates.append(preferred)
    if env_value:
        candidates.append(env_value)
    candidates.extend(
        [
            "/dev/shm",
            tempfile.gettempdir(),
        ]
    )

    seen: set[str] = set()
    best_path: str | None = None
    best_free = -1
    for raw_path in candidates:
        if not raw_path:
            continue
        path = os.path.abspath(os.path.expanduser(raw_path))
        if path in seen:
            continue
        seen.add(path)
        try:
            os.makedirs(path, exist_ok=True)
            free = _free_bytes(path)
        except OSError:
            continue
        if free > best_free:
            best_free = free
            best_path = path
    if best_path is None:
        raise RuntimeError("Unable to select a video runtime working directory")
    return best_path


def _safe_fraction(value: str) -> Fraction:
    try:
        return Fraction(value)
    except (ZeroDivisionError, ValueError):
        return Fraction(0, 1)


def ensure_nchw_video(video: torch.Tensor, channels: int | None = None) -> torch.Tensor:
    if video.ndim != 4:
        raise ValueError(f"Expected 4D video tensor [F,C,H,W], got {tuple(video.shape)}")
    if channels is not None and video.shape[1] != channels:
        raise ValueError(
            f"Expected video channel count {channels}, got {video.shape[1]}"
        )
    return video


def crop_video(video: torch.Tensor, bbox: tuple[int, int, int, int]) -> torch.Tensor:
    ensure_nchw_video(video)
    x, y, w, h = bbox
    return video[:, :, y : y + h, x : x + w]


def splice_video(
    base_video: torch.Tensor,
    patch_video: torch.Tensor,
    left_top: tuple[int, int],
) -> torch.Tensor:
    ensure_nchw_video(base_video)
    ensure_nchw_video(patch_video)
    output = base_video.clone()
    x, y = left_top
    output[:, :, y : y + patch_video.shape[-2], x : x + patch_video.shape[-1]] = patch_video
    return output


def splice_video_inplace(
    base_video: torch.Tensor,
    patch_video: torch.Tensor,
    left_top: tuple[int, int],
) -> torch.Tensor:
    ensure_nchw_video(base_video)
    ensure_nchw_video(patch_video)
    x, y = left_top
    base_video[:, :, y : y + patch_video.shape[-2], x : x + patch_video.shape[-1]] = patch_video
    return base_video


def align_video(
    video: torch.Tensor,
    align_w: int | None = None,
    align_h: int | None = None,
    fill: float = 0.0,
    padding_mode: str = "replicate",
) -> torch.Tensor:
    ensure_nchw_video(video)
    width_pad = 0
    height_pad = 0
    if align_w is not None and align_w > 0:
        width_new = math.ceil(video.shape[-1] / align_w) * align_w
        width_pad = width_new - video.shape[-1]
    if align_h is not None and align_h > 0:
        height_new = math.ceil(video.shape[-2] / align_h) * align_h
        height_pad = height_new - video.shape[-2]
    if width_pad <= 0 and height_pad <= 0:
        return video

    pad = (0, width_pad, 0, height_pad)
    if padding_mode == "constant":
        return F.pad(video, pad, mode=padding_mode, value=fill)
    return F.pad(video, pad, mode=padding_mode)


def repeat_video_frames(
    video: torch.Tensor,
    num_frames: int,
    recycle: bool = False,
    check_num: bool = True,
) -> torch.Tensor:
    ensure_nchw_video(video)
    current_frames = video.shape[0]
    if check_num and current_frames > num_frames:
        raise ValueError(f"origin num_frame={current_frames} > {num_frames}")
    padding = num_frames - current_frames
    if padding < 0:
        return video[:num_frames]
    if padding == 0:
        return video

    if current_frames < 2 or not recycle:
        tail = video[-1:, ...].repeat(padding, 1, 1, 1)
        return torch.cat([video, tail], dim=0)

    frames = [video]
    index = current_frames
    step = -1
    for _ in range(padding):
        index += step
        frames.append(video.narrow(dim=0, start=index, length=1))
        if index == 0:
            step = 1
        elif index == current_frames - 1:
            step = -1
    return torch.cat(frames, dim=0)


def ensure_bcfhw_video(video: torch.Tensor, channels: int | None = None) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError(f"Expected 5D video tensor [B,C,F,H,W], got {tuple(video.shape)}")
    if channels is not None and video.shape[1] != channels:
        raise ValueError(
            f"Expected video channel count {channels}, got {video.shape[1]}"
        )
    return video


def read_video_metadata(video_path: str) -> dict[str, object]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path does not exist: {video_path}")
    probe = ffmpeg.probe(video_path)
    video_stream = next(
        (stream for stream in probe["streams"] if stream.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise ValueError(f"No video stream found in {video_path}")

    width = int(video_stream["width"])
    height = int(video_stream["height"])
    avg_frame_rate = video_stream.get("avg_frame_rate", "0/1")
    fps_fraction = _safe_fraction(avg_frame_rate)
    fps = float(fps_fraction) if fps_fraction.numerator > 0 else 0.0
    num_frames_raw = video_stream.get("nb_frames")
    num_frames = int(num_frames_raw) if num_frames_raw not in (None, "N/A") else None
    codec_name = video_stream.get("codec_name")
    pix_fmt = video_stream.get("pix_fmt")
    color_space = video_stream.get("color_space")
    color_transfer = video_stream.get("color_transfer")
    color_primaries = video_stream.get("color_primaries")
    color_range = video_stream.get("color_range")
    field_order = video_stream.get("field_order")
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "fps_fraction": str(fps_fraction),
        "num_frames": num_frames,
        "codec_name": codec_name,
        "bit_rate": video_stream.get("bit_rate"),
        "pix_fmt": pix_fmt,
        "color_space": color_space,
        "color_transfer": color_transfer,
        "color_primaries": color_primaries,
        "color_range": color_range,
        "field_order": field_order,
    }


def read_video_tensor(video_path: str) -> tuple[torch.Tensor, dict[str, object]]:
    metadata = read_video_metadata(video_path)
    width = int(metadata["width"])
    height = int(metadata["height"])
    output, _ = (
        ffmpeg.input(video_path)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run(capture_stdout=True, capture_stderr=True)
    )
    frame_size = width * height * 3
    if len(output) % frame_size != 0:
        raise ValueError(
            f"Raw video payload size {len(output)} is not divisible by frame size {frame_size}"
        )
    num_frames = len(output) // frame_size
    array = np.frombuffer(output, np.uint8).reshape(num_frames, height, width, 3)
    tensor = torch.from_numpy(array.copy()).permute(0, 3, 1, 2).float() / 255.0
    metadata["num_frames"] = num_frames
    return tensor, metadata


def read_video_array(video_path: str) -> tuple[np.ndarray, dict[str, object]]:
    metadata = read_video_metadata(video_path)
    width = int(metadata["width"])
    height = int(metadata["height"])
    output, _ = (
        ffmpeg.input(video_path)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run(capture_stdout=True, capture_stderr=True)
    )
    frame_size = width * height * 3
    if len(output) % frame_size != 0:
        raise ValueError(
            f"Raw video payload size {len(output)} is not divisible by frame size {frame_size}"
        )
    num_frames = len(output) // frame_size
    array = np.frombuffer(output, np.uint8).reshape(num_frames, height, width, 3).copy()
    metadata["num_frames"] = num_frames
    return array, metadata


def binarize_mask_tensor(
    mask: torch.Tensor,
    threshold_ratio: float = 0.3,
) -> torch.Tensor:
    if mask.ndim != 4:
        raise ValueError(f"Expected 4D mask tensor [F,C,H,W], got {tuple(mask.shape)}")
    if mask.shape[1] > 1:
        mask = mask.max(dim=1, keepdim=True)[0]
    if mask.numel() == 0:
        return mask
    max_value = float(mask.max().item())
    if max_value <= 0.0:
        return torch.zeros_like(mask)
    threshold = max_value * float(threshold_ratio)
    zeros = torch.zeros((), dtype=mask.dtype, device=mask.device)
    ones = torch.ones((), dtype=mask.dtype, device=mask.device)
    return torch.where(mask <= threshold, zeros, ones)


def binarize_mask_array(
    mask: np.ndarray,
    threshold_ratio: float = 0.3,
) -> np.ndarray:
    if mask.ndim == 4 and mask.shape[-1] > 1:
        mask = mask.max(axis=-1)
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask array [F,H,W], got {tuple(mask.shape)}")
    if mask.size == 0:
        return mask
    max_value = float(mask.max())
    if max_value <= 0.0:
        return np.zeros_like(mask)
    threshold = max_value * float(threshold_ratio)
    # Python integer branches create an int64 temporary (2.4 GB for 145
    # 1080p frames). Keep the binary values in bytes before restoring the
    # caller's dtype, including the original integer conversion semantics.
    return np.where(mask <= threshold, np.uint8(0), np.uint8(255)).astype(
        mask.dtype, copy=False
    )


def read_mask_rgb_array(
    video_path: str,
    threshold_ratio: float = 0.3,
) -> tuple[np.ndarray, dict[str, object]]:
    """Mask reader that keeps the source's RGB decode instead of a luma collapse.

    This reader thresholds each RGB channel of the mask stream
    independently; decoding through ``-pix_fmt gray`` shifts a small fraction of
    pixels by one code value and flips their binarisation decision.  Returns
    ``[F, H, W]`` built from the first channel (the mask stream's channels are
    equal by construction).
    """
    metadata = read_video_metadata(video_path)
    width = int(metadata["width"])
    height = int(metadata["height"])
    output, _ = (
        ffmpeg.input(video_path)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run(capture_stdout=True, capture_stderr=True)
    )
    frame_size = width * height * 3
    if len(output) % frame_size != 0:
        raise ValueError(
            f"Raw mask payload size {len(output)} is not divisible by RGB frame size"
        )
    num_frames = len(output) // frame_size
    array = (
        np.frombuffer(output, np.uint8)
        .reshape(num_frames, height, width, 3)[..., 0]
        .copy()
    )
    metadata["num_frames"] = num_frames
    return binarize_mask_array(array, threshold_ratio=threshold_ratio), metadata


def read_mask_tensor(video_path: str) -> tuple[torch.Tensor, dict[str, object]]:
    tensor, metadata = read_video_tensor(video_path)
    return binarize_mask_tensor(tensor), metadata


def read_mask_array(
    video_path: str,
    threshold_ratio: float = 0.3,
) -> tuple[np.ndarray, dict[str, object]]:
    metadata = read_video_metadata(video_path)
    width = int(metadata["width"])
    height = int(metadata["height"])
    output, _ = (
        ffmpeg.input(video_path)
        .output("pipe:", format="rawvideo", pix_fmt="gray")
        .run(capture_stdout=True, capture_stderr=True)
    )
    frame_size = width * height
    if len(output) % frame_size != 0:
        raise ValueError(
            f"Raw mask payload size {len(output)} is not divisible by frame size {frame_size}"
        )
    num_frames = len(output) // frame_size
    array = np.frombuffer(output, np.uint8).reshape(num_frames, height, width).copy()
    metadata["num_frames"] = num_frames
    return binarize_mask_array(array, threshold_ratio=threshold_ratio), metadata


def frames_uint8_to_tensor(frames: np.ndarray) -> torch.Tensor:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"Expected uint8 video frames with shape [F,H,W,3], got {tuple(frames.shape)}"
        )
    return (
        # Own one float32 allocation: no intermediate uint8 copy or second
        # float32 normalization buffer. NumPy also handles negative strides
        # and read-only inputs without borrowing their mutable storage.
        torch.from_numpy(np.array(frames, dtype=np.float32, copy=True, order="C"))
        .permute(0, 3, 1, 2)
        .div_(255.0)
    )


def mask_uint8_to_tensor(mask: np.ndarray) -> torch.Tensor:
    if mask.ndim != 3:
        raise ValueError(
            f"Expected uint8 mask frames with shape [F,H,W], got {tuple(mask.shape)}"
        )
    return torch.from_numpy(
        np.array(mask, dtype=np.float32, copy=True, order="C")
    )[:, None, :, :].div_(255.0)


def frames_tensor_to_uint8(frames: torch.Tensor) -> np.ndarray:
    ensure_nchw_video(frames, channels=3)
    return (
        frames.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(dtype=torch.uint8, device="cpu")
        .permute(0, 2, 3, 1)
        .contiguous()
        .numpy()
    )


def _validate_video_io_thread_count(thread_count: int | str) -> int | str:
    if isinstance(thread_count, str):
        # ffmpeg's own default; used when a caller must match an encoder
        # invocation that passes no ``-threads`` at all.
        if thread_count == "auto":
            return thread_count
        raise ValueError("thread_count must be a positive integer or 'auto'")
    if isinstance(thread_count, bool) or not isinstance(thread_count, int):
        raise TypeError("thread_count must be a positive integer or 'auto'")
    if thread_count < 1:
        raise ValueError("thread_count must be a positive integer")
    return thread_count


class SequentialVideoReader:
    def __init__(
        self,
        video_path: str,
        *,
        width: int,
        height: int,
        pix_fmt: str = "rgb24",
        squeeze_single_channel: bool = False,
        thread_count: int = 4,
    ) -> None:
        self.video_path = os.path.abspath(video_path)
        self.width = int(width)
        self.height = int(height)
        self.pix_fmt = pix_fmt
        self.channels = 3 if pix_fmt == "rgb24" else 1
        self.squeeze_single_channel = bool(squeeze_single_channel and self.channels == 1)
        self.thread_count = _validate_video_io_thread_count(thread_count)
        self._closed = False
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-threads",
            str(self.thread_count),
            "-i",
            self.video_path,
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            self.pix_fmt,
            "pipe:1",
        ]
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def read_frames(self, num_frames: int) -> np.ndarray:
        if num_frames <= 0:
            if self.squeeze_single_channel:
                shape = (0, self.height, self.width)
            else:
                shape = (0, self.height, self.width, self.channels)
            return np.empty(shape, dtype=np.uint8)
        if self.process.stdout is None:
            raise RuntimeError("video decoder stdout is unavailable")
        frame_size = self.width * self.height * self.channels
        expected = frame_size * int(num_frames)
        payload = self.process.stdout.read(expected)
        if len(payload) != expected:
            stderr = (
                self.process.stderr.read().decode("utf-8", errors="ignore")
                if self.process.stderr is not None
                else ""
            )
            raise RuntimeError(
                f"Decoded payload mismatch for {self.video_path}: expected {expected} bytes, got {len(payload)}. ffmpeg stderr: {stderr}"
            )
        array = np.frombuffer(payload, np.uint8).reshape(
            int(num_frames),
            self.height,
            self.width,
            self.channels,
        )
        array = array.copy()
        if self.squeeze_single_channel:
            return array[..., 0]
        return array

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.process.stdout is not None and not self.process.stdout.closed:
            self.process.stdout.close()
        if self.process.stderr is not None and not self.process.stderr.closed:
            self.process.stderr.close()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


class SequentialVideoWriter:
    def __init__(
        self,
        output_path: str,
        *,
        width: int | None = None,
        height: int | None = None,
        fps: float | str | None = None,
        codec_name: str | None = None,
        video_bitrate: str | None = None,
        color_space: str | None = None,
        color_transfer: str | None = None,
        color_primaries: str | None = None,
        color_range: str | None = None,
        thread_count: int = 4,
        encoding_profile: VideoEncodingProfile | None = None,
        async_queue_depth: int = 0,
    ) -> None:
        self.output_path = os.path.abspath(output_path)
        if encoding_profile is not None:
            self.width = encoding_profile.width
            self.height = encoding_profile.height
            self.fps = encoding_profile.frame_rate
            self.codec_name = encoding_profile.encoder
            self.video_bitrate = encoding_profile.video_bitrate
        else:
            if width is None or height is None or fps is None:
                raise ValueError("width, height and fps are required without encoding_profile")
            self.width = int(width)
            self.height = int(height)
            self.fps = str(max(float(fps), 1.0))
            self.codec_name = codec_name or "libx264"
            self.video_bitrate = video_bitrate or "10M"
        self.thread_count = _validate_video_io_thread_count(thread_count)
        if (
            isinstance(async_queue_depth, bool)
            or not isinstance(async_queue_depth, int)
            or async_queue_depth < 0
        ):
            raise ValueError("async_queue_depth must be a non-negative integer")
        self.async_queue_depth = async_queue_depth
        self._closed = False
        self._async_error: BaseException | None = None
        self._async_sentinel = object()
        self._async_queue: queue.Queue[np.ndarray | object] | None = None
        self._async_thread: threading.Thread | None = None
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            self.fps,
            "-i",
            "pipe:0",
            "-an",
            "-threads",
            str(self.thread_count),
        ]
        if encoding_profile is not None:
            cmd.extend(encoding_profile.ffmpeg_output_args())
        else:
            cmd += [
                "-vcodec",
                self.codec_name,
                "-b:v",
                self.video_bitrate,
                "-pix_fmt",
                "yuv420p",
            ]
            if color_space:
                cmd += ["-colorspace", color_space]
            if color_transfer:
                cmd += ["-color_trc", color_transfer]
            if color_primaries:
                cmd += ["-color_primaries", color_primaries]
            if color_range:
                cmd += ["-color_range", color_range]
        cmd.append(self.output_path)
        self._stderr_file = tempfile.TemporaryFile(mode="w+b")
        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr_file,
            )
        except BaseException:
            self._stderr_file.close()
            raise
        if self.async_queue_depth > 0:
            self._async_queue = queue.Queue(maxsize=self.async_queue_depth)
            self._async_thread = threading.Thread(
                target=self._run_async_writer,
                name="nndeploy-video-writer",
                daemon=True,
            )
            self._async_thread.start()

    def _run_async_writer(self) -> None:
        assert self._async_queue is not None
        while True:
            item = self._async_queue.get()
            try:
                if item is self._async_sentinel:
                    return
                if self._async_error is None:
                    assert isinstance(item, np.ndarray)
                    self._write_frames_sync(item)
            except BaseException as exc:
                self._async_error = exc
            finally:
                self._async_queue.task_done()

    def _raise_async_error(self) -> None:
        if self._async_error is not None:
            raise RuntimeError(
                f"asynchronous ffmpeg writer failed for {self.output_path}: "
                f"{self._async_error}"
            ) from self._async_error

    def _enqueue_async(self, item: np.ndarray | object) -> None:
        assert self._async_queue is not None
        while True:
            self._raise_async_error()
            try:
                self._async_queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _wait_and_collect_stderr(
        self,
    ) -> tuple[int, str, BrokenPipeError | None]:
        self._closed = True
        pipe_error: BrokenPipeError | None = None
        if self.process.stdin is not None and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except BrokenPipeError as exc:
                pipe_error = exc
        try:
            returncode = self.process.wait()
            self._stderr_file.seek(0)
            stderr = self._stderr_file.read().decode("utf-8", errors="ignore")
        finally:
            self._stderr_file.close()
        return returncode, stderr, pipe_error

    def _validate_frames(self, frames: np.ndarray) -> None:
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(
                f"Expected uint8 video frames with shape [F,H,W,3], got {tuple(frames.shape)}"
            )
        if frames.shape[1] != self.height or frames.shape[2] != self.width:
            raise ValueError(
                f"Frame size mismatch: expected {self.height}x{self.width}, got {frames.shape[1]}x{frames.shape[2]}"
            )

    def _write_frames_sync(self, frames: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("video writer stdin is unavailable")
        try:
            contiguous_frames = np.ascontiguousarray(frames)
            self.process.stdin.write(memoryview(contiguous_frames).cast("B"))
        except BrokenPipeError as exc:
            returncode, stderr, _ = self._wait_and_collect_stderr()
            raise RuntimeError(
                f"ffmpeg writer broken pipe for {self.output_path} "
                f"(exit code {returncode}): {stderr}"
            ) from exc

    def write_frames(self, frames: np.ndarray) -> None:
        self._validate_frames(frames)
        self._raise_async_error()
        self._write_frames_sync(frames)

    def write_frames_owned(self, frames: np.ndarray) -> None:
        """Write frames after transferring ownership to the writer.

        With an asynchronous queue configured, the caller must not read or
        mutate ``frames`` after this method returns.
        """

        self._validate_frames(frames)
        if self._async_queue is None:
            self._write_frames_sync(frames)
            return
        self._enqueue_async(frames)

    def close(self) -> None:
        # A broken pipe may already have closed the subprocess from the writer
        # thread. The queue consumer still needs its sentinel and join.
        if self._async_thread is not None and self._async_thread.is_alive():
            assert self._async_queue is not None
            while True:
                try:
                    self._async_queue.put(self._async_sentinel, timeout=0.1)
                    break
                except queue.Full:
                    continue
            assert self._async_thread is not None
            self._async_thread.join()
        if self._closed:
            self._raise_async_error()
            return
        if self._async_error is not None:
            self._wait_and_collect_stderr()
            self._raise_async_error()
        returncode, stderr, pipe_error = self._wait_and_collect_stderr()
        if returncode != 0 or pipe_error is not None:
            raise RuntimeError(
                f"ffmpeg writer failed for {self.output_path} "
                f"(exit code {returncode}): {stderr}"
            )


class ArrayFrameCache:
    def __init__(self, start_index: int, shape_tail: tuple[int, ...], dtype: np.dtype) -> None:
        self.start_index = int(start_index)
        self.shape_tail = tuple(shape_tail)
        self.dtype = np.dtype(dtype)
        self.data = np.empty((0, *self.shape_tail), dtype=self.dtype)

    @property
    def end_index(self) -> int:
        return self.start_index + self.data.shape[0]

    @property
    def num_frames(self) -> int:
        return int(self.data.shape[0])

    def append(self, frames: np.ndarray) -> None:
        if frames.ndim != len(self.shape_tail) + 1:
            raise ValueError(
                f"Expected frames ndim {len(self.shape_tail) + 1}, got {frames.ndim}"
            )
        if tuple(frames.shape[1:]) != self.shape_tail:
            raise ValueError(
                f"Expected frame shape tail {self.shape_tail}, got {tuple(frames.shape[1:])}"
            )
        if frames.dtype != self.dtype:
            frames = frames.astype(self.dtype, copy=False)
        if self.num_frames == 0:
            self.data = frames.copy()
        else:
            self.data = np.concatenate([self.data, frames], axis=0)

    def append_owned(self, frames: np.ndarray) -> int:
        """Append a buffer whose ownership is transferred to this cache.

        The caller must not mutate ``frames`` after this call. The return value
        is the number of bytes copied internally by the cache.
        """

        if frames.ndim != len(self.shape_tail) + 1:
            raise ValueError(
                f"Expected frames ndim {len(self.shape_tail) + 1}, got {frames.ndim}"
            )
        if tuple(frames.shape[1:]) != self.shape_tail:
            raise ValueError(
                f"Expected frame shape tail {self.shape_tail}, got {tuple(frames.shape[1:])}"
            )
        copied_bytes = 0
        if frames.dtype != self.dtype:
            frames = frames.astype(self.dtype, copy=False)
            copied_bytes += int(frames.nbytes)
        if frames.shape[0] == 0:
            return copied_bytes
        if self.num_frames == 0:
            self.data = frames
        else:
            self.data = np.concatenate([self.data, frames], axis=0)
            copied_bytes += int(self.data.nbytes)
        return copied_bytes

    def slice(self, start_frame: int, end_frame: int) -> np.ndarray:
        if start_frame < self.start_index or end_frame > self.end_index or end_frame < start_frame:
            raise ValueError(
                f"Invalid cache slice [{start_frame}, {end_frame}) for cache range [{self.start_index}, {self.end_index})"
            )
        begin = start_frame - self.start_index
        end = end_frame - self.start_index
        return self.data[begin:end].copy()

    def overwrite(self, start_frame: int, frames: np.ndarray) -> None:
        end_frame = start_frame + int(frames.shape[0])
        if start_frame < self.start_index or end_frame > self.end_index:
            raise ValueError(
                f"Invalid cache overwrite [{start_frame}, {end_frame}) for cache range [{self.start_index}, {self.end_index})"
            )
        begin = start_frame - self.start_index
        end = end_frame - self.start_index
        if frames.dtype != self.dtype:
            frames = frames.astype(self.dtype, copy=False)
        self.data[begin:end] = frames

    def fill_region(
        self,
        start_frame: int,
        end_frame: int,
        bbox: tuple[int, int, int, int],
        value: int | float,
    ) -> None:
        if start_frame < self.start_index or end_frame > self.end_index:
            raise ValueError(
                f"Invalid cache fill [{start_frame}, {end_frame}) for cache range "
                f"[{self.start_index}, {self.end_index})"
            )
        x, y, width, height = (int(item) for item in bbox)
        begin = start_frame - self.start_index
        end = end_frame - self.start_index
        self.data[begin:end, y : y + height, x : x + width] = value

    def pop_before(self, end_frame: int) -> np.ndarray:
        if end_frame <= self.start_index:
            return np.empty((0, *self.shape_tail), dtype=self.dtype)
        if end_frame > self.end_index:
            raise ValueError(
                f"Invalid cache pop end {end_frame} for cache range [{self.start_index}, {self.end_index})"
            )
        length = end_frame - self.start_index
        popped = self.data[:length].copy()
        self.data = self.data[length:].copy()
        self.start_index = end_frame
        return popped


class ChunkedFrameCache:
    def __init__(self, start_index: int, shape_tail: tuple[int, ...], dtype: np.dtype) -> None:
        self.start_index = int(start_index)
        self.shape_tail = tuple(shape_tail)
        self.dtype = np.dtype(dtype)
        self._chunks: deque[np.ndarray] = deque()

    @property
    def end_index(self) -> int:
        return self.start_index + self.num_frames

    @property
    def num_frames(self) -> int:
        return int(sum(chunk.shape[0] for chunk in self._chunks))

    def append(self, frames: np.ndarray) -> None:
        if frames.ndim != len(self.shape_tail) + 1:
            raise ValueError(
                f"Expected frames ndim {len(self.shape_tail) + 1}, got {frames.ndim}"
            )
        if tuple(frames.shape[1:]) != self.shape_tail:
            raise ValueError(
                f"Expected frame shape tail {self.shape_tail}, got {tuple(frames.shape[1:])}"
            )
        if frames.shape[0] == 0:
            return
        if frames.dtype != self.dtype:
            frames = frames.astype(self.dtype, copy=False)
        self._chunks.append(frames.copy())

    def append_owned(self, frames: np.ndarray) -> int:
        """Append a buffer by ownership transfer and report internal copies."""

        if frames.ndim != len(self.shape_tail) + 1:
            raise ValueError(
                f"Expected frames ndim {len(self.shape_tail) + 1}, got {frames.ndim}"
            )
        if tuple(frames.shape[1:]) != self.shape_tail:
            raise ValueError(
                f"Expected frame shape tail {self.shape_tail}, got {tuple(frames.shape[1:])}"
            )
        if frames.shape[0] == 0:
            return 0
        copied_bytes = 0
        if frames.dtype != self.dtype:
            frames = frames.astype(self.dtype, copy=False)
            copied_bytes = int(frames.nbytes)
        self._chunks.append(frames)
        return copied_bytes

    def slice(self, start_frame: int, end_frame: int) -> np.ndarray:
        if start_frame < self.start_index or end_frame > self.end_index or end_frame < start_frame:
            raise ValueError(
                f"Invalid cache slice [{start_frame}, {end_frame}) for cache range [{self.start_index}, {self.end_index})"
            )
        length = end_frame - start_frame
        if length == 0:
            return np.empty((0, *self.shape_tail), dtype=self.dtype)

        result = np.empty((length, *self.shape_tail), dtype=self.dtype)
        cursor = self.start_index
        for chunk in self._chunks:
            chunk_end = cursor + chunk.shape[0]
            overlap_start = max(start_frame, cursor)
            overlap_end = min(end_frame, chunk_end)
            if overlap_start < overlap_end:
                src_begin = overlap_start - cursor
                src_end = overlap_end - cursor
                dst_begin = overlap_start - start_frame
                dst_end = overlap_end - start_frame
                result[dst_begin:dst_end] = chunk[src_begin:src_end]
            cursor = chunk_end
            if cursor >= end_frame:
                break
        return result

    def overwrite(self, start_frame: int, frames: np.ndarray) -> None:
        end_frame = start_frame + int(frames.shape[0])
        if start_frame < self.start_index or end_frame > self.end_index:
            raise ValueError(
                f"Invalid cache overwrite [{start_frame}, {end_frame}) for cache range [{self.start_index}, {self.end_index})"
            )
        if frames.dtype != self.dtype:
            frames = frames.astype(self.dtype, copy=False)

        cursor = self.start_index
        for chunk in self._chunks:
            chunk_end = cursor + chunk.shape[0]
            overlap_start = max(start_frame, cursor)
            overlap_end = min(end_frame, chunk_end)
            if overlap_start < overlap_end:
                src_begin = overlap_start - start_frame
                src_end = overlap_end - start_frame
                dst_begin = overlap_start - cursor
                dst_end = overlap_end - cursor
                chunk[dst_begin:dst_end] = frames[src_begin:src_end]
            cursor = chunk_end
            if cursor >= end_frame:
                break

    def fill_region(
        self,
        start_frame: int,
        end_frame: int,
        bbox: tuple[int, int, int, int],
        value: int | float,
    ) -> None:
        if start_frame < self.start_index or end_frame > self.end_index:
            raise ValueError(
                f"Invalid cache fill [{start_frame}, {end_frame}) for cache range "
                f"[{self.start_index}, {self.end_index})"
            )
        x, y, width, height = (int(item) for item in bbox)
        cursor = self.start_index
        for chunk in self._chunks:
            chunk_end = cursor + int(chunk.shape[0])
            overlap_start = max(start_frame, cursor)
            overlap_end = min(end_frame, chunk_end)
            if overlap_start < overlap_end:
                chunk[
                    overlap_start - cursor : overlap_end - cursor,
                    y : y + height,
                    x : x + width,
                ] = value
            cursor = chunk_end
            if cursor >= end_frame:
                break

    def pop_before(self, end_frame: int) -> np.ndarray:
        if end_frame <= self.start_index:
            return np.empty((0, *self.shape_tail), dtype=self.dtype)
        if end_frame > self.end_index:
            raise ValueError(
                f"Invalid cache pop end {end_frame} for cache range [{self.start_index}, {self.end_index})"
            )

        popped_chunks: list[np.ndarray] = []
        while self._chunks and self.start_index < end_frame:
            chunk = self._chunks[0]
            chunk_end = self.start_index + chunk.shape[0]
            if chunk_end <= end_frame:
                popped_chunks.append(self._chunks.popleft())
                self.start_index = chunk_end
                continue

            take = end_frame - self.start_index
            if take > 0:
                popped_chunks.append(chunk[:take].copy())
                self._chunks[0] = chunk[take:].copy()
                self.start_index = end_frame
            break

        if not popped_chunks:
            return np.empty((0, *self.shape_tail), dtype=self.dtype)
        if len(popped_chunks) == 1:
            return popped_chunks[0].copy()
        return np.concatenate(popped_chunks, axis=0)


class TensorFrameCache:
    """Chunked CPU tensor cache whose leading dimension is the frame axis.

    Streaming video frames use FCHW BF16 here.  Decoder uint8 input is
    normalized exactly once by :meth:`append_uint8`; all other cache methods
    preserve the configured floating dtype.
    """

    def __init__(
        self,
        start_index: int,
        shape_tail: tuple[int, ...],
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if not dtype.is_floating_point:
            raise ValueError("TensorFrameCache requires a floating dtype")
        self.start_index = int(start_index)
        self.shape_tail = tuple(shape_tail)
        self.dtype = dtype
        self._chunks: deque[torch.Tensor] = deque()

    @property
    def end_index(self) -> int:
        return self.start_index + self.num_frames

    @property
    def num_frames(self) -> int:
        return int(sum(chunk.shape[0] for chunk in self._chunks))

    def _validate(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.device.type != "cpu":
            frames = frames.to(device="cpu")
        if frames.ndim != len(self.shape_tail) + 1:
            raise ValueError(
                f"Expected frames ndim {len(self.shape_tail) + 1}, got {frames.ndim}"
            )
        if tuple(frames.shape[1:]) != self.shape_tail:
            raise ValueError(
                f"Expected frame shape tail {self.shape_tail}, got {tuple(frames.shape[1:])}"
            )
        if frames.dtype != self.dtype:
            frames = frames.to(dtype=self.dtype)
        return frames

    def append_uint8(self, frames: np.ndarray) -> None:
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(
                f"Expected uint8 video frames [F,H,W,3], got {tuple(frames.shape)}"
            )
        normalized = (
            torch.from_numpy(frames)
            .permute(0, 3, 1, 2)
            .to(dtype=self.dtype)
            .div_(255.0)
            .contiguous()
        )
        self.append_owned(normalized)

    def append(self, frames: torch.Tensor) -> None:
        frames = self._validate(frames)
        if frames.shape[0] > 0:
            self._chunks.append(frames.clone())

    def append_owned(self, frames: torch.Tensor) -> int:
        original = frames
        frames = self._validate(frames)
        if frames.shape[0] == 0:
            return 0
        self._chunks.append(frames)
        return 0 if frames is original else int(frames.numel() * frames.element_size())

    def _slice_parts(
        self,
        start_frame: int,
        end_frame: int,
        bbox: tuple[int, int, int, int] | None = None,
    ) -> list[torch.Tensor]:
        if (
            start_frame < self.start_index
            or end_frame > self.end_index
            or end_frame < start_frame
        ):
            raise ValueError(
                f"Invalid cache slice [{start_frame}, {end_frame}) for cache range "
                f"[{self.start_index}, {self.end_index})"
            )
        parts: list[torch.Tensor] = []
        cursor = self.start_index
        for chunk in self._chunks:
            chunk_end = cursor + int(chunk.shape[0])
            overlap_start = max(start_frame, cursor)
            overlap_end = min(end_frame, chunk_end)
            if overlap_start < overlap_end:
                part = chunk[overlap_start - cursor : overlap_end - cursor]
                if bbox is not None:
                    x, y, width, height = (int(value) for value in bbox)
                    part = part[:, :, y : y + height, x : x + width]
                parts.append(part)
            cursor = chunk_end
            if cursor >= end_frame:
                break
        return parts

    def slice(self, start_frame: int, end_frame: int) -> torch.Tensor:
        parts = self._slice_parts(start_frame, end_frame)
        if not parts:
            return torch.empty((0, *self.shape_tail), dtype=self.dtype)
        if len(parts) == 1:
            return parts[0].clone()
        return torch.cat(parts, dim=0)

    def slice_crop(
        self,
        start_frame: int,
        end_frame: int,
        bbox: tuple[int, int, int, int],
    ) -> torch.Tensor:
        x, y, width, height = (int(value) for value in bbox)
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError(f"Invalid crop bbox: {bbox}")
        if x + width > self.shape_tail[-1] or y + height > self.shape_tail[-2]:
            raise ValueError(f"Crop bbox {bbox} exceeds cache frame shape {self.shape_tail}")
        parts = self._slice_parts(start_frame, end_frame, bbox)
        if not parts:
            return torch.empty(
                (0, self.shape_tail[0], height, width), dtype=self.dtype
            )
        if len(parts) == 1:
            return parts[0].clone()
        return torch.cat(parts, dim=0)

    def overwrite(self, start_frame: int, frames: torch.Tensor) -> None:
        frames = self._validate(frames)
        end_frame = start_frame + int(frames.shape[0])
        if start_frame < self.start_index or end_frame > self.end_index:
            raise ValueError(
                f"Invalid cache overwrite [{start_frame}, {end_frame}) for cache range "
                f"[{self.start_index}, {self.end_index})"
            )
        cursor = self.start_index
        for chunk in self._chunks:
            chunk_end = cursor + int(chunk.shape[0])
            overlap_start = max(start_frame, cursor)
            overlap_end = min(end_frame, chunk_end)
            if overlap_start < overlap_end:
                chunk[overlap_start - cursor : overlap_end - cursor].copy_(
                    frames[overlap_start - start_frame : overlap_end - start_frame]
                )
            cursor = chunk_end
            if cursor >= end_frame:
                break

    def pop_before(self, end_frame: int) -> torch.Tensor:
        if end_frame <= self.start_index:
            return torch.empty((0, *self.shape_tail), dtype=self.dtype)
        if end_frame > self.end_index:
            raise ValueError(
                f"Invalid cache pop end {end_frame} for cache range "
                f"[{self.start_index}, {self.end_index})"
            )
        popped: list[torch.Tensor] = []
        while self._chunks and self.start_index < end_frame:
            chunk = self._chunks[0]
            chunk_end = self.start_index + int(chunk.shape[0])
            if chunk_end <= end_frame:
                popped.append(self._chunks.popleft())
                self.start_index = chunk_end
                continue
            take = end_frame - self.start_index
            popped.append(chunk[:take].clone())
            self._chunks[0] = chunk[take:].clone()
            self.start_index = end_frame
            break
        if not popped:
            return torch.empty((0, *self.shape_tail), dtype=self.dtype)
        if len(popped) == 1:
            return popped[0]
        return torch.cat(popped, dim=0)


def _write_rawvideo_ffmpeg(
    video: torch.Tensor,
    output_path: str,
    fps: float,
    codec_name: str | None = None,
    video_bitrate: str = "10M",
    *,
    color_space: str | None = None,
    color_transfer: str | None = None,
    color_primaries: str | None = None,
    color_range: str | None = None,
) -> None:
    ensure_nchw_video(video, channels=3)
    frames = (
        video.clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(dtype=torch.uint8, device="cpu")
        .permute(0, 2, 3, 1)
        .contiguous()
        .numpy()
    )
    height = int(frames.shape[1])
    width = int(frames.shape[2])
    codec = codec_name or "libx264"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(max(fps, 1.0)),
        "-i",
        "pipe:0",
        "-an",
        "-vcodec",
        codec,
        "-b:v",
        video_bitrate,
        "-pix_fmt",
        "yuv420p",
    ]
    if color_space:
        cmd += ["-colorspace", color_space]
    if color_transfer:
        cmd += ["-color_trc", color_transfer]
    if color_primaries:
        cmd += ["-color_primaries", color_primaries]
    if color_range:
        cmd += ["-color_range", color_range]
    cmd.append(output_path)
    process = subprocess.run(
        cmd,
        input=frames.tobytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        error = process.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg write failed for {output_path}: {error}")


def save_video_tensor(
    video: torch.Tensor,
    output_path: str,
    fps: float,
    codec_name: str | None = None,
    video_bitrate: str = "10M",
    *,
    color_space: str | None = None,
    color_transfer: str | None = None,
    color_primaries: str | None = None,
    color_range: str | None = None,
) -> str:
    ensure_nchw_video(video, channels=3)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    _write_rawvideo_ffmpeg(
        video, output_path,
        fps=fps,
        codec_name=codec_name,
        video_bitrate=video_bitrate,
        color_space=color_space,
        color_transfer=color_transfer,
        color_primaries=color_primaries,
        color_range=color_range,
    )
    return output_path


def copy_audio_to_video(
    audio_source_video: str,
    video_path: str,
    output_path: str | None = None,
) -> str:
    if not os.path.exists(audio_source_video):
        raise FileNotFoundError(f"Audio source video does not exist: {audio_source_video}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video path does not exist: {video_path}")

    target_path = output_path or video_path
    abs_video_path = os.path.abspath(video_path)
    abs_target_path = os.path.abspath(target_path)
    replace_in_place = abs_video_path == abs_target_path
    temp_output_path = (
        abs_target_path + ".tmp_audio.mp4" if replace_in_place else abs_target_path
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        audio_source_video,
        "-i",
        video_path,
        "-map",
        "1:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        temp_output_path,
    ]
    process = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        error = process.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"ffmpeg audio mux failed for video={video_path} audio_source={audio_source_video}: {error}"
        )

    if replace_in_place:
        os.replace(temp_output_path, abs_target_path)
    return abs_target_path


class WindowedVideoStore:
    """Window-addressable runtime store for long-form video processing."""

    def __init__(
        self,
        *,
        video_path: str,
        mask_path: str,
        num_frames: int,
        height: int,
        width: int,
        fps: float,
        codec_name: str | None = None,
        workdir: str | None = None,
    ) -> None:
        self.video_path = os.path.abspath(video_path)
        self.mask_path = os.path.abspath(mask_path)
        self.num_frames = int(num_frames)
        self.height = int(height)
        self.width = int(width)
        self.fps = float(fps)
        self.codec_name = codec_name
        self._frame_shape = (self.height, self.width, 3)
        self._mask_shape = (self.height, self.width)
        workdir_root = select_runtime_workdir(workdir)
        self._workdir = Path(
            tempfile.mkdtemp(prefix="nndeploy_frame_store_", dir=workdir_root)
        )
        self._workdir.mkdir(parents=True, exist_ok=True)
        self._final_store_path = self._workdir / "final_video.float32.mmap"
        self._mask_store_path = self._workdir / "mask_cache.uint8.mmap"
        self._written_store_path = self._workdir / "final_written.uint8.mmap"
        self._source_store_path = self._workdir / "source_video.uint8.mmap"
        self._final_store = np.memmap(
            self._final_store_path,
            dtype=np.float32,
            mode="w+",
            shape=(self.num_frames, self.height, self.width, 3),
        )
        self._mask_store = np.memmap(
            self._mask_store_path,
            dtype=np.uint8,
            mode="w+",
            shape=(self.num_frames, self.height, self.width),
        )
        self._written_store = np.memmap(
            self._written_store_path,
            dtype=np.uint8,
            mode="w+",
            shape=(self.num_frames,),
        )
        self._source_store = np.memmap(
            self._source_store_path,
            dtype=np.uint8,
            mode="w+",
            shape=(self.num_frames, self.height, self.width, 3),
        )
        self._source_loaded_frames = 0
        self._mask_loaded_frames = 0
        self._source_stream = self._open_stream_decoder(
            source_path=self.video_path,
            pix_fmt="rgb24",
        )
        self._mask_stream = self._open_stream_decoder(
            source_path=self.mask_path,
            pix_fmt="gray",
        )
        self._writer_process: subprocess.Popen[bytes] | None = None
        self._writer_stderr_path = self._workdir / "writer.stderr.log"

    def _open_stream_decoder(self, *, source_path: str, pix_fmt: str) -> subprocess.Popen[bytes]:
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            source_path,
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix_fmt,
            "pipe:1",
        ]
        return subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _ensure_source_loaded(self, end_frame: int) -> None:
        target = min(max(int(end_frame), 0), self.num_frames)
        if target <= self._source_loaded_frames:
            return
        if self._source_stream.stdout is None:
            raise RuntimeError("video source decoder stdout is unavailable")
        channels = 3
        frame_size = self.width * self.height * channels
        start = self._source_loaded_frames
        frames_to_read = target - start
        expected = frame_size * frames_to_read
        payload = self._source_stream.stdout.read(expected)
        if len(payload) != expected:
            stderr = (
                self._source_stream.stderr.read().decode("utf-8", errors="ignore")
                if self._source_stream.stderr is not None
                else ""
            )
            raise RuntimeError(
                f"Decoded payload mismatch for {self.video_path}: expected {expected} bytes, got {len(payload)}. ffmpeg stderr: {stderr}"
            )
        chunk = np.frombuffer(payload, np.uint8).reshape(
            frames_to_read,
            self.height,
            self.width,
            channels,
        )
        self._source_store[start:target] = chunk
        self._source_store.flush()
        self._source_loaded_frames = target
        if self._source_loaded_frames >= self.num_frames and self._source_stream.poll() is None:
            self._source_stream.stdout.close()

    def _ensure_mask_loaded(self, end_frame: int) -> None:
        target = min(max(int(end_frame), 0), self.num_frames)
        if target <= self._mask_loaded_frames:
            return
        if self._mask_stream.stdout is None:
            raise RuntimeError("mask source decoder stdout is unavailable")
        channels = 1
        frame_size = self.width * self.height * channels
        start = self._mask_loaded_frames
        frames_to_read = target - start
        expected = frame_size * frames_to_read
        payload = self._mask_stream.stdout.read(expected)
        if len(payload) != expected:
            stderr = (
                self._mask_stream.stderr.read().decode("utf-8", errors="ignore")
                if self._mask_stream.stderr is not None
                else ""
            )
            raise RuntimeError(
                f"Decoded payload mismatch for {self.mask_path}: expected {expected} bytes, got {len(payload)}. ffmpeg stderr: {stderr}"
            )
        chunk = np.frombuffer(payload, np.uint8).reshape(
            frames_to_read,
            self.height,
            self.width,
            channels,
        )
        self._mask_store[start:target] = chunk[..., 0]
        self._mask_store.flush()
        self._mask_loaded_frames = target
        if self._mask_loaded_frames >= self.num_frames and self._mask_stream.poll() is None:
            self._mask_stream.stdout.close()

    def read_video_window(self, start_frame: int, end_frame: int) -> torch.Tensor:
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        if start_frame < 0 or end_frame > self.num_frames or end_frame <= start_frame:
            raise ValueError(f"Invalid window [{start_frame}, {end_frame})")
        self._ensure_source_loaded(end_frame)
        base_frames = self._source_store[start_frame:end_frame].astype(np.float32)
        final_slice = self._final_store[start_frame:end_frame]
        final_written = self._written_store[start_frame:end_frame] > 0
        if final_written.any():
            base_frames[final_written] = np.clip(
                np.rint(final_slice[final_written] * 255.0),
                0,
                255,
            ).astype(np.float32)
        tensor = torch.from_numpy(base_frames).permute(0, 3, 1, 2).float() / 255.0
        return tensor

    def read_mask_window(self, start_frame: int, end_frame: int) -> torch.Tensor:
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        self._ensure_mask_loaded(end_frame)
        mask_slice = self._mask_store[start_frame:end_frame].astype(np.float32) / 255.0
        tensor = torch.from_numpy(mask_slice[:, None, :, :]).float()
        return tensor

    def clear_mask_window(
        self,
        start_frame: int,
        end_frame: int,
        bbox: tuple[int, int, int, int] | None = None,
    ) -> None:
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        if bbox is None:
            self._mask_store[start_frame:end_frame] = 0
        else:
            x, y, w, h = bbox
            self._mask_store[start_frame:end_frame, y : y + h, x : x + w] = 0
        self._mask_store.flush()

    def commit_video_window(
        self,
        start_frame: int,
        end_frame: int,
        frames: torch.Tensor,
    ) -> None:
        ensure_nchw_video(frames, channels=3)
        frame_count = end_frame - start_frame
        if frames.shape[0] != frame_count:
            raise ValueError(
                f"Commit frame count mismatch: expected {frame_count}, got {frames.shape[0]}"
            )
        array = (
            frames.detach()
            .clamp(0.0, 1.0)
            .to(dtype=torch.float32, device="cpu")
            .permute(0, 2, 3, 1)
            .contiguous()
            .numpy()
        )
        self._final_store[start_frame:end_frame] = array
        self._written_store[start_frame:end_frame] = 1
        self._final_store.flush()
        self._written_store.flush()

    def commit_patch_window(
        self,
        start_frame: int,
        end_frame: int,
        patch_frames: torch.Tensor,
        left_top: tuple[int, int],
        base_frames: torch.Tensor | None = None,
    ) -> None:
        ensure_nchw_video(patch_frames, channels=3)
        frame_count = end_frame - start_frame
        if patch_frames.shape[0] != frame_count:
            raise ValueError(
                f"Patch frame count mismatch: expected {frame_count}, got {patch_frames.shape[0]}"
            )
        if base_frames is None:
            base_frames = self.read_video_window(start_frame, end_frame)
        ensure_nchw_video(base_frames, channels=3)
        if base_frames.shape[0] != frame_count:
            raise ValueError(
                f"Base frame count mismatch: expected {frame_count}, got {base_frames.shape[0]}"
            )
        x, y = left_top
        merged = base_frames.clone()
        merged[:, :, y : y + patch_frames.shape[-2], x : x + patch_frames.shape[-1]] = patch_frames
        self.commit_video_window(start_frame, end_frame, merged)

    def finalize_to_video(self, output_path: str) -> str:
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._ensure_source_loaded(self.num_frames)
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(max(self.fps, 1.0)),
            "-i",
            "pipe:0",
            "-an",
            "-vcodec",
            self.codec_name or "libx264",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
        with tempfile.TemporaryFile(mode="w+b") as stderr_file:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
            )
            try:
                assert process.stdin is not None
                pipe_error: BrokenPipeError | None = None
                try:
                    for start in range(0, self.num_frames, 8):
                        end = min(self.num_frames, start + 8)
                        chunk = self._final_store[start:end]
                        written = self._written_store[start:end] > 0
                        if not written.all():
                            base = self._source_store[start:end].astype(np.float32) / 255.0
                            missing = ~written
                            chunk = chunk.copy()
                            chunk[missing] = base[missing]
                        bytes_chunk = (
                            np.clip(np.rint(chunk * 255.0), 0, 255)
                            .astype(np.uint8)
                            .tobytes()
                        )
                        process.stdin.write(bytes_chunk)
                except BrokenPipeError as exc:
                    pipe_error = exc
                finally:
                    if not process.stdin.closed:
                        try:
                            process.stdin.close()
                        except BrokenPipeError as exc:
                            pipe_error = pipe_error or exc

                returncode = process.wait()
                stderr_file.seek(0)
                stderr = stderr_file.read().decode("utf-8", errors="ignore")
                if returncode != 0 or pipe_error is not None:
                    raise RuntimeError(
                        f"ffmpeg finalize failed for {output_path} "
                        f"(exit code {returncode}): {stderr}"
                    ) from pipe_error
            finally:
                if process.stdin is not None and not process.stdin.closed:
                    try:
                        process.stdin.close()
                    except BrokenPipeError:
                        pass
                if process.poll() is None:
                    process.kill()
                    process.wait()
        return output_path

    def close(self) -> None:
        for process in (self._source_stream, self._mask_stream, self._writer_process):
            if process is None:
                continue
            try:
                if process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
            except Exception:
                pass
            try:
                if process.stdout is not None and not process.stdout.closed:
                    process.stdout.close()
            except Exception:
                pass
            try:
                if process.stderr is not None and not process.stderr.closed:
                    process.stderr.close()
            except Exception:
                pass
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            except Exception:
                pass
        self._writer_process = None
        # These mmap-backed stores are only temporary runtime scratch space.
        # Avoid forcing large flushes during teardown; the caller either already
        # finalized the output video or is discarding the scratch store.
        try:
            del self._final_store
        except Exception:
            pass
        try:
            del self._mask_store
        except Exception:
            pass
        try:
            del self._written_store
        except Exception:
            pass
        try:
            del self._source_store
        except Exception:
            pass
        try:
            shutil.rmtree(self._workdir, ignore_errors=True)
        except Exception:
            pass


def read_mask_rgb_max(video_path, *, width, height, num_frames, chunk_frames=16, checkpoint=None):
    """Find the preload RGB-mask global maximum with bounded memory.

    Stop at 255: no later uint8 frame can raise the threshold. Soft/dark masks
    require a full scan so chunk boundaries cannot change binarization.
    """
    reader = SequentialVideoReader(video_path, width=width, height=height,
                                   pix_fmt="rgb24", thread_count="auto")
    maximum = 0
    try:
        for start in range(0, num_frames, chunk_frames):
            if checkpoint is not None:
                checkpoint()
            frames = reader.read_frames(min(chunk_frames, num_frames - start))
            maximum = max(maximum, int(frames[..., 0].max()))
            if maximum == 255:
                break
    finally:
        reader.close()
    return maximum
