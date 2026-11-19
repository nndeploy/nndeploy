# SPDX-License-Identifier: Apache-2.0
# Adapted from VideoEdit-diffusers streaming inference utilities (2026-09-23).
"""Streaming video read/write over ffmpeg, for chunked long-video inference.

Trimmed from ``utils/video_info_ffmpeg.py`` / ``utils/video_stream_writer_ffmpeg.py``
down to what inference needs, with their out-of-tree dependencies (``VideoData``,
``VideoInfo``, ``bench.time_bench``, ``common.global_values``) and the hard-coded
``/opt/conda/bin/ffmpeg`` path dropped.

Reading is frame-exact: ranges are selected with ``-vf trim=start_frame=..`` plus
``-vsync 0``, i.e. by absolute frame number with no timestamp seek, so a VFR or
open-GOP source cannot silently shift the video/mask pairing by a frame. Verified
bit-identical to the ``cv2.VideoCapture`` sequential read this replaces.

Two read strategies share one interface:

* :class:`SequentialReader` keeps one decoder process alive and walks forward — one
  decode pass for the whole clip. Used for clips whose chunks advance forward.
* :class:`RandomReader` re-decodes per request. Used for backward-running clips
  (their chunks descend, so a forward pipe cannot serve them).

:func:`open_reader` picks between them from the planned chunk order.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Iterable, Sequence
from fractions import Fraction
from functools import lru_cache

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Probing
# ──────────────────────────────────────────────────────────────────────────────

#: Pixel formats carrying more than 8 bits per component, per ``VideoInfo.is_high_bit``.
_HIGH_BIT_MARKERS = (
    "10le",
    "10be",
    "12le",
    "12be",
    "14le",
    "14be",
    "16le",
    "16be",
    "p010",
)


def _probe(path, count_frames=False):
    command = ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json"]
    if count_frames:
        command += ["-count_frames", "-select_streams", "v:0"]
    result = subprocess.run(
        command + [path], check=True, capture_output=True, text=True
    )
    return json.loads(result.stdout)


def probe_video(path: str) -> dict:
    """ffprobe metadata for the first video stream.

    Ported from ``VideoInfoFfmpeg.get_video_meta_info`` / ``init``, returning the same
    fields as a flat dict (there is no ``VideoInfo`` base class here). ``nb_frames`` is
    missing from some muxers' output, so count decoded frames in that case rather
    than estimating duration*fps (which can omit the last frame).

    ``codec_profile`` and ``source_field_order`` keep their raw ffprobe values;
    :func:`encode_profile` interprets them, mirroring ``construct_profile``'s use of a
    second ``ffmpeg.probe`` on the reference file.
    """
    probe = _probe(path)
    streams = [s for s in probe["streams"] if s["codec_type"] == "video"]
    if not streams:
        raise ValueError(f"[VideoIO] No video stream in {path}")
    s = streams[0]

    fps = float(Fraction(s["avg_frame_rate"]))
    if not fps:
        raise ValueError(f"[VideoIO] Could not determine fps for {path}")

    try:
        num_frame = int(s["nb_frames"])
        duration = float(s["duration"])
    except (KeyError, ValueError):
        counted = _probe(path, count_frames=True)
        num_frame = int(counted["streams"][0]["nb_read_frames"])
        duration = float(
            s.get("duration")
            or probe.get("format", {}).get("duration")
            or num_frame / fps
        )

    if "bit_rate" in s:
        bit_rate = int(s["bit_rate"])
    elif "bit_rate" in probe["format"]:
        bit_rate = int(probe["format"]["bit_rate"])
    else:
        bit_rate = 500000000

    def tag(name: str) -> str | None:
        # ffprobe reports an unset tag as "unknown"; leave it None so the encoder does
        # not write a bogus one. ("reserved" is remapped to bt709 in encode_profile,
        # as the original writer did.)
        value = s.get(name)
        return None if value in (None, "unknown") else value

    pix_fmt = s["pix_fmt"]
    is_high_bit = any(marker in pix_fmt for marker in _HIGH_BIT_MARKERS)

    return dict(
        path=path,
        num_frame=num_frame,
        duration=duration,
        fps=fps,
        width=s["width"],
        height=s["height"],
        channel=1 if pix_fmt in ["gray"] else 3,
        codec_name=s["codec_name"],
        codec_profile=s.get("profile"),
        pix_fmt=pix_fmt,
        is_high_bit=is_high_bit,
        bit_rate=bit_rate,
        color_space=tag("color_space"),
        color_transfer=tag("color_transfer"),
        color_primaries=tag("color_primaries"),
        source_field_order=s.get("field_order"),
        sample_aspect_ratio=s.get("sample_aspect_ratio"),
        display_aspect_ratio=s.get("display_aspect_ratio"),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Reading
# ──────────────────────────────────────────────────────────────────────────────


class _BaseReader:
    """Shared frame bookkeeping. Frames are always returned as RGB uint8 [n,H,W,3].

    RGB (never ``gray``, even for mask videos): ffmpeg's ``gray`` hands back the Y
    plane directly, which differs from the ``BGR -> RGB -> PIL convert("L")`` path the
    rest of the pipeline uses by up to 2/255. That is invisible in the middle of a
    mask but can flip a pixel sitting exactly on the binarisation threshold, so mask
    callers convert from RGB themselves and stay bit-identical to the old behaviour.
    """

    def __init__(self, meta: dict):
        self.meta = meta
        self.path = meta["path"]
        self.num_frame = meta["num_frame"]
        self.fps = meta["fps"]
        self.width = meta["width"]
        self.height = meta["height"]
        self._frame_bytes = self.width * self.height * 3

    def _check_range(self, lo: int, hi: int) -> None:
        if not (0 <= lo < hi <= self.num_frame):
            raise ValueError(
                f"[VideoIO] range [{lo},{hi}) out of bounds for {self.path} "
                f"({self.num_frame} frames)"
            )

    def _reshape(self, raw: bytes, want: int) -> np.ndarray:
        if len(raw) != want * self._frame_bytes:
            raise RuntimeError(
                f"[VideoIO] short read from {self.path}: wanted {want} frames "
                f"({want * self._frame_bytes} bytes), got {len(raw)} bytes"
            )
        return (
            np.frombuffer(raw, np.uint8)
            .reshape(want, self.height, self.width, 3)
            .copy()
        )

    def read_range(self, lo: int, hi: int) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class RandomReader(_BaseReader):
    """Decode an arbitrary frame range on demand.

    Each call re-decodes from the start of the file, so cost is O(hi) per call. Only
    used where a forward pipe cannot be: backward-running clips.
    """

    def read_range(self, lo: int, hi: int) -> np.ndarray:
        self._check_range(lo, hi)
        proc = subprocess.Popen(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                self.path,
                "-map",
                "0:v:0",
                "-vf",
                f"trim=start_frame={lo}:end_frame={hi}",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-vsync",
                "0",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        raw, err = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"[VideoIO] ffmpeg failed on {self.path}: {err.decode(errors='replace')}"
            )
        return self._reshape(raw, hi - lo)


class SequentialReader(_BaseReader):
    """One decoder process walking the file forward.

    ``read_range`` must be called with non-decreasing ``lo``; frames between the
    cursor and ``lo`` are decoded and discarded. Total decode cost for a whole clip
    is one pass regardless of how many chunks ask for frames.

    Consecutive inference windows overlap, so requests step *backwards* by up to
    ``overlap`` frames. ``lookback`` keeps that many already-decoded frames buffered
    so such a request is served from memory instead of forcing a re-decode.
    """

    def __init__(self, meta: dict, lookback: int = 0):
        super().__init__(meta)
        self._proc: subprocess.Popen | None = None
        self._cursor = 0
        self._lookback = max(0, lookback)
        # Frames [cursor - len(tail), cursor), kept to serve overlapping requests.
        self._tail: np.ndarray | None = None

    def _start(self) -> None:
        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                self.path,
                "-map",
                "0:v:0",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-vsync",
                "0",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._cursor = 0

    def _read_exact(self, nbytes: int) -> bytes:
        assert self._proc is not None
        buf = bytearray()
        while len(buf) < nbytes:
            block = self._proc.stdout.read(nbytes - len(buf))
            if not block:
                break
            buf.extend(block)
        return bytes(buf)

    def _buffered_start(self) -> int:
        """Lowest frame index still available from the lookback buffer."""
        return self._cursor - (0 if self._tail is None else len(self._tail))

    def read_range(self, lo: int, hi: int) -> np.ndarray:
        self._check_range(lo, hi)
        if self._proc is None:
            self._start()

        if lo < self._buffered_start():
            raise RuntimeError(
                f"[VideoIO] SequentialReader on {self.path} cannot rewind to {lo} "
                f"(buffered from {self._buffered_start()}, lookback={self._lookback}); "
                f"use RandomReader for this clip"
            )

        # Part of the request may already be buffered from the previous window's overlap.
        from_buffer = None
        if lo < self._cursor:
            assert self._tail is not None
            offset = lo - self._buffered_start()
            from_buffer = self._tail[offset : offset + (hi - lo)]
            lo = min(hi, self._cursor)

        fresh = None
        if lo < hi:
            if lo > self._cursor:
                while self._cursor < lo:
                    self._reshape(self._read_exact(self._frame_bytes), 1)
                    self._cursor += 1
            fresh = self._reshape(
                self._read_exact((hi - lo) * self._frame_bytes), hi - lo
            )
            self._cursor = hi

        if fresh is None:
            out = from_buffer
        elif from_buffer is None:
            out = fresh
        else:
            out = np.concatenate([from_buffer, fresh], axis=0)

        # The tail must stay flush with the cursor. A fully-buffered request does not
        # move the cursor, so it must leave the existing tail alone.
        if self._lookback > 0 and hi == self._cursor:
            self._tail = out[-self._lookback :].copy()
        return out

    def close(self) -> None:
        self._tail = None
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.kill()
        proc.wait()
        proc.stderr.close()


def open_reader(
    path: str, ranges: Sequence[tuple[int, int]] | None = None
) -> _BaseReader:
    """Reader for `path`, sequential when `ranges` only ever moves forward.

    Pass the chunk ranges the caller intends to request, in request order. A forward
    clip gets the single-pass :class:`SequentialReader`, sized with exactly the
    lookback its overlapping windows need; a backward clip (or an unknown order)
    falls back to :class:`RandomReader`.
    """
    meta = probe_video(path)
    if ranges is None:
        return RandomReader(meta)

    ranges = list(ranges)
    lookback = 0
    cursor = 0
    for lo, hi in ranges:
        if lo < cursor:
            lookback = max(lookback, cursor - lo)
        cursor = max(cursor, hi)
    if all(a[0] <= b[0] for a, b in zip(ranges, ranges[1:])):
        return SequentialReader(meta, lookback=lookback)
    return RandomReader(meta)


# ──────────────────────────────────────────────────────────────────────────────
# Writing
# ──────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _ffmpeg_major_version() -> int:
    """Major ffmpeg version, or -1 if it cannot be determined.

    Ported from ``VideoStreamWriterFfmpeg.get_ffmpeg_version``.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], stdout=subprocess.PIPE, text=True, check=True
        )
        match = re.search(r"ffmpeg version (\d+)\.", result.stdout)
        if match:
            return int(match.group(1))
    except (OSError, subprocess.CalledProcessError):
        return -1
    return -1


def _field_order_compatible(profile: dict) -> dict:
    """ffmpeg >= 7 dropped ``-top``; express it as the ``setfield`` filter instead.

    Ported from ``VideoStreamWriterFfmpeg.field_order_compatible``.
    """
    if _ffmpeg_major_version() >= 7:
        if "top" in profile.get("field_order_kwargs", {}):
            if profile["field_order_kwargs"]["top"] == 1:
                del profile["field_order_kwargs"]["top"]
                profile["setfield"] = "tff"
            elif profile["field_order_kwargs"]["top"] == 0:
                del profile["field_order_kwargs"]["top"]
                profile["setfield"] = "bff"
            else:
                raise ValueError(
                    f"unknown -top={profile['field_order_kwargs']['top']} "
                    f"for early ffmpeg version to -vf"
                )
    return profile


_DEFAULT_ENCODE_ARGS = {
    "format": "rawvideo",
    "pix_fmt": "yuv420p",
    "rgb_fmt": "rgb24",
    "fps": 30,
    "vcodec": "libx264",
    "bit_rate": 10000000,
    "field_order": "progressive",
    "field_order_kwargs": {},
}

_PRORES_PROFILE_VALUE = {
    "unknown": "3",
    "proxy": "0",
    "lt": "1",
    "standard": "2",
    "hq": "3",
    "4444": "4",
    "4444 xq": "5",
}


def encode_profile(meta: dict, lossless: bool = False, **overrides) -> dict:
    """Encoder settings that reproduce the source video's format.

    Ported from ``VideoStreamWriterFfmpeg.construct_profile``, with the reference
    file's metadata supplied as ``meta`` (a :func:`probe_video` result) instead of
    being probed again. Keyword ``overrides`` take precedence, as ``args`` did there.

    Everything that defines the *format* comes from the source: codec, pixel format
    (which carries chroma subsampling, bit depth, and — via ``yuvj*`` — full range),
    colour matrix/transfer/primaries, bitrate, field order. Writing a ``yuvj444p``
    source as the usual ``yuv420p`` would both throw away chroma resolution and
    reinterpret the levels as limited-range, shifting every pixel.

    ``lossless`` changes only the *quality* setting (CRF 0 in place of the source's
    bitrate), for intermediates that get re-encoded on merge; without it the final
    output would carry two lossy generations at the source's bitrate.
    """
    profile = dict(meta)
    profile.update(overrides)

    if "cmd_scale" not in profile:
        if profile.get("color_space", None) is not None:
            if "bt601" in profile["color_space"]:
                color_matrix = "bt601"
            elif "bt470" in profile["color_space"]:
                color_matrix = "bt470"
            elif "bt2020" in profile["color_space"]:
                color_matrix = "bt2020"
            else:
                color_matrix = profile["color_space"]
            profile["cmd_scale"] = {
                "in_color_matrix": f"{color_matrix}",
                "out_color_matrix": f"{color_matrix}",
            }

    if "cmd" not in profile:
        cmd = {}
        if profile.get("color_space", None) is not None:
            cmd["colorspace"] = profile["color_space"]
        if profile.get("color_transfer", None) is not None:
            cmd["color_trc"] = profile["color_transfer"]
        if profile.get("color_primaries", None) is not None:
            cmd["color_primaries"] = profile["color_primaries"]
        profile["cmd"] = cmd
    # "reserved" is ffprobe's placeholder for an unset tag; map it to bt709 rather
    # than handing it to the encoder, as the original construct_profile did.
    for key in ("color_trc", "color_primaries", "colorspace"):
        if profile["cmd"].get(key) == "reserved":
            profile["cmd"][key] = "bt709"

    if "vcodec" not in profile:
        vcodec = profile.get("codec_name", "libx264")
        if vcodec.lower() == "prores":
            vcodec = "prores_ks"
        profile["vcodec"] = vcodec
    if "ffmpeg_profile_value" not in profile:
        profile["ffmpeg_profile_value"] = _PRORES_PROFILE_VALUE.get(
            (profile.get("codec_profile") or "unknown").lower(), "3"
        )

    if "field_order" not in profile:
        field_order = profile.get("source_field_order")
        if field_order in ["top", "top_field_first", "tb", "tt", "tff"]:
            profile["field_order"] = "top_field_first"
        elif field_order in ["bottom", "bottom_field_first", "bb", "bt", "bff"]:
            profile["field_order"] = "bottom_field_first"
        elif field_order == "progressive":
            profile["field_order"] = "progressive"
        elif field_order == "interlaced":
            profile["field_order"] = "interlaced"
        elif field_order is None:
            profile["field_order"] = "progressive"

    if "field_order_kwargs" not in profile and "field_order" in profile:
        profile["field_order_kwargs"] = {}
        if profile["field_order"] in [
            "top_field_first",
            "bottom_field_first",
            "interlaced",
        ]:
            profile["field_order_kwargs"]["flags"] = "+ildct+ilme"
            if profile["field_order"] == "top_field_first":
                profile["field_order_kwargs"]["top"] = 1
            elif profile["field_order"] == "bottom_field_first":
                profile["field_order_kwargs"]["top"] = 0
        elif profile["field_order"] == "progressive":
            pass
        else:
            print(
                f"Warning: Unrecognized scan type {profile['field_order']}. "
                f"Defaulting to progressive processing."
            )
        profile = _field_order_compatible(profile)

    for key, value in _DEFAULT_ENCODE_ARGS.items():
        if key not in profile:
            profile[key] = value

    # Frames are fed in as 8-bit RGB regardless of the source's bit depth — the model
    # and paste-back are 8-bit throughout — so `rgb_fmt` stays rgb24 and the encoder
    # up-converts to the source's `pix_fmt` (e.g. yuv422p10le), keeping the container
    # format intact.
    profile["rgb_fmt"] = "rgb24"

    if lossless:
        profile["vcodec"] = "libx264"
        profile["bit_rate"] = None
        profile["crf"] = 0
        # CRF 0 is lossless in YUV, but a 4:2:0 pixel format throws away chroma before
        # the encoder ever sees it — an RGB round-trip through yuv420p/crf0 still
        # shifts pixels by ~40/255, versus ~2/255 through 4:4:4 (matrix rounding only).
        # So the intermediate drops subsampling while keeping the range flavour
        # (yuvj* = full range); the final encode does the one subsampling pass to the
        # source's real pixel format.
        source_pix = profile.get("pix_fmt") or ""
        profile["pix_fmt"] = "yuvj444p" if source_pix.startswith("yuvj") else "yuv444p"
    elif not profile.get("bit_rate"):
        profile["bit_rate"] = None
        profile["crf"] = 12
    return profile


class VideoWriterFfmpeg:
    """Stream RGB uint8 frames into an encoder process.

    Ported from ``VideoStreamWriterFfmpeg.__init__``, taking a ``profile`` from
    :func:`encode_profile` in place of a ``refer_file``. The colour properties are
    tagged on the frames via ``setparams`` *and* passed as output options: as the
    original writer's comment notes, output options alone do not always make the muxer
    emit them, and a downstream reader then guesses bt601 and shifts the colours.
    """

    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        fps: float,
        profile: dict | None = None,
        loglevel: str = "error",
    ):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.width = width
        self.height = height
        self.frames_written = 0
        profile = dict(profile or _DEFAULT_ENCODE_ARGS)
        for key, value in _DEFAULT_ENCODE_ARGS.items():
            profile.setdefault(key, value)
        if (width % 2 or height % 2) and profile.get("vcodec") in (
            "h264",
            "libx264",
            "hevc",
            "libx265",
        ):
            # Preserve exact crop geometry when chroma subsampling would require
            # padding. This is also the eager VideoEdit writer's contract.
            profile["pix_fmt"] = "yuv444p"

        addition_filter = {}
        if "cmd_scale" in profile:
            addition_filter["scale"] = profile["cmd_scale"]
        if "setfield" in profile:
            addition_filter["setfield"] = profile["setfield"]

        setparams = {}
        if profile.get("color_space", None) is not None:
            setparams["colorspace"] = profile["color_space"]
        if profile.get("color_transfer", None) is not None:
            setparams["color_trc"] = profile["color_transfer"]
        if profile.get("color_primaries", None) is not None:
            setparams["color_primaries"] = profile["color_primaries"]
        if setparams:
            addition_filter["setparams"] = setparams

        # The original writer had sample_aspect_ratio commented out of `cmd`. Tag it
        # instead: leaving it unset only *displays* the same for square pixels, and for
        # an anamorphic source it would show at the wrong aspect. Tagging also keeps
        # ffprobe's SAR/DAR fields on the output equal to the source's.
        sar = profile.get("sample_aspect_ratio")
        if sar and sar != "0:1":  # 0:1 is ffprobe's "unknown"
            addition_filter["setsar"] = sar.replace(":", "/")

        addtion_cmd = {}
        if profile.get("vcodec") == "prores_ks":
            addtion_cmd["profile:v"] = profile["ffmpeg_profile_value"]

        command = [
            "ffmpeg",
            "-v",
            loglevel,
            "-y",
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
        filters = []
        for name, value in addition_filter.items():
            if isinstance(value, dict):
                value = ":".join(f"{key}={item}" for key, item in value.items())
            filters.append(f"{name}={value}")
        if filters:
            command += ["-vf", ",".join(filters)]
        command += [
            "-c:v",
            profile.get("vcodec", "libx264"),
            "-pix_fmt",
            profile.get("pix_fmt", "yuv420p"),
            "-r",
            str(fps),
        ]
        if profile.get("bit_rate"):
            command += ["-b:v", str(profile["bit_rate"])]
        else:
            command += ["-crf", str(profile.get("crf", 12))]
        for options in (addtion_cmd, profile["cmd"], profile["field_order_kwargs"]):
            for key, value in options.items():
                command += [f"-{key}", str(value)]
        self._proc = subprocess.Popen(command + [path], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"[VideoIO] frame {frame.shape[:2]} does not match writer "
                f"{(self.height, self.width)} for {self.path}"
            )
        self._proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        self.frames_written += 1

    def close(self) -> None:
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        try:
            proc.stdin.close()
        except BrokenPipeError:
            # The return code below reports encoder failure, and wait still reaps it.
            pass
        finally:
            proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(
                f"[VideoIO] encoder failed for {self.path} (rc={proc.returncode})"
            )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ──────────────────────────────────────────────────────────────────────────────
# Merging clip files
# ──────────────────────────────────────────────────────────────────────────────


def copy_video(src: str, dst: str) -> None:
    """Remux `src` to `dst` without re-encoding."""
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", src, "-map", "0:v:0", "-c", "copy", dst],
        check=True,
    )


def merge_clips(
    parts: Iterable[tuple[str, bool]],
    out_path: str,
    fps: float,
    profile: dict | None = None,
    block: int = 16,
) -> int:
    """Concatenate clip files into one output in global time order.

    ``parts`` is ``(path, reverse)`` pairs already ordered by the global index of
    their first output frame; ``reverse=True`` means that file was written in
    inference order and runs backwards in time.

    Reversed parts are read back-to-front in blocks of ``block`` frames and flipped
    in memory, rather than with ffmpeg's ``reverse`` filter — that filter buffers
    every decoded frame of the stream at once, which for a long clip is exactly the
    memory this whole path exists to avoid.

    Returns the number of frames written.
    """
    parts = list(parts)
    if not parts:
        raise ValueError("[VideoIO] merge_clips got no parts")

    metas = [probe_video(p) for p, _ in parts]
    width, height = metas[0]["width"], metas[0]["height"]
    for meta in metas[1:]:
        if (meta["width"], meta["height"]) != (width, height):
            raise ValueError(
                f"[VideoIO] clip size mismatch: {metas[0]['path']} is {width}x{height}, "
                f"{meta['path']} is {meta['width']}x{meta['height']}"
            )

    written = 0
    with VideoWriterFfmpeg(out_path, width, height, fps, profile=profile) as writer:
        for (path, reverse), meta in zip(parts, metas):
            total = meta["num_frame"]
            if reverse:
                # Descending blocks, each flipped: global-ascending output, bounded memory.
                reader = RandomReader(meta)
                hi = total
                while hi > 0:
                    lo = max(0, hi - block)
                    for frame in reader.read_range(lo, hi)[::-1]:
                        writer.write(frame)
                        written += 1
                    hi = lo
            else:
                reader = SequentialReader(meta)
                lo = 0
                while lo < total:
                    hi = min(total, lo + block)
                    for frame in reader.read_range(lo, hi):
                        writer.write(frame)
                        written += 1
                    lo = hi
            reader.close()
    return written
