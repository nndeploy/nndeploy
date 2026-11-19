# SPDX-License-Identifier: Apache-2.0
"""Constant-memory streaming restore: read -> generate -> write as three stages.

Port of ``infer/stream.py``.

The whole-volume path keeps roughly four copies of the entire clip in host RAM,
so peak memory grows with the frame count. Streaming removes the two
synchronisation barriers that force that:

* **Temporal blending no longer waits for every window.** A frame is retired as
  soon as no unprocessed window can still cover it, which for a monotonic
  sliding window is ``[this window's start, next window's start)``. The only
  frames kept in hand are those a later window will touch, held as a weighted
  accumulator (``live_acc`` / ``live_w``) normalised on the way out.
* **Encoding no longer waits for the full volume.** Each retired range goes
  straight to the encoder and is dropped.

Spatial blending is untouched: the shared :func:`tiled_restore_rect` is called
once per temporal chunk with ``t_overlap=0``, which degenerates its 3D feather
mask to a purely spatial one and shrinks its accumulator from the whole volume
to a single chunk.

Two things the port must not "improve", both of which change the output:

* the temporal blend is a *normalised accumulation*, not a two-sided cross-fade.
  Where a frame is covered by >= 3 windows the two differ, and that case is not
  exotic -- ``T=64`` with the default tiling already hits it (``requirements.md``
  §7-4).
* ``color_ref`` correction is applied per chunk and **before** the temporal
  blend, over the whole chunk including overlap regions that will later be
  blended away (§7-5).

The streaming core takes explicit geometry / queue / IO arguments and never
looks at a CLI object, so the same code can be wrapped by a local CLI or by a
server later (``requirements.md`` §5.2).
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path

import torch
import torch.nn.functional as F
from nndeploy.e_dit.algorithms.vsr.audio import video_output
from nndeploy.e_dit.algorithms.vsr.blending import (
    temporal_weight,
    tiled_restore_rect,
)
from nndeploy.e_dit.algorithms.vsr.color import (
    match_color,
    match_color_to_stats,
    scan_color_reference,
)
from nndeploy.e_dit.algorithms.vsr.geometry import (
    ALIGN,
    compute_tile_positions,
    pad_hw_to_multiple,
    reflect_pad_time,
    resize_video,
)
from nndeploy.e_dit.algorithms.vsr.video_io import (
    WindowReader,
    open_video_writer,
    probe_video,
    to_uint8_hwc,
)

#: ``global`` keeps the whole-volume path's colour semantics via a cheap
#: pre-pass; ``chunk`` matches every chunk against its own input (pure
#: streaming); ``none`` disables colour correction.
COLOR_REF_MODES = ("global", "chunk", "none")


class _WorkerError:
    """Carries a worker-thread exception across a queue.

    Raising inside the reader would leave the main thread blocked in ``get()``
    forever, so the exception travels as an ordinary queue item instead.
    """

    __slots__ = ("exc",)

    def __init__(self, exc: BaseException):
        self.exc = exc


def _drain(q: queue.Queue) -> None:
    """Empty a bounded queue so a worker parked in ``put()`` can wake up."""
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


def _put_read_item(read_q, item, stop_evt):
    """Allow request cancellation to release a reader blocked by backpressure."""
    while not stop_evt.is_set():
        try:
            read_q.put(item, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


def _reader_worker(
    path,
    t_pos: list[tuple[int, int]],
    target_h: int,
    target_w: int,
    padded_h: int,
    padded_w: int,
    tile_t: int,
    read_q: queue.Queue,
    stop_evt: threading.Event,
) -> None:
    """Decode one temporal window at a time and hand it over model-ready."""
    try:
        # The decoder handle is built *inside* the thread on purpose: decord
        # readers are single-thread affine.
        reader = WindowReader(path)
        pad_h, pad_w = padded_h - target_h, padded_w - target_w

        for i, (ts, te) in enumerate(t_pos):
            if stop_evt.is_set():
                return
            # fp32 all the way: CPU has no bicubic / replicate-pad kernel for
            # bf16 or fp16. The cast to the model dtype happens on the GPU.
            x = reader.read(ts, te)
            x = resize_video(x, target_h, target_w)
            if x.shape[2] < tile_t:          # whole clip shorter than one window
                x = reflect_pad_time(x, tile_t)
            if pad_h or pad_w:
                # Pad right / bottom only, so nothing shifts and a plain crop
                # undoes it.
                x = F.pad(x, (0, pad_w, 0, pad_h, 0, 0), mode="replicate")
            if not _put_read_item(read_q, (i, ts, te, x.contiguous()), stop_evt):
                return
            del x

        _put_read_item(read_q, None, stop_evt)
    except BaseException as exc:  # noqa: BLE001 - relay worker failures to caller
        _put_read_item(read_q, _WorkerError(exc), stop_evt)


def _writer_worker(writer, write_q: queue.Queue, errors: list[BaseException]) -> None:
    """Append uint8 ``[n, H, W, C]`` batches in order until the None sentinel.

    Encoding is inherently sequential, so this stays a single thread. After a
    failure it keeps draining: the main thread would otherwise block forever on
    a full queue instead of seeing the error.
    """
    while True:
        item = write_q.get()
        if item is None:
            return
        if errors:
            continue
        try:
            for frame in item:
                writer.append_data(frame)
        except BaseException as exc:  # noqa: BLE001 - relay worker failures to caller
            errors.append(exc)


def stream_restore(
    restorer,
    input_path,
    output_path,
    *,
    preserve_audio: bool = True,
    check_interrupt=None,
    **kwargs,
) -> int:
    """Restore and atomically publish a video, preserving all source audio by default.

    Geometry, tiling and encoding options are forwarded to ``_stream_restore_video``.
    Cancellation and deadlines cover both inference and audio finalization.
    """
    with video_output(
        input_path, output_path, preserve_audio=preserve_audio,
        check_interrupt=check_interrupt,
    ) as video_path:
        written = _stream_restore_video(
            restorer, input_path, video_path, check_interrupt=check_interrupt, **kwargs,
        )
    return written


@torch.no_grad()
def _stream_restore_video(
    restorer,
    input_path,
    output_path,
    *,
    target_h: int,
    target_w: int,
    fps: float | None = None,
    total_frames: int | None = None,
    color_ref: str = "global",
    color_samples: int = 64,
    crf: int = 5,
    read_queue: int = 2,
    write_queue: int = 4,
    gpu_postprocess: bool = False,
    align: int = ALIGN,
    show_progress: bool = True,
    save_tiles_dir: str | None = None,
    check_interrupt=None,
) -> int:
    """Restore ``input_path`` to ``output_path`` with memory independent of length.

    Args:
        restorer: a :class:`~...vsr.model.VSRRestorer`; its ``tile_*`` /
            ``*_overlap`` attributes define the windows.
        target_h/w: output geometry in pixels (resolve ``--long_edge`` first).
        fps/total_frames: probed from the file when omitted.
        color_ref: one of :data:`COLOR_REF_MODES`.
        color_samples: frames sampled by the ``global`` pre-pass; 0 = all.
        read_queue: decoded chunks held in flight -- the memory knob.
        write_queue: encoded (uint8) batches held in flight.
        gpu_postprocess: keep spatial/colour/temporal operations on the model device.
            Uses additional memory proportional to one spatial chunk.
        save_tiles_dir: debug aid; each chunk's tiles land in ``chunkNNN/``.

    Returns:
        The number of frames written, which always equals the source count.
    """
    if color_ref not in COLOR_REF_MODES:
        raise ValueError(f"color_ref must be one of {COLOR_REF_MODES}, got {color_ref!r}")

    tile_t = int(restorer.tile_t)
    t_overlap = int(restorer.t_overlap)
    if t_overlap > tile_t // 2 and show_progress:
        # Handled correctly (the blend below normalises by the accumulated
        # weight however many windows cover a frame), just wasteful.
        print(f"[stream] warning: temporal_overlap={t_overlap} is more than half "
              f"of tile_t={tile_t}; every frame gets restored 3+ times")

    if fps is None or total_frames is None:
        probed_fps, probed_t, _, _ = probe_video(input_path)
        fps = probed_fps if fps is None else fps
        total_frames = probed_t if total_frames is None else total_frames
    T = int(total_frames)
    if T <= 0:
        raise ValueError(f"{input_path} decodes to {T} frames")

    padded_h, padded_w, _, _ = pad_hw_to_multiple(target_h, target_w, align)

    # Plan every window up front. The *actual* overlap is read back out of these
    # positions below, never from t_overlap: the last window is shifted back to
    # stay full-size, so its overlap with its predecessor is usually larger.
    t_pos = compute_tile_positions(T, tile_t, t_overlap)
    n_chunks = len(t_pos)

    if show_progress:
        chunk_bytes = 3 * tile_t * padded_h * padded_w * 4          # fp32 volume
        out_bytes = 3 * tile_t * target_h * target_w                # uint8 output
        live = (read_queue + 1) * chunk_bytes
        live += chunk_bytes * 4 / 3
        live += chunk_bytes * 2
        live += write_queue * out_bytes
        print(f"[stream] {T} frames -> {n_chunks} chunks of {tile_t} frames "
              f"(stride {max(1, tile_t - t_overlap)}), "
              f"{target_h}x{target_w} padded to {padded_h}x{padded_w}")
        print(f"[stream] color_ref={color_ref}, queues read={read_queue} "
              f"write={write_queue} -> ~{live / 2 ** 30:.1f} GiB peak host RAM")

    ref_mean = ref_std = None
    if color_ref == "global":
        ref_mean, ref_std = scan_color_reference(
            input_path, T, target_h, target_w, max_samples=color_samples,
        )
        if show_progress:
            n_scanned = T if (not color_samples or color_samples >= T) else color_samples
            print(f"[stream] global color reference from {n_scanned} sampled frames: "
                  f"mean={ref_mean.flatten().tolist()}, std={ref_std.flatten().tolist()}")

    writer = open_video_writer(output_path, fps, crf=crf)
    read_q: queue.Queue = queue.Queue(maxsize=max(1, read_queue))
    write_q: queue.Queue = queue.Queue(maxsize=max(1, write_queue))
    stop_evt = threading.Event()
    writer_errors: list[BaseException] = []

    reader_t = threading.Thread(
        target=_reader_worker, name="vsr-stream-reader", daemon=True,
        args=(input_path, t_pos, target_h, target_w, padded_h, padded_w,
              tile_t, read_q, stop_evt),
    )
    writer_t = threading.Thread(
        target=_writer_worker, name="vsr-stream-writer", daemon=True,
        args=(writer, write_q, writer_errors),
    )
    reader_t.start()
    writer_t.start()

    live_acc = None   # [1, C, n_live, H, W]  sum of w_j * v_j, un-normalised
    live_w = None     # [n_live]               sum of w_j
    emitted = 0
    try:
        for expected in range(n_chunks):
            item = read_q.get()
            if item is None:
                raise RuntimeError(f"reader ended after {expected}/{n_chunks} chunks")
            if isinstance(item, _WorkerError):
                raise item.exc
            if writer_errors:
                raise writer_errors[0]
            i, ts, te, chunk = item

            # --- generate: spatial tiling only -----------------------------
            # tile_t = the whole chunk and t_overlap = 0 make the shared tiling
            # helper degenerate to one temporal window, i.e. pure spatial
            # feathering, with a one-chunk accumulator instead of a whole-volume
            # one. The spatial blending code itself is unchanged.
            tile_dir = (str(Path(save_tiles_dir) / f"chunk{i:03d}")
                        if save_tiles_dir is not None else None)
            restored = tiled_restore_rect(
                chunk.to(restorer.device) if gpu_postprocess else chunk,
                restorer.restore_window,
                check_interrupt=check_interrupt,
                restore_windows_fn=getattr(restorer, "restore_windows", None),
                tile_t=chunk.shape[2],
                t_overlap=0,
                tile_h=restorer.tile_h,
                tile_w=restorer.tile_w,
                s_overlap=restorer.s_overlap,
                show_progress=show_progress,
                save_dir=tile_dir,
                tile_fps=fps,
                chunk_label=f"chunk {i + 1}/{n_chunks} src frames [{ts}:{te}) of {T}",
            )
            restored = restored[:, :, :tile_t, :target_h, :target_w].float()

            # --- colour correction ------------------------------------------
            if color_ref == "global":
                restored = match_color_to_stats(restored, ref_mean, ref_std)
            elif color_ref == "chunk":
                ref = chunk[:, :, :tile_t, :target_h, :target_w].float()
                restored = match_color(restored, ref)
                del ref
            del chunk

            # --- temporal blend --------------------------------------------
            # Overlaps are read back out of the planned positions, never taken
            # from t_overlap: the last window is shifted back to stay full-size,
            # so the last two overlaps are usually larger than requested.
            #
            # The retired range below covers everything up to the next window's
            # start; the last chunk goes to its own end, which also crops the
            # temporal reflect-pad of a clip shorter than one window.
            ov_prev = 0 if i == 0 else (t_pos[i - 1][1] - ts)
            emit_end = (t_pos[i + 1][0] - ts) if i < n_chunks - 1 else (te - ts)
            ov_next = (tile_t - emit_end) if i < n_chunks - 1 else 0

            # Weight this window and fold in what earlier windows contributed to
            # the frames still in play. Both buffers stay un-normalised until a
            # frame retires, so a frame covered by three windows is averaged,
            # not faded twice.
            w = temporal_weight(tile_t, ov_prev, ov_next, restored.dtype).to(restored.device)
            restored.mul_(w.view(1, 1, tile_t, 1, 1))
            if ov_prev > 0:
                if live_acc is None or live_acc.shape[2] != ov_prev:
                    have = None if live_acc is None else live_acc.shape[2]
                    raise RuntimeError(
                        f"chunk {i}: need a {ov_prev}-frame live range, have {have}")
                restored[:, :, :ov_prev] += live_acc
                w[:ov_prev] += live_w
                live_acc = live_w = None

            # --- retire: no later window can touch these frames -------------
            done = restored[:, :, :emit_end]
            done.div_(w[:emit_end].view(1, 1, emit_end, 1, 1).clamp_min(1e-8))
            write_q.put(to_uint8_hwc(done))
            emitted += emit_end
            del done

            # --- keep only what a later window can still touch --------------
            # .clone() is not optional: a view would pin the whole chunk's
            # storage and O(1) memory would be lost.
            if emit_end < tile_t:
                live_acc = restored[:, :, emit_end:tile_t].clone()
                live_w = w[emit_end:tile_t].clone()
            del restored

        write_q.put(None)
        # Consume the reader's end-of-stream sentinel so its final put() can
        # never sit on a full queue while we join it.
        _drain(read_q)
        reader_t.join(timeout=60)
        writer_t.join()
    except BaseException:
        stop_evt.set()
        # A bounded queue makes stop_evt alone insufficient: the reader may be
        # parked in put() and never reach the flag check. Drain it awake first.
        _drain(read_q)
        _drain(write_q)
        try:
            write_q.put_nowait(None)
        except queue.Full:
            pass
        writer_t.join(timeout=5)
        raise
    finally:
        writer.close()

    if writer_errors:
        raise writer_errors[0]
    # Catches essentially every off-by-one in the retirement arithmetic.
    if emitted != T:
        raise RuntimeError(f"emitted {emitted} frames but the source has {T}")
    return emitted
