# SPDX-License-Identifier: Apache-2.0
"""Window-scoped VideoEdit orchestration, independent of the model executor."""

from __future__ import annotations

import os
import tempfile
from nndeploy.e_dit.video.artifacts import video_output_paths, validate_output_paths
from contextlib import ExitStack

import cv2
import numpy as np
from PIL import Image
from nndeploy.e_dit.algorithms.wan.video.chunk_plan import (
    assign_chunk_bboxes,
    merge_order,
    plan_clips,
    scan_mask_bboxes,
)
from nndeploy.e_dit.algorithms.wan.video.composite import paste_back_frame
from nndeploy.e_dit.algorithms.wan.video.ffmpeg_io import mux_source_audio
from nndeploy.e_dit.algorithms.wan.video.mask_io import (
    _decode_coco_rle,
    _detect_mask_type,
    _load_coco_records,
    _load_numpy_mask_array,
    _to_binary_pil,
)
from nndeploy.e_dit.algorithms.wan.video.mask_stabilize import (
    make_window_stabilizer,
)
from nndeploy.e_dit.algorithms.wan.video.preprocess import (
    crop_frames,
    expand_mask_frames,
    resize_frames,
)
from nndeploy.e_dit.algorithms.wan.video.progress import (
    build_window_progress_payload,
    write_videoedit_progress,
)
from nndeploy.e_dit.algorithms.wan.video.stream_io import (
    VideoWriterFfmpeg,
    copy_video,
    encode_profile,
    merge_clips,
    open_reader,
    probe_video,
)
from nndeploy.e_dit.algorithms.wan.video.windowing import (
    build_videoedit_pass_window_specs,
)


class MaskReader:
    """Adapt existing video/NumPy/COCO contracts to frame-range reads."""

    def __init__(self, path, size, ranges=None):
        self.size = size
        self.video = None
        self.array = None
        self.records = None
        kind = _detect_mask_type(path)
        if kind == "video":
            self.video = open_reader(path, ranges)
            if (self.video.width, self.video.height) != size:
                self.video.close()
                raise ValueError("VideoEdit video/mask size mismatch")
            self.num_frame = self.video.num_frame
        elif kind == "coco":
            self.records = _load_coco_records(path)
            self.num_frame = len(self.records)
        else:
            # Numeric .npy files stay memory mapped. Legacy object/NPZ payloads
            # retain their existing eager-loading contract.
            try:
                self.array = np.load(path, mmap_mode="r", allow_pickle=False)
                if not isinstance(self.array, np.ndarray):
                    self.array.close()
                    self.array = _load_numpy_mask_array(path)
            except ValueError:
                self.array = _load_numpy_mask_array(path)
            if self.array.ndim == 2:
                self.array = self.array[None]
            self.num_frame = len(self.array)

    def read_range(self, lo, hi):
        if self.video is not None:
            return self.video.read_range(lo, hi)
        frames = []
        for index in range(lo, hi):
            if self.records is not None:
                record = self.records[index]
                size = record["size"]
                arr = np.zeros(size, dtype=np.uint8)
                for item in record.get("counts", []):
                    value = item["mask"]
                    item_size = (
                        value.get("size", size) if isinstance(value, dict) else size
                    )
                    counts = value["counts"] if isinstance(value, dict) else value
                    arr |= (_decode_coco_rle(item_size, counts) > 0).astype(np.uint8)
            else:
                arr = self.array[index]
                if self.array.ndim == 4 and not (
                    self.array.shape[-1] in (1, 3, 4) and self.array.shape[1] > 4
                ):
                    arr = arr.max(axis=0)
            frames.append(np.asarray(_to_binary_pil(arr, self.size).convert("RGB")))
        return frames

    def close(self):
        if self.video is not None:
            self.video.close()
        self.array = None
        self.records = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class EagerReader:
    """Explicit eager input mode, sharing the same window algorithm."""

    def __init__(self, reader):
        self.reader = reader
        try:
            self.frames = reader.read_range(0, reader.num_frame)
        except BaseException:
            reader.close()
            raise

    def read_range(self, lo, hi):
        return self.frames[lo:hi]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.frames = None
        self.reader.close()


def run_streaming_edit(
    params,
    generate,
    output_path,
    *,
    write_output=True,
    collect_frames=False,
    check_cancel=lambda: None,
):
    """Run the native window algorithm with a model callback.

    ``generate(frames, masks, spec, chunk)`` returns one window of RGB PIL frames.
    Only the output rank writes files; every rank runs identical model callbacks.
    Pixel memory is bounded by a window plus overlap/bridge, excluding model state.
    """
    output_paths = video_output_paths(output_path, include_crop=params.save_crop_only) if write_output else []
    if write_output:
        validate_output_paths(output_paths, (params.video_input_path, params.mask_input_path,
                                             params.reference_image_path))
    meta = probe_video(params.video_input_path)
    width, height = meta["width"], meta["height"]
    if min(width, height) < 16:
        raise ValueError("VideoEdit source dimensions must be at least 16 pixels")
    total = meta["num_frame"]
    if params.num_frames is not None and params.num_frames > 0:
        total = min(total, params.num_frames)
    clips = plan_clips(
        total,
        params.ref_frame_idx,
        params.infer_len,
        params.overlap,
        params.bridge_overlap,
    )
    params.runtime_num_input_frames = total
    params.runtime_fps = meta["fps"]
    params.runtime_window_specs = [
        spec
        for clip in clips
        for spec in build_videoedit_pass_window_specs(
            clip.seq_idx, params.infer_len, params.overlap
        )
    ]
    total_windows = len(params.runtime_window_specs)
    steps_per_window = max(1, int(getattr(params, "num_inference_steps", 1)))
    check_cancel()
    with MaskReader(
        params.mask_input_path, (width, height), [(0, min(total, 32))]
    ) as reader:
        if reader.num_frame != meta["num_frame"]:
            raise ValueError("VideoEdit video/mask length mismatch")
        boxes = scan_mask_bboxes(
            reader, params.dilate_px, params.mask_scale, frame_limit=total
        )
    assign_chunk_bboxes(
        clips,
        boxes,
        height,
        width,
        mode=params.chunk_bbox_mode,
        expand_scale=2 * params.bbox_expand_scale + 1,
        bbox_padding=params.bbox_padding,
        feather_px=params.feather_px,
    )
    del boxes
    stabilize = make_window_stabilizer(
        params.stabilize_mask_union,
        params.stabilize_mask_shape,
        params.stabilize_smooth_window,
    )
    with Image.open(params.reference_image_path) as image:
        reference = np.asarray(
            image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
        )
    canvas_h = max(c.crop_h for p in clips for c in p.chunks)
    canvas_w = max(c.crop_w for p in clips for c in p.chunks)
    profile = encode_profile(meta)
    intermediate = (
        encode_profile(meta, lossless=True)
        if any(p.reverse for p in clips)
        else profile
    )
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if write_output:
        os.makedirs(output_dir, exist_ok=True)
    records = []
    bridge = []
    committed = set()
    collected = [None] * total if collect_frames else None
    audio_preserved = False
    with ExitStack() as stack:
        work = (
            stack.enter_context(
                tempfile.TemporaryDirectory(prefix=".videoedit-", dir=output_dir)
            )
            if write_output
            else None
        )
        paths = {}
        for clip in clips:
            carry = bridge if clip.label == "short" else [reference]
            specs = build_videoedit_pass_window_specs(
                clip.seq_idx, params.infer_len, params.overlap
            )
            with ExitStack() as clip_stack:
                video = clip_stack.enter_context(
                    open_reader(params.video_input_path, clip.read_ranges())
                )
                mask = clip_stack.enter_context(
                    MaskReader(
                        params.mask_input_path, (width, height), clip.read_ranges()
                    )
                )
                if getattr(params, "decode_mode", "stream") == "eager":
                    video = clip_stack.enter_context(EagerReader(video))
                    mask = clip_stack.enter_context(EagerReader(mask))
                writers = {}
                variants = ["full"] + (["crop"] if params.save_crop_only else [])
                if write_output:
                    for variant in variants:
                        path = os.path.join(work, f"{clip.label}-{variant}.mkv")
                        # Intermediate container must support the source encoder.
                        if intermediate.get("vcodec") == "prores_ks":
                            path = os.path.splitext(path)[0] + ".mov"
                        paths[clip.label, variant] = path
                        out_w, out_h = (
                            (width, height)
                            if variant == "full" and params.enable_paste_back
                            else (canvas_w, canvas_h)
                        )
                        writers[variant] = clip_stack.enter_context(
                            VideoWriterFfmpeg(
                                path, out_w, out_h, meta["fps"], intermediate
                            )
                        )
                for chunk, spec in zip(clip.chunks, specs, strict=True):
                    check_cancel()
                    params.runtime_progress_window_index = len(records)
                    if write_output:
                        payload = build_window_progress_payload(
                            stage="window_start",
                            total_frames=total,
                            infer_len=params.infer_len,
                            overlap=params.overlap,
                            total_windows=total_windows,
                            current_window_index=len(records),
                            steps_per_window=steps_per_window,
                        )
                        payload["pass"] = clip.label
                        write_videoedit_progress(
                            getattr(params, "progress_path", None), payload
                        )
                    lo, hi = chunk.read_range or (0, 0)
                    source = video.read_range(lo, hi) if hi else None
                    masks = mask.read_range(lo, hi) if hi else None
                    raw_frames, raw_masks = [], []
                    for i, g in enumerate(chunk.context):
                        raw_frames.append(
                            Image.fromarray(
                                carry[i] if i < len(carry) else source[g - lo]
                            )
                        )
                        raw_masks.append(
                            Image.new("L", (width, height), 0)
                            if g is None
                            else Image.fromarray(masks[g - lo]).convert("L")
                        )
                    processed = expand_mask_frames(
                        raw_masks, params.dilate_px, params.mask_scale
                    )
                    frames = crop_frames(raw_frames, chunk.bbox)
                    window_masks = crop_frames(processed, chunk.bbox)
                    if (chunk.crop_h, chunk.crop_w) != (
                        chunk.aligned_h,
                        chunk.aligned_w,
                    ):
                        frames = resize_frames(frames, chunk.aligned_h, chunk.aligned_w)
                        window_masks = resize_frames(
                            window_masks, chunk.aligned_h, chunk.aligned_w
                        )
                    for i in range(min(len(carry), len(window_masks))):
                        window_masks[i] = Image.new(
                            "L", (chunk.aligned_w, chunk.aligned_h), 0
                        )
                    if stabilize:
                        window_masks = stabilize(window_masks)
                    valid = len(frames)
                    for i in range(params.infer_len - valid):
                        mirror = max(valid - 1 - i, 0)
                        frames.append(frames[mirror])
                        window_masks.append(window_masks[mirror])
                    generated = generate(frames, window_masks, spec, chunk)
                    if len(generated) != params.infer_len:
                        raise ValueError(
                            "VideoEdit model returned an incomplete window"
                        )
                    composites = {}

                    def composite(
                        i,
                        *,
                        composites=composites,
                        chunk=chunk,
                        source=source,
                        lo=lo,
                        carry=carry,
                        generated=generated,
                        window_masks=window_masks,
                    ):
                        if i not in composites:
                            g = chunk.context[i]
                            original = source[g - lo] if g is not None else carry[i]
                            composites[i] = paste_back_frame(
                                original,
                                generated[i],
                                window_masks[i],
                                chunk.bbox,
                                chunk.crop_h,
                                chunk.crop_w,
                                params.feather_px,
                                params.adain_boundary_dilate,
                                crop_edge_feather=params.crop_edge_feather,
                            )
                        return composites[i]

                    for i in range(*chunk.owned):
                        g = chunk.context[i]
                        if g is None:
                            continue
                        if g in committed:
                            raise RuntimeError(f"Duplicate VideoEdit source frame {g}")
                        committed.add(g)
                        if write_output or collect_frames:
                            crop = np.asarray(generated[i])
                            if crop.shape[:2] != (chunk.crop_h, chunk.crop_w):
                                crop = cv2.resize(crop, (chunk.crop_w, chunk.crop_h))
                            pad_h, pad_w = (
                                canvas_h - chunk.crop_h,
                                canvas_w - chunk.crop_w,
                            )
                            padded = np.pad(
                                crop,
                                (
                                    (pad_h // 2, pad_h - pad_h // 2),
                                    (pad_w // 2, pad_w - pad_w // 2),
                                    (0, 0),
                                ),
                            )
                            full = composite(i) if params.enable_paste_back else padded
                            if write_output:
                                writers["full"].write(full)
                            if collect_frames:
                                collected[g] = Image.fromarray(full)
                            if write_output and "crop" in writers:
                                writers["crop"].write(padded)
                    if clip.label == "long" and chunk.w_idx == 0 and len(clips) > 1:
                        bridge = [composite(i) for i in range(1, 1 + clips[1].lead)][
                            ::-1
                        ]
                    stride = params.infer_len - params.overlap
                    next_carry = [
                        composite(i)
                        for i in range(stride, min(stride + params.overlap, valid))
                    ]
                    records.append(
                        {
                            "pass": clip.label,
                            "window_index": chunk.w_idx,
                            "start_index": chunk.seq_start,
                            "valid_len": valid,
                            "reflected_count": params.infer_len - valid,
                            "bbox": chunk.bbox,
                            "crop_h": chunk.crop_h,
                            "crop_w": chunk.crop_w,
                            "aligned_h": chunk.aligned_h,
                            "aligned_w": chunk.aligned_w,
                            "committed_global_indices": chunk.owned_global(),
                        }
                    )
                    carry = next_carry
                    if write_output:
                        payload = build_window_progress_payload(
                            stage="window_done",
                            total_frames=total,
                            infer_len=params.infer_len,
                            overlap=params.overlap,
                            total_windows=total_windows,
                            current_window_index=len(records) - 1,
                            current_step_index=steps_per_window - 1,
                            steps_per_window=steps_per_window,
                        )
                        payload["pass"] = clip.label
                        write_videoedit_progress(
                            getattr(params, "progress_path", None), payload
                        )
                    del (
                        composite,
                        source,
                        masks,
                        raw_frames,
                        raw_masks,
                        processed,
                        frames,
                        generated,
                        composites,
                    )
        if committed != set(range(total)):
            raise RuntimeError("VideoEdit output does not cover the source timeline")
        check_cancel()
        if write_output:
            for variant in variants:
                target = output_paths[0 if variant == 'full' else 1]
                staged = os.path.join(
                    work, "final-" + variant + os.path.splitext(target)[1]
                )
                parts = [
                    (paths[label, variant], reverse)
                    for label, reverse in merge_order(clips)
                ]
                if len(parts) == 1 and not parts[0][1]:
                    copy_video(parts[0][0], staged)
                else:
                    merge_clips(parts, staged, meta["fps"], profile)
                if variant == "full" and params.preserve_audio:
                    audio_preserved = mux_source_audio(
                        staged, params.video_input_path, total / meta["fps"]
                    )
                check_cancel()
                os.replace(staged, target)
    return {
        "output_file_paths": output_paths,
        "num_input_frames": total,
        "num_output_frames": total,
        "fps": meta["fps"],
        "width": width if params.enable_paste_back else canvas_w,
        "height": height if params.enable_paste_back else canvas_h,
        "crop_h": canvas_h,
        "crop_w": canvas_w,
        "chunk_bbox_mode": params.chunk_bbox_mode,
        "window_specs": records,
        "drop_reference_frame": False,
        "enable_paste_back": params.enable_paste_back,
        "preserve_audio": params.preserve_audio,
        "audio_preserved": audio_preserved,
        "video_input_path": params.video_input_path,
        "mask_input_path": params.mask_input_path,
        "reference_image_path": params.reference_image_path,
        "bbox_expand_scale": params.bbox_expand_scale,
        "bbox_multiplier": 2 * params.bbox_expand_scale + 1,
        "stabilize_mask_union": params.stabilize_mask_union,
        "stabilize_mask_shape": params.stabilize_mask_shape,
        "frames": collected,
    }
