# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import threading
from typing import Sequence

from PIL import Image

from nndeploy.e_dit.algorithms.wan.video.contracts import VideoEditWindowSpec
from nndeploy.e_dit.algorithms.wan.video.postprocess import paste_back_frame
from nndeploy.e_dit.algorithms.wan.video.preprocess import (
    VideoEditSequence,
    apply_videoedit_overlap,
    crop_frame,
    expand_mask_frame,
    resize_frame,
)
from nndeploy.e_dit.algorithms.wan.video.stream_decoder import (
    SequentialMaskDecoder,
    SequentialVideoDecoder,
)
from nndeploy.e_dit.algorithms.wan.video.windowing import VideoEditPassPlan


@dataclass
class _FrameEntry:
    index: int
    resized_frame: Image.Image
    resized_mask: Image.Image


class WindowFrameProvider:
    def __init__(
        self,
        *,
        video_input_path: str,
        mask_input_path: str,
        reference_image_path: str | None,
        num_frames: int,
        fps: float,
        frame_size: tuple[int, int],
        bbox: tuple[int, int, int, int],
        crop_h: int,
        crop_w: int,
        aligned_h: int,
        aligned_w: int,
        dilate_px: int,
        mask_scale: float,
        infer_len: int,
        enable_prefetch: bool = True,
        cache_max_frames: int | None = None,
        prefetch_ahead_frames: int | None = None,
    ):
        self.video_input_path = video_input_path
        self.mask_input_path = mask_input_path
        self.reference_image_path = reference_image_path
        self.num_frames = num_frames
        self.fps = fps
        self.frame_size = frame_size
        self.bbox = bbox
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.aligned_h = aligned_h
        self.aligned_w = aligned_w
        self.dilate_px = dilate_px
        self.mask_scale = mask_scale
        self.enable_prefetch = enable_prefetch
        self.cache_max_frames = cache_max_frames or max(infer_len * 2, infer_len + 8)
        self.prefetch_ahead_frames = prefetch_ahead_frames or infer_len
        self._reference_frame = self._load_reference_frame()
        self._resized_reference_frame: Image.Image | None = None
        self._video_decoder = SequentialVideoDecoder(video_input_path)
        self._mask_decoder = SequentialMaskDecoder(mask_input_path, target_size=frame_size)
        self._next_decode_index = 0
        self._target_decode_index = -1
        self._cache: OrderedDict[int, _FrameEntry] = OrderedDict()
        self._decode_error: Exception | None = None
        self._closed = False
        self._cond = threading.Condition()
        self._prefetch_thread: threading.Thread | None = None
        if enable_prefetch:
            self._prefetch_thread = threading.Thread(
                target=self._prefetch_loop,
                name="videoedit-frame-prefetch",
                daemon=True,
            )
            self._prefetch_thread.start()

    @classmethod
    def from_scanned_geometry(
        cls,
        *,
        video_input_path: str,
        mask_input_path: str,
        reference_image_path: str | None,
        scanned_geometry: dict,
        dilate_px: int,
        mask_scale: float,
        infer_len: int,
        enable_prefetch: bool = True,
    ) -> "WindowFrameProvider":
        return cls(
            video_input_path=video_input_path,
            mask_input_path=mask_input_path,
            reference_image_path=reference_image_path,
            num_frames=int(scanned_geometry["num_frames"]),
            fps=float(scanned_geometry["fps"]),
            frame_size=tuple(scanned_geometry["frame_size"]),
            bbox=tuple(scanned_geometry["bbox"]),
            crop_h=int(scanned_geometry["crop_h"]),
            crop_w=int(scanned_geometry["crop_w"]),
            aligned_h=int(scanned_geometry["aligned_h"]),
            aligned_w=int(scanned_geometry["aligned_w"]),
            dilate_px=dilate_px,
            mask_scale=mask_scale,
            infer_len=infer_len,
            enable_prefetch=enable_prefetch,
        )

    def _load_reference_frame(self) -> Image.Image | None:
        if self.reference_image_path is None:
            return None
        with Image.open(self.reference_image_path) as image:
            reference = image.convert("RGB")
        if reference.size != self.frame_size:
            reference = reference.resize(
                self.frame_size, Image.Resampling.BICUBIC
            )
        return reference

    def _make_video_decoder(self) -> SequentialVideoDecoder:
        return SequentialVideoDecoder(self.video_input_path)

    def _make_mask_decoder(self) -> SequentialMaskDecoder:
        return SequentialMaskDecoder(self.mask_input_path, target_size=self.frame_size)

    def _prefetch_loop(self) -> None:
        try:
            while True:
                with self._cond:
                    while (
                        not self._closed
                        and (
                            self._next_decode_index >= self.num_frames
                            or self._next_decode_index > self._target_decode_index
                        )
                    ):
                        self._cond.wait()
                    if self._closed or self._next_decode_index >= self.num_frames:
                        return
                entry = self._decode_one(self._next_decode_index)
                with self._cond:
                    self._store_entry(entry)
                    self._next_decode_index += 1
                    self._cond.notify_all()
        except Exception as exc:  # pragma: no cover - exercised via waiters
            with self._cond:
                self._decode_error = exc
                self._cond.notify_all()

    def _store_entry(self, entry: _FrameEntry) -> None:
        self._cache[entry.index] = entry
        self._cache.move_to_end(entry.index)
        while len(self._cache) > self.cache_max_frames:
            self._cache.popitem(last=False)

    def _decode_one(self, global_index: int) -> _FrameEntry:
        frame = self._video_decoder.read_next()
        raw_mask = self._mask_decoder.read_next()
        if frame is None or raw_mask is None:
            raise RuntimeError(
                "Unexpected end of input while streaming VideoEdit frames "
                f"at global index {global_index}"
            )
        expanded_mask = expand_mask_frame(
            raw_mask,
            dilate_px=self.dilate_px,
            scale=self.mask_scale,
            force_zero=False,
        )
        cropped_frame = crop_frame(frame, self.bbox)
        cropped_mask = crop_frame(expanded_mask, self.bbox)
        return _FrameEntry(
            index=global_index,
            resized_frame=resize_frame(cropped_frame, self.aligned_h, self.aligned_w),
            resized_mask=resize_frame(cropped_mask, self.aligned_h, self.aligned_w),
        )

    def _ensure_decoded_through(self, index: int) -> None:
        if index < 0:
            return
        if self.enable_prefetch:
            self._wait_for_index(index)
            return
        self._decode_sync_until(index)

    def _decode_sync_until(self, index: int) -> None:
        while self._next_decode_index <= index:
            self._raise_if_error()
            entry = self._decode_one(self._next_decode_index)
            self._store_entry(entry)
            self._next_decode_index += 1

    def _wait_for_index(self, index: int) -> None:
        if index >= self.num_frames:
            raise IndexError(f"Frame index out of range: {index} >= {self.num_frames}")
        target = min(self.num_frames - 1, index + self.prefetch_ahead_frames)
        with self._cond:
            self._target_decode_index = max(self._target_decode_index, target)
            self._cond.notify_all()
            while self._next_decode_index <= index and self._decode_error is None and not self._closed:
                self._cond.wait()
            self._raise_if_error()

    def _reopen_decode_entry(self, index: int) -> _FrameEntry:
        video_decoder = self._make_video_decoder()
        mask_decoder = self._make_mask_decoder()
        try:
            for global_index in range(index + 1):
                frame = video_decoder.read_next()
                raw_mask = mask_decoder.read_next()
                if frame is None or raw_mask is None:
                    raise RuntimeError(
                        f"Unexpected end of input while reopening frame {index}"
                    )
                if global_index == index:
                    expanded_mask = expand_mask_frame(
                        raw_mask,
                        dilate_px=self.dilate_px,
                        scale=self.mask_scale,
                        force_zero=False,
                    )
                    cropped_frame = crop_frame(frame, self.bbox)
                    cropped_mask = crop_frame(expanded_mask, self.bbox)
                    return _FrameEntry(
                        index=index,
                        resized_frame=resize_frame(cropped_frame, self.aligned_h, self.aligned_w),
                        resized_mask=resize_frame(cropped_mask, self.aligned_h, self.aligned_w),
                    )
            raise RuntimeError(f"Could not reopen frame {index}")
        finally:
            video_decoder.close()
            mask_decoder.close()

    def _lookup_entry(self, index: int) -> _FrameEntry:
        entry = self._cache.get(index)
        if entry is not None:
            return entry
        return self._reopen_decode_entry(index)

    def _raise_if_error(self) -> None:
        if self._decode_error is not None:
            raise RuntimeError("VideoEdit streaming decode failed") from self._decode_error

    def materialize_window(self, input_indices: list[int]) -> tuple[list[Image.Image], list[Image.Image]]:
        if not input_indices:
            return [], []
        self._ensure_decoded_through(max(input_indices))
        frames = []
        masks = []
        for index in input_indices:
            entry = self._lookup_entry(index)
            frames.append(entry.resized_frame.copy())
            masks.append(entry.resized_mask.copy())
        return frames, masks

    def get_resized_frame(self, index: int) -> Image.Image:
        self._ensure_decoded_through(index)
        return self._lookup_entry(index).resized_frame.copy()

    def get_resized_reference_frame(self) -> Image.Image | None:
        """Return the edited reference through the same crop/resize geometry."""

        if self._reference_frame is None:
            return None
        if self._resized_reference_frame is None:
            cropped = crop_frame(self._reference_frame, self.bbox)
            self._resized_reference_frame = resize_frame(
                cropped, self.aligned_h, self.aligned_w
            )
        return self._resized_reference_frame.copy()

    def materialize_pass_window(
        self,
        pass_plan: VideoEditPassPlan,
        spec: VideoEditWindowSpec,
        *,
        bridge_frames: Sequence[Image.Image] | None = None,
        previous_output_frames: Sequence[Image.Image] | None = None,
    ) -> VideoEditSequence:
        """Materialize one strict pass window without changing source indices."""

        if pass_plan.prefix_kind == "reference":
            if pass_plan.prefix_length != 1:
                raise ValueError(
                    "Reference conditioning prefix must contain exactly one frame, "
                    f"got {pass_plan.prefix_length}"
                )
            reference = self.get_resized_reference_frame()
            if reference is None:
                raise ValueError("Long VideoEdit pass requires an edited reference")
            prefix_frames = (reference,)
        else:
            if bridge_frames is None:
                raise ValueError("Short VideoEdit pass requires bridge frames")
            if len(bridge_frames) != pass_plan.prefix_length:
                raise ValueError(
                    "VideoEdit bridge length mismatch: "
                    f"expected {pass_plan.prefix_length}, got {len(bridge_frames)}"
                )
            prefix_frames = tuple(frame.copy() for frame in bridge_frames)

        expected_size = (self.aligned_w, self.aligned_h)
        if any(frame.size != expected_size for frame in prefix_frames):
            raise ValueError(
                "VideoEdit conditioning frames must match inference size "
                f"{expected_size}"
            )

        source_global_indices = []
        for pass_local_idx in spec.input_indices:
            if pass_local_idx >= pass_plan.prefix_length:
                source_position = pass_local_idx - pass_plan.prefix_length
                if source_position >= len(pass_plan.source_indices):
                    raise IndexError(
                        f"Pass-local index {pass_local_idx} out of range for "
                        f"{pass_plan.name} sequence"
                    )
                source_global_indices.append(
                    pass_plan.source_indices[source_position]
                )
        if source_global_indices:
            self._ensure_decoded_through(max(source_global_indices))

        frames: list[Image.Image] = []
        masks: list[Image.Image] = []
        for pass_local_idx in spec.input_indices:
            if pass_local_idx < pass_plan.prefix_length:
                frame = prefix_frames[pass_local_idx].copy()
                frames.append(frame)
                masks.append(Image.new("L", frame.size, 0))
                continue
            source_position = pass_local_idx - pass_plan.prefix_length
            global_idx = pass_plan.source_indices[source_position]
            entry = self._lookup_entry(global_idx)
            frames.append(entry.resized_frame.copy())
            masks.append(entry.resized_mask.copy())

        global_indices = tuple(
            pass_plan.sequence_indices[spec.start_index + local_idx]
            if spec.start_index + local_idx < len(pass_plan.sequence_indices)
            else None
            for local_idx in range(len(spec.input_indices))
        )
        raw_window = VideoEditSequence(
            frames=tuple(frames),
            masks=tuple(masks),
            global_indices=global_indices,
        )
        return apply_videoedit_overlap(
            raw_window, spec, previous_output_frames
        )

    def paste_back_frames(
        self,
        generated_frames: list[Image.Image],
        *,
        feather_px: int,
        adain_boundary_dilate: int,
    ) -> list[Image.Image]:
        resized_generated = [
            resize_frame(frame, self.crop_h, self.crop_w) for frame in generated_frames
        ]
        result: list[Image.Image] = []
        video_decoder = self._make_video_decoder()
        mask_decoder = self._make_mask_decoder()
        try:
            for global_index, generated_frame in enumerate(resized_generated):
                original_frame = video_decoder.read_next()
                raw_mask = mask_decoder.read_next()
                if original_frame is None or raw_mask is None:
                    raise RuntimeError(
                        "Unexpected end of input while paste-back streaming "
                        f"at global index {global_index}"
                    )
                expanded_mask = expand_mask_frame(
                    raw_mask,
                    dilate_px=self.dilate_px,
                    scale=self.mask_scale,
                    force_zero=False,
                )
                cropped_mask = crop_frame(expanded_mask, self.bbox)
                result.append(
                    paste_back_frame(
                        original_frame=original_frame,
                        generated_frame=generated_frame,
                        mask_frame=cropped_mask,
                        bbox=self.bbox,
                        feather_px=feather_px,
                        adain_boundary_dilate=adain_boundary_dilate,
                    )
                )
        finally:
            video_decoder.close()
            mask_decoder.close()
        return result

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=5.0)
            self._prefetch_thread = None
        self._video_decoder.close()
        self._mask_decoder.close()
        self._cache.clear()
