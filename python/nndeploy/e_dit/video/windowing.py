"""Windowing helpers for video processing."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nndeploy.e_dit.video.bbox import align_value
from nndeploy.e_dit.video.io import align_video, repeat_video_frames


@dataclass(frozen=True)
class WindowSpec:
    window_index: int
    scene_index: int
    scene_start: int
    scene_end: int
    load_start: int
    load_end: int
    deal_start: int
    deal_length: int
    load_length: int
    overlap_left: int
    overlap_right: int
    infer_len: int
    overlap: int
    time_sample: int
    time_shift: int

    @property
    def input_len(self) -> int:
        return self.load_end - self.load_start

    @property
    def future_length(self) -> int:
        return max(0, self.load_length - self.deal_length)

    @property
    def active_start_offset(self) -> int:
        return self.overlap_left

    @property
    def active_end_offset(self) -> int:
        return self.overlap_left + self.deal_length

    @property
    def start_index(self) -> int:
        return self.load_start

    @property
    def end_index(self) -> int:
        return self.load_end

    @property
    def commit_start(self) -> int:
        return self.deal_start

    @property
    def commit_end(self) -> int:
        return self.deal_start + self.deal_length


def align_spatial_size(height: int, width: int, align_h: int, align_w: int) -> tuple[int, int]:
    return align_value(height, align_h), align_value(width, align_w)


def pad_single_window(
    video: torch.Tensor,
    mask: torch.Tensor,
    infer_len: int,
    align_h: int,
    align_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    video = align_video(video, align_w=align_w, align_h=align_h, fill=0.0, padding_mode="replicate")
    mask = align_video(mask, align_w=align_w, align_h=align_h, fill=0.0, padding_mode="replicate")
    video = repeat_video_frames(video, infer_len, recycle=False, check_num=True)
    mask = repeat_video_frames(mask, infer_len, recycle=False, check_num=True)
    return video, mask


def infer_window_num_frames(
    current_num: int,
    expect_infer_len: int,
    time_sample: int,
    time_shift: int,
    use_dynamic: bool = False,
) -> int:
    if current_num <= 0:
        raise ValueError("current_num must be positive")
    if not use_dynamic:
        return expect_infer_len
    return min(
        expect_infer_len,
        time_shift + ((current_num - time_shift + time_sample - 1) // time_sample) * time_sample,
    )


def infer_latent_spatial_size(height: int, width: int, spatial_ratio: int) -> tuple[int, int]:
    return height // spatial_ratio, width // spatial_ratio


def infer_latent_frames(num_frames: int, temporal_ratio: int) -> int:
    return (num_frames - 1) // temporal_ratio + 1


def normalize_scene_specs(
    scenes: list[tuple[int, int]] | None,
    num_frames: int,
) -> list[tuple[int, int]]:
    if not scenes:
        return [(0, num_frames)]

    normalized: list[tuple[int, int]] = []
    for scene in scenes:
        if scene is None:
            continue
        if len(scene) != 2:
            raise ValueError(f"scene spec must be (start, length), got {scene}")
        start, length = int(scene[0]), int(scene[1])
        if length <= 0:
            continue
        if start < 0:
            start = 0
        if start >= num_frames:
            continue
        normalized.append((start, min(length, num_frames - start)))
    return normalized or [(0, num_frames)]


def build_window_specs(
    num_frames: int,
    infer_len: int = 121,
    overlap: int = 9,
    time_sample: int = 8,
    time_shift: int = 1,
    scenes: list[tuple[int, int]] | None = None,
) -> list[WindowSpec]:
    if infer_len <= 0:
        raise ValueError("infer_len must be positive")
    if overlap < 0 or overlap >= infer_len:
        raise ValueError("overlap must satisfy 0 <= overlap < infer_len")
    if time_sample <= 0:
        raise ValueError("time_sample must be positive")
    if time_shift <= 0:
        raise ValueError("time_shift must be positive")

    scene_specs = normalize_scene_specs(scenes, num_frames)
    specs: list[WindowSpec] = []
    window_index = 0
    for scene_index, (scene_start, scene_length) in enumerate(scene_specs):
        scene_end = min(num_frames, scene_start + scene_length)
        next_start_index = scene_start
        while next_start_index < scene_end:
            overlap_left = overlap if next_start_index > scene_start else 0
            need_new_frame = max(1, infer_len - overlap_left)
            deal_length = min(need_new_frame, scene_end - next_start_index)
            load_length = min(num_frames - next_start_index, need_new_frame)
            load_start = next_start_index - overlap_left
            load_end = next_start_index + load_length
            overlap_right = overlap if (next_start_index + deal_length) < scene_end else 0
            specs.append(
                WindowSpec(
                    window_index=window_index,
                    scene_index=scene_index,
                    scene_start=scene_start,
                    scene_end=scene_end,
                    load_start=load_start,
                    load_end=load_end,
                    deal_start=next_start_index,
                    deal_length=deal_length,
                    load_length=load_length,
                    overlap_left=overlap_left,
                    overlap_right=overlap_right,
                    infer_len=infer_len,
                    overlap=overlap,
                    time_sample=time_sample,
                    time_shift=time_shift,
                )
            )
            window_index += 1
            next_start_index += deal_length
    return specs
