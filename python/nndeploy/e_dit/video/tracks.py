"""Bounding-box tracks from arrays, tensors, JSON and CSV; XYXY internal layout."""
from __future__ import annotations
import ast
import csv
import json
from pathlib import Path
from typing import Any
import torch


def _xywh_to_xyxy(box: list[int] | tuple[int, int, int, int]) -> list[int]:
    x, y, w, h = (int(v) for v in box)
    return [x, y, x + w, y + h]


def _finalize_bbox_track(track: torch.Tensor, num_frames: int) -> torch.Tensor:
    if track.ndim != 2 or track.shape[-1] != 4:
        raise ValueError(f"bbox track must have shape [F, 4], got {tuple(track.shape)}")
    track = track.detach().cpu().to(torch.int64)
    if track.shape[0] == 0:
        return torch.zeros(num_frames, 4, dtype=torch.int64)
    if track.shape[0] < num_frames:
        tail = track[-1:, ...].repeat(num_frames - track.shape[0], 1)
        track = torch.cat([track, tail], dim=0)
    return track[:num_frames]


def _load_bbox_tracks_from_csv(path: str, num_frames: int) -> list[torch.Tensor]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows: list[list[int]] = []
        multi_rows: list[list[list[int]]] = []
        object_count: int | None = None
        for row in reader:
            if "bboxes" in fieldnames and row.get("bboxes"):
                parsed = ast.literal_eval(row["bboxes"])
                if not isinstance(parsed, (list, tuple)):
                    raise ValueError(f"Invalid bbox entry in {path}: {parsed}")
                if object_count is None:
                    object_count = len(parsed)
                elif len(parsed) != object_count:
                    raise ValueError(
                        f"Inconsistent object count in {path}: expected {object_count}, got {len(parsed)}"
                    )
                frame_boxes: list[list[int]] = []
                for box in parsed:
                    if len(box) != 4:
                        raise ValueError(f"Invalid bbox entry in {path}: {box}")
                    frame_boxes.append([int(v) for v in box])
                multi_rows.append(frame_boxes)
                continue
            if {"x1", "y1", "w", "h"}.issubset(fieldnames):
                rows.append(
                    _xywh_to_xyxy(
                        [
                            int(row["x1"]),
                            int(row["y1"]),
                            int(row["w"]),
                            int(row["h"]),
                        ]
                    )
                )
                continue
            if {"x", "y", "w", "h"}.issubset(fieldnames):
                rows.append(
                    _xywh_to_xyxy(
                        [
                            int(row["x"]),
                            int(row["y"]),
                            int(row["w"]),
                            int(row["h"]),
                        ]
                    )
                )
                continue
            raise ValueError(
                f"Unsupported bbox csv schema in {path}. Expected `bboxes` or x1/y1/w/h columns."
            )
    if multi_rows:
        bbox_tensor = torch.tensor(multi_rows, dtype=torch.int64).movedim(0, 1)
        return [_finalize_bbox_track(track, num_frames) for track in bbox_tensor]
    if not rows:
        return [torch.zeros(num_frames, 4, dtype=torch.int64)]
    return [_finalize_bbox_track(torch.tensor(rows, dtype=torch.int64), num_frames)]


def load_bbox_tracks(bbox_value: Any, num_frames: int) -> list[torch.Tensor] | None:
    if bbox_value is None:
        return None
    if isinstance(bbox_value, str):
        path = Path(bbox_value)
        if not path.exists():
            raise FileNotFoundError(f"bbox path does not exist: {bbox_value}")
        if path.suffix.lower() == ".csv":
            return _load_bbox_tracks_from_csv(str(path), num_frames)
        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            bbox_frames = torch.tensor(payload, dtype=torch.int64)
        else:
            raise ValueError(f"Unsupported bbox path format: {bbox_value}")
    elif isinstance(bbox_value, torch.Tensor):
        bbox_frames = bbox_value.detach().cpu().to(torch.int64)
    else:
        bbox_frames = torch.as_tensor(bbox_value, dtype=torch.int64)

    if bbox_frames.ndim == 1 and bbox_frames.numel() == 4:
        return [_finalize_bbox_track(bbox_frames.view(1, 4), num_frames)]
    if bbox_frames.ndim == 2 and bbox_frames.shape[-1] == 4:
        return [_finalize_bbox_track(bbox_frames, num_frames)]
    if bbox_frames.ndim == 3 and bbox_frames.shape[-1] == 4:
        if bbox_frames.shape[0] == num_frames and bbox_frames.shape[1] != num_frames:
            bbox_frames = bbox_frames.movedim(0, 1)
        return [_finalize_bbox_track(track, num_frames) for track in bbox_frames]
    raise ValueError(
        f"Unsupported bbox tensor shape {tuple(bbox_frames.shape)}; expected [..., 4]"
    )

