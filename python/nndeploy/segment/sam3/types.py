"""SAM3 workflow contracts. Coordinates use original-image pixels, xyxy boxes."""

from dataclasses import dataclass
import math


class SAM3Error(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ModelConfig:
    checkpoint: str
    device: str = "cuda"
    precision: str = "bfloat16"
    max_contexts: int = 8
    max_sessions: int = 4

    def __post_init__(self):
        if not self.checkpoint:
            raise SAM3Error("invalid_config", "An explicit local checkpoint is required")
        if self.precision not in ("float32", "bfloat16"):
            raise SAM3Error("invalid_config", "precision must be float32 or bfloat16")
        if self.max_contexts < 1 or self.max_sessions < 1:
            raise SAM3Error("invalid_config", "Resource limits must be positive")


def fields(value, allowed, required=()):
    if not isinstance(value, dict):
        raise SAM3Error("invalid_request", "Expected an object")
    extra, missing = set(value) - set(allowed), set(required) - set(value)
    if extra or missing:
        raise SAM3Error("invalid_request", f"Unknown fields: {sorted(extra)}; missing: {sorted(missing)}")


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SAM3Error("invalid_request", f"{name} must be an integer >= {minimum}")
    return value


def box_xyxy(value, width, height):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise SAM3Error("invalid_box", "bbox must contain four pixel coordinates")
    x0, y0, x1, y1 = map(float, value)
    if not all(math.isfinite(x) for x in (x0, y0, x1, y1)) or not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise SAM3Error("invalid_box", "bbox must be a nonempty xyxy box inside the image")
    return [x0, y0, x1, y1]


def points_xy(value, labels, width, height):
    if len(value) != len(labels) or not value:
        raise SAM3Error("invalid_points", "Nonempty points and matching labels are required")
    out = []
    for point, label in zip(value, labels):
        if len(point) != 2 or label not in (0, 1):
            raise SAM3Error("invalid_points", "Points require xy coordinates and binary labels")
        x, y = map(float, point)
        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
            raise SAM3Error("invalid_points", "Point must lie inside the original image")
        out.append([x, y])
    return out


def concept_boxes(values, width, height, video=False):
    result, labels = [], []
    for item in values:
        fields(item, {"box", "positive"}, {"box", "positive"})
        if not isinstance(item["positive"], bool):
            raise SAM3Error("invalid_box", "positive must be boolean")
        x0, y0, x1, y1 = box_xyxy(item["box"], width, height)
        result.append(([x0 / width, y0 / height] if video else [(x0 + x1) / (2 * width), (y0 + y1) / (2 * height)]) + [(x1 - x0) / width, (y1 - y0) / height])
        labels.append(int(item["positive"]))
    return result, labels
