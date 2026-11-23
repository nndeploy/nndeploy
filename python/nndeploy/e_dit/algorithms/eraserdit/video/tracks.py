"""BBox/object track helpers for pipelines.runtime."""

from __future__ import annotations

from typing import Any

from nndeploy.e_dit.video.tracks import load_bbox_tracks as _load_bbox_tracks



def _is_scene_tuple(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, (int, float)) for item in value)
    )


def _resolve_object_value(value: Any, object_index: int, object_count: int) -> Any:
    if not isinstance(value, (list, tuple)):
        return value
    if not value:
        return None
    has_nested_sequences = any(isinstance(item, (list, tuple)) for item in value)
    if has_nested_sequences and not all(_is_scene_tuple(item) for item in value):
        return value[min(object_index, len(value) - 1)]
    if object_count > 1 and len(value) == object_count:
        return value[min(object_index, len(value) - 1)]
    return value


def _resolve_object_scenes(
    scenes: list[tuple[int, int]] | None,
    object_index: int,
    object_count: int,
) -> list[tuple[int, int]] | None:
    if scenes is None:
        return None
    if not isinstance(scenes, (list, tuple)) or not scenes:
        return None
    if all(_is_scene_tuple(item) for item in scenes):
        return [tuple(int(v) for v in scene) for scene in scenes]
    object_scenes = scenes[min(object_index, len(scenes) - 1)]
    if object_scenes is None:
        return None
    if _is_scene_tuple(object_scenes):
        return [tuple(int(v) for v in object_scenes)]
    return [tuple(int(v) for v in scene) for scene in object_scenes]


def _select_sequence_item(value: Any, index: int) -> Any:
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return value[min(index, len(value) - 1)]
    return value
