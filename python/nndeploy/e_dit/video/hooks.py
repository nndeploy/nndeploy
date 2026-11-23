"""Callbacks for raw/modified frame eviction and scene transitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class RuntimeHookRegistry:
    pop_raw_input_hooks: dict[str, Callable[[int, int], None]] = field(default_factory=dict)
    pop_modified_hooks: dict[str, Callable[[int, int], None]] = field(default_factory=dict)
    pop_scene_hooks: dict[
        str, Callable[[int, int, tuple[int, int] | None], None]
    ] = field(default_factory=dict)

    def register_pop_raw_input_hook(
        self,
        key: str,
        hook_func: Callable[[int, int], None],
    ) -> None:
        self.pop_raw_input_hooks[key] = hook_func

    def remove_pop_raw_input_hook(
        self,
        key: str,
    ) -> Callable[[int, int], None] | None:
        return self.pop_raw_input_hooks.pop(key, None)

    def register_pop_modified_hook(
        self,
        key: str,
        hook_func: Callable[[int, int], None],
    ) -> None:
        self.pop_modified_hooks[key] = hook_func

    def remove_pop_modified_hook(
        self,
        key: str,
    ) -> Callable[[int, int], None] | None:
        return self.pop_modified_hooks.pop(key, None)

    def register_pop_scene_hook(
        self,
        key: str,
        hook_func: Callable[[int, int, tuple[int, int] | None], None],
    ) -> None:
        self.pop_scene_hooks[key] = hook_func

    def remove_pop_scene_hook(
        self,
        key: str,
    ) -> Callable[[int, int, tuple[int, int] | None], None] | None:
        return self.pop_scene_hooks.pop(key, None)

    def emit_pop_raw_input(self, start_index: int, length: int) -> None:
        if length <= 0:
            return
        for hook in self.pop_raw_input_hooks.values():
            hook(int(start_index), int(length))

    def emit_pop_modified(self, start_index: int, length: int) -> None:
        if length <= 0:
            return
        for hook in self.pop_modified_hooks.values():
            hook(int(start_index), int(length))

    def emit_pop_scene(
        self,
        new_scene_index: int,
        scene_count: int,
        new_scene: tuple[int, int] | None,
    ) -> None:
        for hook in self.pop_scene_hooks.values():
            hook(int(new_scene_index), int(scene_count), new_scene)
