"""Lightweight runtime progress helpers for diffusion pipelines."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


@dataclass
class RuntimeProgressState:
    progress: Progress
    pipeline_task_id: TaskID | None = None
    denoise_task_id: TaskID | None = None
    lock: threading.Lock | None = None

    def stop(self) -> None:
        self.progress.stop()

    def add_pipeline_task(self, total: int) -> None:
        with self.lock or threading.Lock():
            if self.pipeline_task_id is None:
                self.pipeline_task_id = self.progress.add_task(
                    "Erase Pipeline",
                    total=max(int(total), 1),
                    completed=0,
                    meta="waiting",
                )

    def update_pipeline(
        self,
        completed: int,
        total: int,
        *,
        object_index: int,
        object_count: int,
        window_index: int,
        window_count: int,
    ) -> None:
        with self.lock or threading.Lock():
            if self.pipeline_task_id is None:
                self.add_pipeline_task(total)
            assert self.pipeline_task_id is not None
            self.progress.update(
                self.pipeline_task_id,
                total=max(int(total), 1),
                completed=max(0, min(int(completed), max(int(total), 1))),
                meta=(
                    f"object {object_index + 1}/{object_count} "
                    f"window {window_index + 1}/{window_count}"
                ),
            )

    def reset_denoise_task(
        self,
        *,
        object_index: int,
        object_count: int,
        window_index: int,
        window_count: int,
        total_steps: int,
    ) -> None:
        with self.lock or threading.Lock():
            description = (
                f"Denoise O{object_index + 1}/{object_count} "
                f"W{window_index + 1}/{window_count}"
            )
            if self.denoise_task_id is None:
                self.denoise_task_id = self.progress.add_task(
                    description,
                    total=max(int(total_steps), 1),
                    completed=0,
                    meta="step 0",
                )
            else:
                self.progress.update(
                    self.denoise_task_id,
                    description=description,
                    total=max(int(total_steps), 1),
                    completed=0,
                    meta="step 0",
                    visible=True,
                )

    def update_denoise(
        self,
        step_index: int,
        total_steps: int,
        timestep_value: float | None,
        cfg_enabled: bool | None = None,
        guidance_scale: float | None = None,
    ) -> None:
        with self.lock or threading.Lock():
            if self.denoise_task_id is None:
                return
            meta = f"step {step_index + 1}"
            if timestep_value is not None:
                meta += f" t={timestep_value:.2f}"
            if cfg_enabled is not None:
                meta += f" cfg={'on' if cfg_enabled else 'off'}"
            if guidance_scale is not None:
                meta += f" gs={guidance_scale:.2f}"
            self.progress.update(
                self.denoise_task_id,
                total=max(int(total_steps), 1),
                completed=max(0, min(int(step_index) + 1, max(int(total_steps), 1))),
                meta=meta,
            )

    def hide_denoise(self) -> None:
        with self.lock or threading.Lock():
            if self.denoise_task_id is None:
                return
            self.progress.update(self.denoise_task_id, visible=False)

    def set_pipeline_meta(self, meta: str) -> None:
        with self.lock or threading.Lock():
            if self.pipeline_task_id is None:
                return
            self.progress.update(self.pipeline_task_id, meta=meta)


def create_runtime_progress() -> RuntimeProgressState:
    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("|"),
        TimeRemainingColumn(),
        TextColumn("{task.fields[meta]}"),
        console=Console(),
        transient=False,
        refresh_per_second=10,
    )
    progress.start()
    return RuntimeProgressState(
        progress=progress,
        lock=threading.Lock(),
    )
