"""Minimal platform helpers used by diffusion inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def get_local_torch_device(device: str | None = None) -> torch.device:
    """Resolve the runtime device with safe CPU fallback."""
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class _CurrentPlatform:
    """A tiny platform facade replacing the old sglang platform object."""

    @property
    def device_type(self) -> str:
        return get_local_torch_device().type

    def is_mps(self) -> bool:
        return self.device_type == "mps"

    def get_available_gpu_memory(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        free_bytes, _ = torch.cuda.mem_get_info()
        return round(free_bytes / (1024**3), 2)

    def optimize_vae(self, module):
        return module

    def verify_model_arch(self, _model_arch: str) -> None:
        return None


current_platform = _CurrentPlatform()

