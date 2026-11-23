"""Deterministic numerical profile.

Pins kernel selection and reduction order without changing sampling parameters.
Enabled by default; set ``NNDEPLOY_DIT_DETERMINISTIC=0`` to disable it.
Use the same profile for both configurations in a numerical comparison.
"""

from __future__ import annotations

from nndeploy.e_dit.utils.env import get_env

import os
from typing import Any

import torch

DETERMINISTIC = get_env("NNDEPLOY_DIT_DETERMINISTIC", "1") != "0"

# cuBLAS reads the workspace configuration when the CUDA context is created, so
# it has to be set before any CUDA work happens.
if DETERMINISTIC:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

_APPLIED = False


def enable_deterministic_mode() -> dict[str, Any]:
    """Apply the deterministic switches and return what actually took effect."""
    global _APPLIED
    settings: dict[str, Any] = {
        "enabled": DETERMINISTIC,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if not DETERMINISTIC:
        return settings
    if not _APPLIED:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        _APPLIED = True
    settings.update(
        use_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
    )
    return settings


__all__ = ["DETERMINISTIC", "enable_deterministic_mode"]
