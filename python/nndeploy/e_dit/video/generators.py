"""Per-window random generator construction."""

from __future__ import annotations

import torch


def build_window_generator(
    *,
    seed: int | None,
    request_generator: torch.Generator | None,
) -> torch.Generator | None:
    """Build a fresh per-window generator from the request seed and device."""
    if seed is None:
        return None
    if not isinstance(request_generator, torch.Generator):
        raise ValueError(
            "seed is set but request_generator must be a torch.Generator "
            "to provide the target device"
        )
    return torch.Generator(device=request_generator.device).manual_seed(int(seed))
