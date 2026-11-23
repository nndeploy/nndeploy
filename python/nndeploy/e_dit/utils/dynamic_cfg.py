"""Dynamic classifier-free guidance helpers."""

from __future__ import annotations

import math
from typing import Tuple

import torch


def calc_current_cfg(
    max_cfg: float,
    current_step: int,
    max_step: int = 15,
    min_cfg: float = 1.0,
    dynamic_cfg: bool = True,
    do_space: bool = False,
    guss_tensor: torch.Tensor | None = None,
) -> Tuple[bool, float] | Tuple[bool, torch.Tensor]:
    if dynamic_cfg:
        if current_step < max_step:
            add_cfg = max(
                (max_cfg - min_cfg)
                / (
                    math.pow(
                        (max_cfg - min_cfg),
                        1.0 / max_step,
                    )
                    ** current_step
                ),
                0,
            )
        else:
            add_cfg = 0

        do_cfg = (min_cfg + add_cfg) > 1.0

        if do_space:
            assert guss_tensor is not None, "guss_tensor is required while do_space==True"
            guidance_scale = (min_cfg + add_cfg * guss_tensor).to(
                device=guss_tensor.device, dtype=guss_tensor.dtype
            )
        else:
            guidance_scale = min_cfg + add_cfg
    else:
        do_cfg = max_cfg > 1.0
        guidance_scale = max_cfg

    return do_cfg, guidance_scale
