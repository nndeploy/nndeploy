"""Simple elementwise helpers for the minimal runtime."""

from __future__ import annotations

import torch
import torch.nn as nn


class MulAdd(nn.Module):
    """Compute ``c + a * (k + b)`` with light shape handling."""

    def __init__(self, prefix: str = ""):
        super().__init__()
        del prefix

    def forward(
        self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, k: int = 0
    ) -> torch.Tensor:
        if b.dim() == 4:
            num_frames = b.shape[1]
            frame_seqlen = a.shape[1] // num_frames
            return c + (
                a.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (k + b)
            ).flatten(1, 2)
        return c + a * (k + b)
