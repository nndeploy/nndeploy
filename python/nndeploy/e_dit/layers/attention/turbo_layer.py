"""Compatibility shim for attention helper exports."""

from __future__ import annotations

import torch
import torch.nn as nn


class MinimalA2AAttnOp(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
