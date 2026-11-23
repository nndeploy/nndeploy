"""Custom activation functions for the minimal runtime."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return F.silu(x[..., :half]) * x[..., half:]


class GeluAndMul(nn.Module):
    def __init__(self, approximate: str = "none"):
        super().__init__()
        if approximate not in ("none", "tanh"):
            raise ValueError(f"Unknown approximate mode: {approximate}")
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return F.gelu(x[..., :half], approximate=self.approximate) * x[..., half:]


class NewGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * torch.pow(x, 3.0))))


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


_ACTIVATION_REGISTRY = {
    "gelu": lambda: nn.GELU(),
    "gelu_new": NewGELU,
    "gelu_pytorch_tanh": lambda: nn.GELU(approximate="tanh"),
    "relu": lambda: nn.ReLU(),
    "silu": lambda: nn.SiLU(),
    "quick_gelu": QuickGELU,
}


def get_act_fn(act_fn_name: str) -> nn.Module:
    act_fn_name = act_fn_name.lower()
    if act_fn_name not in _ACTIVATION_REGISTRY:
        raise ValueError(f"Activation function {act_fn_name!r} is not supported.")
    return _ACTIVATION_REGISTRY[act_fn_name]()


_ACTIVATION_AND_MUL_REGISTRY = {
    "gelu": GeluAndMul,
    "silu": SiluAndMul,
}


def get_act_and_mul_fn(act_fn_name: str) -> nn.Module:
    act_fn_name = act_fn_name.lower()
    if act_fn_name not in _ACTIVATION_AND_MUL_REGISTRY:
        raise ValueError(f"Activation function {act_fn_name!r} is not supported.")
    return _ACTIVATION_AND_MUL_REGISTRY[act_fn_name]()
