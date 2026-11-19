"""Minimal MLP blocks."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from diffusers.models.activations import (
    ApproximateGELU,
    GEGLU,
    GELU,
    LinearActivation,
    SwiGLU,
)

from nndeploy.e_dit.layers.activation import get_act_fn
from nndeploy.e_dit.layers.linear import ColumnParallelLinear, RowParallelLinear


class MLP(nn.Module):
    """A small feed-forward block with the legacy interface."""

    def __init__(
        self,
        input_dim: int,
        mlp_hidden_dim: int,
        output_dim: int | None = None,
        bias: bool = True,
        act_type: str = "gelu_pytorch_tanh",
        dtype: torch.dtype | None = None,
        prefix: str = "",
        quant_config=None,
        fc_in_quant_prefix=None,
        fc_out_quant_prefix=None,
    ):
        super().__init__()
        del prefix, quant_config
        self.fc_in = ColumnParallelLinear(
            input_dim,
            mlp_hidden_dim,
            bias=bias,
            params_dtype=dtype,
        )
        self.act = get_act_fn(act_type)
        if output_dim is None:
            output_dim = input_dim
        self.fc_out = RowParallelLinear(
            mlp_hidden_dim,
            output_dim,
            bias=bias,
            params_dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc_in(x)
        x = self.act(x)
        x, _ = self.fc_out(x)
        return x


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        activation_fn: str = "geglu",
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        elif activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)
        elif activation_fn == "swiglu":
            act_fn = SwiGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "linear-silu":
            act_fn = LinearActivation(dim, inner_dim, bias=bias, activation="silu")
        else:
            raise ValueError(f"Unsupported activation_fn: {activation_fn}")

        self.net = nn.ModuleList([act_fn, nn.Dropout(0.0), nn.Linear(inner_dim, dim_out, bias=bias)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states
