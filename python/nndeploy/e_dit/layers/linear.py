"""Single-card linear layers for the minimal runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from nndeploy.e_dit.layers.utils import get_group_rank, get_group_size


def _tag_parameter(param: torch.nn.Parameter, *, input_dim: int = 1, output_dim: int = 0):
    param.input_dim = input_dim  # type: ignore[attr-defined]
    param.output_dim = output_dim  # type: ignore[attr-defined]
    param.weight_loader = _load_weight  # type: ignore[attr-defined]
    return param


def _load_weight(param: torch.nn.Parameter, loaded_weight: torch.Tensor) -> None:
    if loaded_weight.shape != param.shape:
        loaded_weight = loaded_weight.reshape(param.shape)
    param.data.copy_(loaded_weight)


class LinearBase(nn.Module, ABC):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        del quant_config, prefix
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.skip_bias_add = bool(skip_bias_add)
        self.params_dtype = params_dtype or torch.get_default_dtype()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__(
            input_size,
            output_size,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.linear = nn.Linear(input_size, output_size, bias=bias)
        self.linear = self.linear.to(dtype=self.params_dtype)
        self.weight = _tag_parameter(self.linear.weight)
        self.bias = _tag_parameter(self.linear.bias, input_dim=0, output_dim=0) if self.linear.bias is not None else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        out = self.linear(x)
        return out, self.bias if self.skip_bias_add else None


class ColumnParallelLinear(ReplicatedLinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config=None,
        output_sizes: list[int] | None = None,
        prefix: str = "",
        tp_group=None,
    ):
        del gather_output, output_sizes, tp_group
        super().__init__(
            input_size,
            output_size,
            bias=bias,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
        )


class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int] | int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config=None,
        prefix: str = "",
        tp_group=None,
    ):
        if isinstance(output_sizes, int):
            output_sizes = [output_sizes]
        self.output_sizes = list(output_sizes)
        super().__init__(
            input_size,
            sum(self.output_sizes),
            bias=bias,
            gather_output=gather_output,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            tp_group=tp_group,
        )


class QKVParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config=None,
        prefix: str = "",
        tp_group=None,
    ):
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.hidden_size = int(hidden_size)
        self.head_size = int(head_size)
        self.total_num_heads = int(total_num_heads)
        self.total_num_kv_heads = int(total_num_kv_heads)
        self.output_sizes = [
            self.total_num_heads * self.head_size,
            self.total_num_kv_heads * self.head_size,
            self.total_num_kv_heads * self.head_size,
        ]
        super().__init__(
            hidden_size,
            sum(self.output_sizes),
            bias=bias,
            gather_output=False,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            tp_group=tp_group,
        )


class RowParallelLinear(ReplicatedLinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        reduce_results: bool = True,
        quant_config=None,
        prefix: str = "",
        tp_group=None,
    ):
        del input_is_parallel, reduce_results, tp_group
        super().__init__(
            input_size,
            output_size,
            bias=bias,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
        )

