"""Device, dtype and pinned-memory operations for tensors and modules."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from nndeploy.e_dit.utils.platform import get_local_torch_device


def module_device(module: nn.Module) -> torch.device:
    execution_device = getattr(module, "_nndeploy_execution_device", None)
    if execution_device is not None:
        return torch.device(execution_device)
    first_param = next(module.parameters(), None)
    if first_param is None:
        return get_local_torch_device("cpu")
    return first_param.device


def module_dtype(module: nn.Module) -> torch.dtype | None:
    first_param = next(module.parameters(), None)
    if first_param is None:
        return None
    return first_param.dtype


def pin_module_cpu_memory(module: nn.Module) -> bool:
    changed = False
    for parameter in module.parameters():
        if parameter.device.type == "cpu" and not parameter.is_pinned():
            parameter.data = parameter.data.pin_memory()
            changed = True
    for buffer in module.buffers():
        if buffer.device.type == "cpu" and not buffer.is_pinned():
            buffer.data = buffer.data.pin_memory()
            changed = True
    return changed


def maybe_pin_tensor(tensor: torch.Tensor, enable: bool) -> torch.Tensor:
    if not enable or tensor.device.type != "cpu" or tensor.is_pinned():
        return tensor
    return tensor.pin_memory()


def move_module_to_device(
    module: nn.Module,
    target_device: torch.device,
    *,
    dtype: torch.dtype | None = None,
) -> bool:
    current_device = module_device(module)
    if current_device == target_device and (dtype is None or module_dtype(module) == dtype):
        return False
    move_kwargs: dict[str, Any] = {"device": target_device}
    if dtype is not None:
        move_kwargs["dtype"] = dtype
    if current_device.type == "cpu" or target_device.type == "cpu":
        move_kwargs["non_blocking"] = True
    module.to(**move_kwargs)
    return True
