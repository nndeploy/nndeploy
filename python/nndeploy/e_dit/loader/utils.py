"""Shared weight loading utilities."""

from __future__ import annotations

import contextlib
import glob
import os
import re
from collections import defaultdict
from collections.abc import Callable, Iterator
from typing import Any, Type

import torch
from torch import nn

from nndeploy.e_dit.config.server_args import ServerArgs
from nndeploy.e_dit.utils.logging_utils import init_logger

logger = init_logger(__name__)


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


class skip_init_modules:
    def __enter__(self):
        self._orig_reset = {}
        for cls in (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d):
            self._orig_reset[cls] = cls.reset_parameters
            cls.reset_parameters = lambda self: None

    def __exit__(self, exc_type, exc_value, traceback):
        for cls, orig in self._orig_reset.items():
            cls.reset_parameters = orig


def _normalize_component_type(module_type: str) -> str:
    if module_type.endswith("_2"):
        return module_type[:-2]
    return module_type


def _clean_hf_config_inplace(model_config: dict[str, Any]) -> None:
    for key in (
        "_name_or_path",
        "transformers_version",
        "model_type",
        "tokenizer_class",
        "torch_dtype",
    ):
        model_config.pop(key, None)


def _list_safetensors_files(model_path: str) -> list[str]:
    return sorted(glob.glob(os.path.join(str(model_path), "*.safetensors")))


def get_param_names_mapping(
    mapping_dict: dict[str, str | tuple[str, int, int]],
) -> Callable[[str], tuple[str, Any, Any]]:
    def mapping_fn(name: str) -> tuple[str, Any, Any]:
        merge_index = None
        total_split_params = None
        max_steps = max(8, len(mapping_dict) * 2)
        applied_patterns: set[str] = set()
        visited_names: set[str] = {name}

        for _ in range(max_steps):
            transformed = False
            for pattern, replacement in mapping_dict.items():
                if pattern in applied_patterns or re.match(pattern, name) is None:
                    continue
                curr_merge_index = None
                curr_total_split_params = None
                if isinstance(replacement, tuple):
                    curr_merge_index = replacement[1]
                    curr_total_split_params = replacement[2]
                    replacement = replacement[0]
                new_name = re.sub(pattern, replacement, name)
                if new_name != name:
                    if curr_merge_index is not None:
                        merge_index = curr_merge_index
                        total_split_params = curr_total_split_params
                    name = new_name
                    applied_patterns.add(pattern)
                    if name in visited_names:
                        transformed = False
                        break
                    visited_names.add(name)
                    transformed = True
                    break
            if not transformed:
                break
        return name, merge_index, total_split_params

    return mapping_fn


def hf_to_custom_state_dict(
    hf_param_sd: dict[str, torch.Tensor] | Iterator[tuple[str, torch.Tensor]],
    param_names_mapping: Callable[[str], tuple[str, Any, Any]],
) -> tuple[dict[str, torch.Tensor], dict[str, tuple[str, Any, Any]]]:
    custom_param_sd = {}
    to_merge_params = defaultdict(dict)
    reverse_param_names_mapping = {}
    if isinstance(hf_param_sd, dict):
        hf_param_sd = hf_param_sd.items()
    for source_param_name, full_tensor in hf_param_sd:
        target_param_name, merge_index, num_params_to_merge = param_names_mapping(
            source_param_name
        )
        if not target_param_name:
            continue
        reverse_param_names_mapping[target_param_name] = (
            source_param_name,
            merge_index,
            num_params_to_merge,
        )
        if merge_index is not None:
            to_merge_params[target_param_name][merge_index] = full_tensor
            if len(to_merge_params[target_param_name]) == num_params_to_merge:
                sorted_tensors = [
                    to_merge_params[target_param_name][i]
                    for i in range(num_params_to_merge)
                ]
                full_tensor = torch.cat(sorted_tensors, dim=0)
                del to_merge_params[target_param_name]
            else:
                continue
        custom_param_sd[target_param_name] = full_tensor
    return custom_param_sd, reverse_param_names_mapping


BYTES_PER_GB = 1024**3


def get_memory_usage_of_component(module: Any) -> float | None:
    if not isinstance(module, nn.Module):
        return None
    if hasattr(module, "get_memory_footprint"):
        usage = module.get_memory_footprint() / BYTES_PER_GB
    else:
        param_size = sum(p.numel() * p.element_size() for p in module.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in module.buffers())
        usage = (param_size + buffer_size) / BYTES_PER_GB
    return round(usage, 2)


component_name_to_loader_cls: dict[str, Type[Any]] = {}


def resolve_component_torch_dtype(
    server_args: ServerArgs,
    component_name: str,
) -> torch.dtype | None:
    """Resolve the requested dtype for a component."""
    if component_name in {"tokenizer", "scheduler"}:
        return None
    return server_args.resolve_component_dtype(component_name)


def move_module_to_device_and_dtype(
    module: nn.Module,
    device: torch.device,
    dtype: torch.dtype | None,
) -> nn.Module:
    if dtype is not None:
        module = module.to(dtype=dtype)
    return module.to(device)


def log_loading_info(
    component_name: str,
    loading_info: dict[str, Any] | None,
) -> None:
    if not loading_info:
        return

    missing = list(loading_info.get("missing_keys") or [])
    unexpected = list(loading_info.get("unexpected_keys") or [])
    mismatched = list(loading_info.get("mismatched_keys") or [])
    errors = list(loading_info.get("error_msgs") or [])

    if missing:
        logger.warning("%s missing keys: %s", component_name, missing[:20])
    if unexpected:
        logger.warning("%s unexpected keys: %s", component_name, unexpected[:20])
    if mismatched:
        logger.warning("%s mismatched keys: %s", component_name, mismatched[:20])
    if errors:
        raise RuntimeError(f"{component_name} load errors: {' | '.join(errors[:5])}")
