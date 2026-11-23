"""Shared checkpoint weight utilities."""

from __future__ import annotations

import json
import os
from collections.abc import Generator

import torch
from safetensors.torch import safe_open

from nndeploy.e_dit.utils.platform import get_local_torch_device


def filter_duplicate_safetensors_files(
    hf_weights_files: list[str],
    hf_folder: str,
    index_file: str,
) -> list[str]:
    index_file_name = os.path.join(hf_folder, index_file)
    if not os.path.isfile(index_file_name):
        return hf_weights_files
    with open(index_file_name, 'r', encoding='utf-8') as fh:
        weight_map = json.load(fh).get('weight_map', {})
    weight_files_in_index = {
        os.path.join(hf_folder, file_name) for file_name in weight_map.values()
    }
    return [f for f in hf_weights_files if f in weight_files_in_index]


def filter_files_not_needed_for_inference(hf_weights_files: list[str]) -> list[str]:
    blacklist = {
        'training_args.bin',
        'optimizer.bin',
        'optimizer.pt',
        'scheduler.pt',
        'scaler.pt',
    }
    return [f for f in hf_weights_files if os.path.basename(f) not in blacklist]


def safetensors_weights_iterator(
    hf_weights_files: list[str],
    to_cpu: bool = True,
    use_runai_model_streamer: bool | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    del use_runai_model_streamer
    device = 'cpu' if to_cpu else str(get_local_torch_device())
    for st_file in hf_weights_files:
        with safe_open(st_file, framework='pt', device=device) as handle:
            for name in handle.keys():
                yield name, handle.get_tensor(name)


def _load_pt_file(bin_file: str, device: str) -> dict:
    return torch.load(bin_file, map_location=device, weights_only=False)


def pt_weights_iterator(
    hf_weights_files: list[str],
    to_cpu: bool = True,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    device = 'cpu' if to_cpu else str(get_local_torch_device())
    for bin_file in hf_weights_files:
        state = _load_pt_file(bin_file, device)
        yield from state.items()
        del state


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    if param.numel() == 1 and loaded_weight.numel() == 1:
        param.data.fill_(loaded_weight.item())
        return
    if param.size() != loaded_weight.size():
        raise ValueError(
            f'Attempted to load weight {tuple(loaded_weight.size())} into '
            f'parameter {tuple(param.size())}'
        )
    param.data.copy_(loaded_weight)
