"""SGLang-style meta construction and safetensors materialization for local models.

Adapted from runtime/loader/fsdp_load.py and weight_utils.py at SGLang
cdd427a588037dc8a8eb860ac17654bb2e55e752 (Apache-2.0).
Checkpoint names must match the constructed model; no external model registry
or HF-to-custom key mapping is needed. Shared parameters are assigned together.
"""
from pathlib import Path

import torch
from safetensors import safe_open


def load_safetensors_model(factory, path, *, dtype, device='cpu', key_mapping=None):
    with torch.device('meta'):
        model = factory()
    if dtype is not None:
        model.to(dtype=dtype)
    tensors = dict(model.named_parameters(remove_duplicate=False))
    buffers = dict(model.named_buffers(remove_duplicate=False))
    tensors.update(buffers)
    aliases = {}
    for name, tensor in tensors.items():
        aliases.setdefault(id(tensor), []).append(name)
    files = [Path(path)] if Path(path).is_file() else sorted(Path(path).glob('*.safetensors'))
    if not files:
        raise ValueError(f'No safetensors weights in {path}')
    seen, loaded = set(), set()
    for file in files:
        with safe_open(file, framework='pt', device='cpu') as handle:
            for name in handle.keys():
                checkpoint_name = name
                name = key_mapping(name) if key_mapping is not None else name
                if name in seen or (key_mapping is not None and name in loaded):
                    raise ValueError(f'Duplicate checkpoint weight: {name}')
                seen.add(name)
                if name not in tensors:
                    raise ValueError(f'Unexpected checkpoint weight: {name}')
                target = tensors[name]
                value = handle.get_tensor(checkpoint_name)
                if value.shape != target.shape:
                    raise ValueError(f'Weight shape mismatch: {name}: {value.shape} != {target.shape}')
                # Own the storage rather than holding the entire mmap checkpoint.
                value = value.to(device=device, dtype=target.dtype, copy=True)
                is_parameter = isinstance(target, torch.nn.Parameter)
                value = torch.nn.Parameter(value, requires_grad=False) if is_parameter else value
                for alias in aliases[id(target)]:
                    parent, _, leaf = alias.rpartition('.')
                    module = model.get_submodule(parent) if parent else model
                    if is_parameter:
                        module._parameters[leaf] = value
                    else:
                        module._buffers[leaf] = value
                    loaded.add(alias)
    missing = set(tensors) - loaded
    if missing:
        raise ValueError(f'Missing checkpoint weights: {sorted(missing)}')
    if any(t.is_meta for t in list(model.parameters()) + list(model.buffers())):
        raise ValueError('Unmaterialized tensors after checkpoint load')
    return model.eval(), {'missing_keys': [], 'unexpected_keys': []}
