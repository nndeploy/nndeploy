"""Unmerged LoRA adapters with atomic activation and reversible module hooks."""
from pathlib import Path
import re
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from nndeploy.e_dit.loader.weight_mapping import map_weight_name

class LoRAManager:
    def __init__(self, model, *, name_mapping=None):
        self.model = model
        self.name_mapping = name_mapping or {}
        self.adapters = {}
        self.active = ()
        self._handles = []

    def load(self, name, path):
        if name in self.adapters:
            raise ValueError(f'Duplicate adapter: {name}')
        weights = load_file(str(Path(path)), device='cpu')
        pairs, alphas = {}, {}
        for key, tensor in weights.items():
            key = re.sub(r'^(base_model\.model\.|diffusion_model\.|transformer\.)', '', key)
            if key.endswith('.alpha'):
                alphas[map_weight_name(key[:-6] + '.weight', self.name_mapping)[:-7]] = float(tensor.item())
                continue
            match = re.match(r'^(.*)\.(lora_A|lora_B|lora_down|lora_up)(?:\.default)?\.weight$', key)
            if match is None:
                raise ValueError(f'Unsupported LoRA weight: {key}')
            target, side = match.groups()
            target = map_weight_name(target + '.weight', self.name_mapping)[:-7]
            pairs.setdefault(target, {})['A' if side in ('lora_A','lora_down') else 'B'] = tensor
        if not pairs:
            raise ValueError('Empty LoRA checkpoint')
        validated = {}
        for target, pair in pairs.items():
            module = self.model.get_submodule(target)
            if set(pair) != {'A','B'}:
                raise ValueError(f'Incomplete LoRA pair: {target}')
            a, b = pair['A'], pair['B']
            if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1] or (b.shape[0], a.shape[1]) != tuple(module.weight.shape):
                raise ValueError(f'LoRA shape mismatch: {target}')
            validated[target] = (a, b, alphas.get(target, a.shape[0]) / a.shape[0])
        self.adapters[name] = validated

    def activate(self, adapters):
        import math
        adapters = tuple((name, float(scale)) for name, scale in adapters)
        if any(not math.isfinite(scale) for _, scale in adapters):
            raise ValueError('LoRA strengths must be finite')
        targets = {}
        for name, scale in adapters:
            if name not in self.adapters:
                raise KeyError(name)
            for target, values in self.adapters[name].items():
                targets.setdefault(target, []).append((*values, scale))
        handles = []
        try:
            for target, entries in targets.items():
                module = self.model.get_submodule(target)
                def apply(module, inputs, output, entries=entries):
                    value = output[0] if isinstance(output, tuple) else output
                    x = inputs[0]
                    for a, b, alpha, scale in entries:
                        delta = F.linear(F.linear(x, a.to(x)), b.to(x))
                        value = value + delta.to(value) * (alpha * scale)
                    return (value, *output[1:]) if isinstance(output, tuple) else value
                handles.append(module.register_forward_hook(apply))
        except BaseException:
            for handle in handles: handle.remove()
            raise
        for handle in self._handles: handle.remove()
        self._handles = handles
        self.active = adapters

    def close(self):
        for handle in self._handles: handle.remove()
        self._handles.clear()
        self.adapters.clear()
        self.active = ()
        self.model = None
