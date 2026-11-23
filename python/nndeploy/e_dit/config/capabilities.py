"""Serializable algorithm capabilities; importing this module needs no ML backend."""
from dataclasses import dataclass, asdict
import os


class CapabilityError(ValueError):
    def __init__(self, algorithm, parameter, value, reason):
        self.algorithm, self.parameter, self.value = algorithm, parameter, value
        self.reason = reason
        super().__init__(f'{algorithm}: {parameter}={value!r}: {reason}')


@dataclass(frozen=True)
class AlgorithmCapabilities:
    tasks: tuple[str, ...] = ()
    devices: tuple[str, ...] = ()
    attention_backends: tuple[str, ...] = ()
    features: tuple[str, ...] = ()
    single_device: bool = False
    # Unknown capabilities are delegated to the algorithm's existing validator.
    enforced: bool = False

    def to_dict(self):
        return asdict(self)

    def validate(self, config):
        if not self.enforced:
            return
        name = config.algorithm
        def reject(key, value, reason):
            raise CapabilityError(name, key, value, reason)
        device = config.device.split(':')[0]
        if device not in self.devices:
            reject('device', config.device, f'supported devices: {self.devices}')
        if config.attention_backend not in self.attention_backends:
            reject('attention_backend', config.attention_backend,
                   f'supported backends: {self.attention_backends}')
        flags = {
            'lora_path': 'lora', 'dit_cpu_offload': 'component_offload',
            'vae_cpu_offload': 'component_offload',
            'text_encoder_cpu_offload': 'component_offload',
            'image_encoder_cpu_offload': 'component_offload',
            'dit_layerwise_offload': 'layerwise_offload',
            'use_fsdp_inference': 'fsdp', 'enable_torch_compile': 'compile',
            'compile_components': 'compile', 'warmup': 'warmup',
        }
        for key, feature in flags.items():
            value = getattr(config, key, None)
            if value and feature not in self.features:
                reject(key, value, f'{feature} is not enabled for this algorithm')
        for key, disabled, feature in (
            ('transformer_quantization', 'none', 'quantization'),
            ('operator_fusion_backend', 'disabled', 'fusion'),
        ):
            value = getattr(config, key, disabled)
            if value != disabled and feature not in self.features:
                reject(key, value, f'{feature} is not enabled for this algorithm')
        if self.single_device:
            for key in ('sp_degree', 'cfg_parallel_degree', 'vae_parallel_degree'):
                value = getattr(config, key, 0)
                if type(value) is not int or value not in (0, 1):
                    reject(key, value, 'requires single-device execution (0 or 1)')
            if config.parallel_mode != 'disabled':
                reject('parallel_mode', config.parallel_mode, 'requires single-device execution')
            if int(os.environ.get('WORLD_SIZE', '1')) != 1:
                reject('WORLD_SIZE', os.environ['WORLD_SIZE'], 'requires one runtime rank')
