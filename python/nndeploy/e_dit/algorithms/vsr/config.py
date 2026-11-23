"""Strict instance and request settings for single-step VSR."""
from dataclasses import dataclass, asdict
import math

DTYPES = {'bf16': 'bfloat16', 'fp16': 'float16', 'fp32': 'float32',
          'bfloat16': 'bfloat16', 'float16': 'float16', 'float32': 'float32'}


@dataclass
class VSRConfig:
    checkpoint_dir: str = ''
    wan_root: str = ''
    device: str = 'cuda'
    weight_dtype: str = 'bf16'
    tile_devices: list[str] | None = None
    vae_cpu_offload: bool = False
    dit_cpu_offload: bool = False
    dit_layerwise_offload: bool = False
    dit_offload_prefetch_size: float = 0.0
    pin_cpu_memory: bool = True
    cudnn_benchmark: bool = False
    channels_last_3d: bool = False
    compile_decoder: bool = False
    compile_encoder: bool = False
    decoder_implicit_padding: bool = False
    cache_dit_condition: bool = False
    tile_t: int = 33
    tile_h: int = 320
    tile_w: int = 640
    temporal_overlap: int = 5
    spatial_overlap: int = 32
    color_ref: str = 'global'
    color_ref_samples: int = 64
    gpu_postprocess: bool = False
    read_queue: int = 2
    write_queue: int = 4
    crf: int = 5
    target_resolution: str | None = None
    long_edge: int = 3840

    def __post_init__(self):
        if self.weight_dtype not in DTYPES:
            raise ValueError('Unsupported VSR weight_dtype')
        if self.dit_cpu_offload and self.dit_layerwise_offload:
            raise ValueError('DiT whole-module and layerwise offload are mutually exclusive')
        if not math.isfinite(self.dit_offload_prefetch_size) or self.dit_offload_prefetch_size < 0:
            raise ValueError('dit_offload_prefetch_size must be finite and non-negative')
        self.request({})

    def request(self, overrides):
        allowed = {'target_resolution', 'long_edge', 'tile_t', 'tile_h', 'tile_w',
                   'temporal_overlap', 'spatial_overlap', 'color_ref', 'color_ref_samples',
                   'crf', 'read_queue', 'write_queue', 'gpu_postprocess',
                   'preserve_audio', 'save_tiles_dir'}
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError(f'Unknown VSR request parameters: {sorted(unknown)}')
        values = {key: value for key, value in asdict(self).items() if key in allowed}
        values.update(preserve_audio=True, save_tiles_dir=None)
        values.update({key: value for key, value in overrides.items() if value is not None})
        if overrides.get('long_edge') is not None and overrides.get('target_resolution') is None:
            values['target_resolution'] = None
        for key in ('tile_t', 'tile_h', 'tile_w', 'long_edge', 'read_queue', 'write_queue'):
            if type(values[key]) is not int or values[key] <= 0:
                raise ValueError(f'{key} must be a positive integer')
        if (values['tile_t'] - 1) % 4 or values['tile_h'] % 32 or values['tile_w'] % 32:
            raise ValueError('VSR tiles require T=4n+1 and H/W divisible by 32')
        for key, limit in [('temporal_overlap', values['tile_t']),
                           ('spatial_overlap', min(values['tile_h'], values['tile_w']))]:
            if type(values[key]) is not int or not 0 <= values[key] < limit:
                raise ValueError(f'{key} must be non-negative and smaller than its tile')
        if values['color_ref'] not in ('global', 'chunk', 'none'):
            raise ValueError('color_ref must be global, chunk or none')
        for key, maximum in [('color_ref_samples', None), ('crf', 51)]:
            value = values[key]
            if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
                raise ValueError(f'Invalid {key}')
        for key in ('preserve_audio', 'gpu_postprocess'):
            if type(values[key]) is not bool:
                raise ValueError(f'{key} must be boolean')
        if values['target_resolution'] is not None:
            from .geometry import parse_resolution
            h, w = parse_resolution(values['target_resolution'])
            if min(h, w) <= 0 or h % 2 or w % 2:
                raise ValueError('target_resolution must contain positive even HxW dimensions')
        return values

    def model_kwargs(self):
        import torch
        keys = ('checkpoint_dir', 'wan_root', 'vae_cpu_offload', 'dit_cpu_offload',
                'dit_layerwise_offload', 'dit_offload_prefetch_size', 'pin_cpu_memory',
                'cudnn_benchmark', 'channels_last_3d', 'compile_decoder', 'compile_encoder',
                'decoder_implicit_padding', 'cache_dit_condition', 'tile_t', 'tile_h', 'tile_w')
        return {**{key: getattr(self, key) for key in keys},
                'dtype': getattr(torch, DTYPES[self.weight_dtype]),
                't_overlap': self.temporal_overlap, 's_overlap': self.spatial_overlap}
