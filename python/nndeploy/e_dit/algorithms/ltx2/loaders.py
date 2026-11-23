"""LTX component factories using shared strict, alias-aware checkpoint loading."""
from copy import deepcopy
from pathlib import Path
import json
from nndeploy.e_dit.loader.component_loaders.component_loader import ComponentLoader
from nndeploy.e_dit.loader.meta_load import load_safetensors_model
from nndeploy.e_dit.loader.weight_mapping import map_weight_name

class LTXComponentLoader(ComponentLoader):
    component_names = []

    def should_offload(self, args):
        # Components are moved to their execution device by their owning node.
        return True

    def load_component(self, component_model_path, server_args, component_name, transformers_or_diffusers, dtype):
        from .models.ltx_2 import LTX2VideoTransformer3DModel
        from .models.ltx_2_vae import AutoencoderKLLTX2Video
        from .models.ltx_2_audio import AutoencoderKLLTX2Audio
        from .models.ltx_2_connector import LTX2TextConnectors
        from .models.ltx_2_vocoder import LTX2Vocoder
        from .configs.ltx_2_connector import LTX2ConnectorConfig
        from .configs.ltx_vocoder import LTXVocoderConfig
        pc = server_args.pipeline_config
        factories = {
            'transformer': (LTX2VideoTransformer3DModel, pc.dit_config),
            'vae': (AutoencoderKLLTX2Video, pc.vae_config),
            'audio_vae': (AutoencoderKLLTX2Audio, pc.audio_vae_config),
            'connectors': (LTX2TextConnectors, LTX2ConnectorConfig()),
            'vocoder': (LTX2Vocoder, LTXVocoderConfig()),
        }
        cls, config = factories[component_name]
        config = deepcopy(config)
        raw = json.loads((Path(component_model_path) / 'config.json').read_text())
        config.update_model_arch(raw)
        if component_name in ('transformer', 'vae', 'audio_vae'):
            setattr(pc, {'transformer': 'dit_config', 'vae': 'vae_config', 'audio_vae': 'audio_vae_config'}[component_name], config)
        def factory():
            if component_name == 'transformer':
                return cls(config=config, hf_config=raw)
            return cls(config)
        rules = getattr(config, 'param_names_mapping', {})
        return load_safetensors_model(factory, component_model_path, dtype=dtype,
                                      key_mapping=lambda name: map_weight_name(name, rules))

class LTXSchedulerLoader(ComponentLoader):
    component_names = []
    def load_component(self, component_model_path, server_args, component_name, transformers_or_diffusers, dtype):
        from .scheduler import LTX2FlowMatchScheduler
        return LTX2FlowMatchScheduler.from_pretrained(component_model_path, local_files_only=True), None
