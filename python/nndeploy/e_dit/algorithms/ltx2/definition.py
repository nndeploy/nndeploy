"""LTX-2 single-stage and two-stage algorithms on nndeploy DAG."""
from copy import deepcopy
from pathlib import Path
from nndeploy.e_dit.algorithms.base import AlgorithmAdapter

COMPONENTS = ('transformer', 'text_encoder', 'tokenizer', 'scheduler', 'vae', 'audio_vae', 'vocoder', 'connectors')

class LTX2Adapter(AlgorithmAdapter):
    def capabilities(self):
        from nndeploy.e_dit.config.capabilities import AlgorithmCapabilities
        return AlgorithmCapabilities(
            tasks=('T2V', 'I2V', 'AUDIO_VIDEO'), devices=('cpu', 'cuda'),
            attention_backends=('sdpa',), features=('lora', 'component_offload'),
            single_device=True, enforced=True,
        )

    required_components = COMPONENTS
    two_stage = False

    def build_config(self, params):
        from .config import LTX2PipelineConfig
        from nndeploy.e_dit.config.server_args import ServerArgs
        pipeline_options = deepcopy(params.pipeline)
        from nndeploy.e_dit.config.model_base import update_dataclass_config
        pc = update_dataclass_config(LTX2PipelineConfig(), pipeline_options)
        pc.component_loaders = {
            **{name: 'nndeploy.e_dit.algorithms.ltx2.loaders:LTXComponentLoader' for name in ('transformer','vae','audio_vae','vocoder','connectors')},
            'text_encoder': 'nndeploy.e_dit.loader.component_loaders.native_text:NativeTextEncoderLoader',
            'scheduler': 'nndeploy.e_dit.algorithms.ltx2.loaders:LTXSchedulerLoader',
            'spatial_upsampler': 'nndeploy.e_dit.algorithms.ltx2.upsampler_loader:UpsamplerLoader',
        }
        options = dict(device='cuda', component_dtypes={'transformer':'bf16','text_encoder':'bf16',
                       'vae':'fp32','audio_vae':'fp32','vocoder':'fp32','connectors':'bf16','spatial_upsampler':'bf16'})
        options.update(deepcopy(params.options))
        return ServerArgs(model_path=params.model_path, algorithm=params.model,
                          pipeline_class_name='LTX2TwoStagePipeline' if self.two_stage else 'LTX2Pipeline',
                          pipeline_config=pc, **options)

    def validate_config(self, args):
        from nndeploy.e_dit.config.request_validation import finite_number
        self.capabilities().validate(args)
        if not finite_number(args.lora_scale):
            raise ValueError('lora_scale must be finite')

    def validate_loading(self, args):
        import os
        if int(os.environ.get('WORLD_SIZE','1')) > 1:
            raise ValueError('LTX-2 graph requires one runtime rank')
        if self.two_stage:
            from .scheduler import _resolve_ltx2_two_stage_component_paths
            args.component_paths = _resolve_ltx2_two_stage_component_paths(args.model_path, args.component_paths)
            for name in ('spatial_upsampler','distilled_lora'):
                if not args.component_paths.get(name):
                    raise ValueError(f'LTX-2 two-stage requires component_paths[{name!r}]')

    def initialize_resources(self, resources, args):
        from .resource_hooks import LTX2Controller
        controller = LTX2Controller(resources, self.two_stage)
        resources.register_manager(controller)
        resources._ltx_controller = controller
        controller.open()

    def validate_request(self, sampling, config):
        from nndeploy.e_dit.config.request_validation import validate_generation_request
        validate_generation_request(sampling, 'ltx2_two_stage' if self.two_stage else 'ltx2')
        alignment = 64 if self.two_stage else config.pipeline_config.vae_scale_factor
        if sampling.height % alignment or sampling.width % alignment:
            raise ValueError(f'LTX-2 dimensions must be divisible by {alignment}')
        if (sampling.num_frames - 1) % config.pipeline_config.vae_temporal_compression:
            raise ValueError('LTX-2 frame count must be temporal_compression*n + 1')
        if sampling.generator_device not in ('cpu', config.device):
            raise ValueError('generator_device must be cpu or the execution device')
        prompts = sampling.prompt if isinstance(sampling.prompt,list) else [sampling.prompt]
        if sampling.image_path and not Path(sampling.image_path).is_file():
            raise ValueError('image_path must be a local image')
        if sampling.image_path and len(prompts) * sampling.num_outputs_per_prompt != 1:
            raise ValueError('LTX-2 image conditioning currently requires batch size one')

    def build_graph(self, params, name=None):
        from .graph import LTX2Graph
        return LTX2Graph(params, name or params.model)

    def build_workflow(self, params, **kwargs):
        from .workflow import build_workflow
        return build_workflow(params, **kwargs)

    def server_config(self, **kwargs):
        from .graph import server_config
        return server_config(**kwargs)

class LTX2TwoStageAdapter(LTX2Adapter):
    two_stage = True

adapter = LTX2Adapter()
two_stage_adapter = LTX2TwoStageAdapter()
