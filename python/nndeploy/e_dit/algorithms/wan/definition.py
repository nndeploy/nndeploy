"""Wan model families and native DAG factories."""
from copy import deepcopy
from nndeploy.e_dit.algorithms.base import AlgorithmAdapter

VARIANTS = {
    'wan_fun': ('WanI2V480PConfig', 'Wan2_1_Fun_1_3B_InP_SamplingParams'),
    'wan': ('WanT2V480PConfig', 'WanT2V_1_3B_SamplingParams'),
    'wan_t2v_720p': ('WanT2V720PConfig', 'WanT2V_14B_SamplingParams'),
    'wan_i2v': ('WanI2V480PConfig', 'WanI2V_14B_480P_SamplingParam'),
    'wan_i2v_720p': ('WanI2V720PConfig', 'WanI2V_14B_720P_SamplingParam'),
    'wan_dmd': ('FastWan2_1_T2V_480P_Config', 'FastWanT2V480PConfig'),
    'wan_turbo': ('TurboWanT2V480PConfig', 'FastWanT2V480PConfig'),
    'wan_i2v_dmd': ('TurboWanI2V720Config', 'Turbo_Wan2_2_I2V_A14B_SamplingParam'),
    'wan2_2_ti2v': ('Wan2_2_TI2V_5B_Config', 'Wan2_2_TI2V_5B_SamplingParam'),
    'wan2_2_ti2v_dmd': ('FastWan2_2_TI2V_5B_Config', 'Wan2_2_TI2V_5B_SamplingParam'),
    'wan2_2_t2v': ('Wan2_2_T2V_A14B_Config', 'Wan2_2_T2V_A14B_SamplingParam'),
    'wan2_2_i2v': ('Wan2_2_I2V_A14B_Config', 'Wan2_2_I2V_A14B_SamplingParam'),
    'wan_causal_dmd': ('SelfForcingWanT2V480PConfig', 'SelfForcingWanT2V480PConfig'),
}

class WanAdapter(AlgorithmAdapter):
    def capabilities(self):
        from nndeploy.e_dit.config.capabilities import AlgorithmCapabilities
        return AlgorithmCapabilities(
            tasks=('VIDEO_EDIT',) if self.name in ('wan_videoedit', 'wan_videoedit_dmd') else ('TI2V',) if 'ti2v' in self.name else ('I2V',) if 'i2v' in self.name or self.name == 'wan_fun' else ('T2V',), devices=('cpu', 'cuda'),
            attention_backends=('sdpa',), features=('lora', 'component_offload'),
            single_device=True, enforced=True,
        )

    def __init__(self, name):
        self.name = name
        self.required_components = ('transformer','text_encoder','tokenizer','vae','scheduler')
        if name in ('wan2_2_t2v','wan2_2_i2v','wan_i2v_dmd'):
            self.required_components += ('transformer_2',)
        if name in ('wan_i2v','wan_i2v_720p','wan_fun'):
            self.required_components += ('image_encoder','image_processor')

    def build_config(self, params):
        from . import config
        from nndeploy.e_dit.config.server_args import ServerArgs
        from nndeploy.e_dit.config.model_base import update_dataclass_config
        pc = update_dataclass_config(getattr(config,VARIANTS[self.name][0])(),deepcopy(params.pipeline))
        pc.component_loaders = {name:'nndeploy.e_dit.algorithms.wan.loaders:WanComponentLoader' for name in self.required_components if name != 'tokenizer'}
        options = dict(device='cuda',component_dtypes={'transformer':'bf16','transformer_2':'bf16','vae':'fp32','text_encoder':'fp32','image_encoder':'fp32'})
        options.update(deepcopy(params.options))
        return ServerArgs(model_path=params.model_path,algorithm=self.name,pipeline_config=pc,**options)

    def validate_config(self, args):
        import math
        self.capabilities().validate(args)
        pc=args.pipeline_config
        if not math.isfinite(args.lora_scale):raise ValueError('lora_scale must be finite')
        if pc.flow_shift is None or not math.isfinite(pc.flow_shift) or pc.flow_shift <= 0:raise ValueError('flow_shift must be positive and finite')
        boundary=getattr(pc,'boundary_ratio',None)
        if boundary is not None and not 0 < boundary < 1:raise ValueError('boundary_ratio must be between zero and one')
        steps=getattr(pc,'dmd_denoising_steps',None)
        if steps is not None and (not steps or any(type(t) is not int or not 0<t<=1000 for t in steps) or any(a<=b for a,b in zip(steps,steps[1:]))):
            raise ValueError('DMD timesteps must be strictly decreasing integers in (0,1000]')
        if args.pipeline_config.vae_tiling and self.name not in ('wan_videoedit', 'wan_videoedit_dmd'):
            raise ValueError('Native Wan VAE spatial tiling is not enabled; use vae_tiling=False')
        if args.pipeline_config.dit_config.attention_type not in ('original', 'full'):
            raise ValueError('Wan sparse attention is not enabled')

    def sampling(self, options):
        from . import sampling
        values = dict(seed=0, **options) if 'seed' not in options else dict(options)
        return getattr(sampling,VARIANTS[self.name][1])(**values)

    def validate_request(self, sampling, args):
        import math
        from pathlib import Path
        pc = args.pipeline_config
        from nndeploy.e_dit.config.request_validation import validate_generation_request
        validate_generation_request(sampling, self.name)
        scale = pc.vae_config.arch_config.scale_factor_spatial
        if sampling.height % (scale*2) or sampling.width % (scale*2):
            raise ValueError(f'Wan dimensions must be divisible by {scale*2}')
        if (sampling.num_frames-1)%pc.vae_config.arch_config.scale_factor_temporal:
            raise ValueError('Wan requires temporal_stride*n+1 frames')
        if sampling.enable_teacache and (getattr(sampling,'teacache_params',None) is None or getattr(pc,'dmd_denoising_steps',None)):
            raise ValueError('TeaCache requires a calibrated, non-DMD Wan configuration')
        if sampling.image_path and getattr(pc,'is_causal',False):
            raise ValueError('Causal DMD currently accepts text conditioning')
        if pc.task_type.name=='T2V' and sampling.image_path:
            raise ValueError('Use an I2V or TI2V variant for image conditioning')
        from nndeploy.e_dit.config.request_validation import finite_number
        if not finite_number(sampling.guidance_scale_2, minimum=0): raise ValueError('Invalid secondary guidance scale')
        if pc.task_type.name=='I2V' and not sampling.image_path: raise ValueError('I2V requires image_path')
        if sampling.image_path and not Path(sampling.image_path).is_file(): raise ValueError('image_path must be a local file')

    def build_graph(self, params, name=None):
        from .graph import WanGraph
        return WanGraph(params,name or self.name)

    def build_workflow(self,params,**kwargs):
        from .workflow import build_workflow
        return build_workflow(params,**kwargs)

    def server_config(self,**kwargs):
        from .graph import server_config
        return server_config(**kwargs)

    def initialize_resources(self, resources, args):
        if args.lora_path:
            from nndeploy.e_dit.adapters.lora import LoRAManager
            from nndeploy.e_dit.loader.weight_mapping import map_weight_name
            for name in ('transformer','transformer_2'):
                model=resources.modules.get(name)
                if model is None: continue
                manager=LoRAManager(model, name_mapping={**model.lora_param_names_mapping, **model.param_names_mapping})
                resources.register_manager(manager)
                manager.load('default',args.lora_path)
                manager.activate([('default',args.lora_scale)])

for _name in VARIANTS:
    globals()[_name+'_adapter'] = WanAdapter(_name)

class WanVideoEditAdapter(WanAdapter):
    def __init__(self, name='wan_videoedit'):
        super().__init__(name)
    def build_config(self,params):
        from .videoedit_config import WanVideoEditPipelineConfig
        from nndeploy.e_dit.config.server_args import ServerArgs
        from nndeploy.e_dit.config.model_base import update_dataclass_config
        pc=update_dataclass_config(WanVideoEditPipelineConfig(),deepcopy(params.pipeline))
        pc.component_loaders={name:'nndeploy.e_dit.algorithms.wan.loaders:WanVideoEditComponentLoader' for name in self.required_components if name!='tokenizer'}
        options=dict(device='cuda',component_dtypes={'transformer':'bf16','text_encoder':'bf16','vae':'bf16'})
        options.update(deepcopy(params.options))
        return ServerArgs(model_path=params.model_path,algorithm=self.name,pipeline_config=pc,**options)
    def sampling(self,options):
        from .videoedit_sampling import WanVideoEditSamplingParams
        values = {"seed": 0, **options}
        if self.name == 'wan_videoedit_dmd':
            # Source dual-service gateway sampling policy for distilled weights.
            values.update(num_inference_steps=4, guidance_scale=1.0,
                          dynamic_cfg=False, negative_prompt=None)
        return WanVideoEditSamplingParams(**values)
    def validate_request(self,sampling,args):
        from pathlib import Path
        sampling._validate_videoedit()
        if type(sampling.seed) is not int or not 0 <= sampling.seed < 2**63:raise ValueError("Invalid seed")
        if sampling.save_crop_only and not sampling.return_file_paths_only:
            raise ValueError("save_crop_only requires file output; use infer_files()")
        if sampling.return_file_paths_only:
            from nndeploy.e_dit.video.artifacts import video_output_paths, validate_output_paths
            if not sampling.output_path or not sampling.output_file_name:
                raise ValueError('File output requires output_path and output_file_name')
            paths = video_output_paths(Path(sampling.output_path) / sampling.output_file_name,
                                       include_crop=sampling.save_crop_only)
            validate_output_paths(paths, (sampling.video_input_path, sampling.mask_input_path,
                                          sampling.reference_image_path))
        for key in ('video_input_path','mask_input_path','reference_image_path'):
            if not getattr(sampling,key,None) or not Path(getattr(sampling,key)).is_file():raise ValueError(f'{key} must be a local file')
        if type(sampling.num_inference_steps) is not int or sampling.num_inference_steps<=0:raise ValueError('num_inference_steps must be positive')
        if isinstance(sampling.prompt,list) or sampling.num_outputs_per_prompt!=1:raise ValueError('VideoEdit executes one video per request')
        if sampling.use_clip and sampling.clip_preprocess=='diffuser' and 'image_processor' not in args.component_paths:
            raise ValueError('VideoEdit CLIP preprocessing requires image_processor')
        if sampling.use_clip and 'image_encoder' not in args.component_paths:
            raise ValueError('VideoEdit use_clip requires component_paths.image_encoder')
    def initialize_resources(self,resources,args):
        super().initialize_resources(resources,args)
        from pathlib import Path
        for component in ('image_encoder','image_processor'):
            default=Path(args.model_path)/component
            if component not in args.component_paths and default.is_dir():args.component_paths[component]=str(default)
        if args.component_paths.get('image_encoder'):
            from transformers import CLIPVisionModel,CLIPImageProcessor
            import torch
            resources.modules['image_encoder']=CLIPVisionModel.from_pretrained(args.component_paths['image_encoder'],dtype=torch.float32,local_files_only=True).eval()
            processor=args.component_paths.get('image_processor')
            if processor:resources.modules['image_processor']=CLIPImageProcessor.from_pretrained(processor,local_files_only=True)

wan_videoedit_adapter=WanVideoEditAdapter()

wan_videoedit_dmd_adapter=WanVideoEditAdapter("wan_videoedit_dmd")
