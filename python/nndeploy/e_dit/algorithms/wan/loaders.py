"""Strict Wan checkpoint loaders; no SGLang runtime is required."""
import json
from copy import deepcopy
from pathlib import Path
from nndeploy.e_dit.loader.component_loaders.component_loader import ComponentLoader
from nndeploy.e_dit.loader.meta_load import load_safetensors_model
from nndeploy.e_dit.loader.weight_mapping import map_weight_name

class WanComponentLoader(ComponentLoader):
    component_names=[]
    def should_offload(self,args): return True
    def load_component(self,component_model_path,server_args,component_name,transformers_or_diffusers,dtype):
        path=component_model_path
        pc=server_args.pipeline_config
        if component_name=='scheduler':
            if getattr(pc,'dmd_denoising_steps',None):
                from nndeploy.e_dit.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler as Scheduler
            else:
                from nndeploy.e_dit.models.schedulers.scheduling_flow_unipc_multistep import FlowUniPCMultistepScheduler as Scheduler
            return Scheduler(shift=pc.flow_shift),None
        if component_name in ('text_encoder','image_encoder'):
            from transformers import UMT5EncoderModel, CLIPVisionModel
            cls=UMT5EncoderModel if component_name=='text_encoder' else CLIPVisionModel
            model,info=cls.from_pretrained(path,dtype=dtype,local_files_only=True,output_loading_info=True)
            if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys')): raise ValueError(f'Checkpoint mismatch: {info}')
            return model.eval(),info
        if component_name=='image_processor':
            from transformers import CLIPImageProcessor
            return CLIPImageProcessor.from_pretrained(path,local_files_only=True),None
        raw=json.loads((Path(path)/'config.json').read_text())
        if component_name=='vae':
            from .models.wanvae import AutoencoderKLWan as Model
            config=deepcopy(pc.vae_config)
            # Load both halves so full checkpoint validation remains strict.
            config.load_encoder=config.load_decoder=True
        else:
            from .models.wanvideo import WanTransformer3DModel as Model
            if getattr(pc,'is_causal',False):
                from .models.causal_wanvideo import CausalWanTransformer3DModel as Model
            config=deepcopy(pc.dit_config)
        config.update_model_arch(raw)
        if component_name!='transformer_2':
            setattr(pc,'vae_config' if component_name=='vae' else 'dit_config',config)
        factory=(lambda:Model(config)) if component_name=='vae' else (lambda:Model(config,hf_config=raw))
        return load_safetensors_model(factory,path,dtype=dtype,key_mapping=lambda n:map_weight_name(n,getattr(config,"param_names_mapping",{})))

class WanVideoEditComponentLoader(WanComponentLoader):
    component_names=[]
    def load_component(self,component_model_path,server_args,component_name,transformers_or_diffusers,dtype):
        path=component_model_path
        if component_name=='scheduler':
            from nndeploy.e_dit.models.schedulers.videoedit_flow_match import VideoEditFlowMatchScheduler
            return VideoEditFlowMatchScheduler(shift=server_args.pipeline_config.flow_shift or 5.,sigma_min=0.,extra_one_step=True),None
        if component_name=='vae':
            from diffusers import AutoencoderKLWan
            model,info=AutoencoderKLWan.from_pretrained(path,torch_dtype=dtype,local_files_only=True,output_loading_info=True)
            if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys')):raise ValueError(f'VAE checkpoint mismatch: {info}')
            return model.eval(),info
        if component_name=='transformer':
            from .models.wan_videoedit import WanVideoEditTransformer3DModel
            raw=json.loads((Path(path)/'config.json').read_text())
            config=deepcopy(server_args.pipeline_config.dit_config)
            config.update_model_arch(raw)
            server_args.pipeline_config.dit_config=config
            return load_safetensors_model(lambda:WanVideoEditTransformer3DModel(config,raw),path,dtype=dtype,key_mapping=lambda n:map_weight_name(n,getattr(config,"param_names_mapping",{})))
        return super().load_component(path,server_args,component_name,transformers_or_diffusers,dtype)
