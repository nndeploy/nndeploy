from nndeploy.e_dit.models.schedulers.request import prepare_scheduler
import torch
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
import numpy as np
from nndeploy.e_dit.nodes.base import StateNode, operation
from nndeploy.e_dit.nodes.graph import register_type, StateGraph
from nndeploy.e_dit.nodes.generator_loop import GeneratorLoop
from nndeploy.e_dit.nodes.text_encoding import TextEncodingNode
from nndeploy.e_dit.nodes.conditioning import encode_image, encode_vae
from nndeploy.e_dit.execution_context import set_forward_context
from nndeploy.e_dit.nodes.schedule_batch import OutputBatch

@register_type
class WanPrepareNode(StateNode):
    def setup(self,vae,scheduler,image_encoder=None,image_processor=None):
        self.vae,self.scheduler=vae,scheduler
        self.image_encoder,self.image_processor=image_encoder,image_processor

    @operation
    def run(self,batch,args):
        pc=args.pipeline_config
        cfg=pc.vae_config.arch_config
        shape=(batch.batch_size,cfg.z_dim,(batch.num_frames-1)//cfg.scale_factor_temporal+1,batch.height//cfg.scale_factor_spatial,batch.width//cfg.scale_factor_spatial)
        batch.latents=randn_tensor(shape,generator=batch.generator,device=torch.device(args.device),dtype=torch.float32)
        steps=getattr(pc,'dmd_denoising_steps',None)
        if steps:
            # DMD converts velocity using the full shifted training schedule.
            scheduler=prepare_scheduler(self.scheduler,num_inference_steps=1000,device=args.device)
            batch.timesteps=torch.tensor(steps,device=args.device,dtype=torch.long)
        else:
            scheduler=prepare_scheduler(self.scheduler,num_inference_steps=batch.num_inference_steps,device=args.device)
            batch.timesteps=scheduler.timesteps
        batch.extra['scheduler']=scheduler
        if batch.image_path:
            image=Image.open(batch.image_path).convert('RGB')
            if self.image_encoder is not None:
                out=encode_image(self.image_encoder,self.image_processor,image,
                                 device=args.device,offload=args.image_encoder_cpu_offload,
                                 output_hidden_states=True)
                batch.image_embeds=[out.hidden_states[-2].repeat(batch.batch_size,1,1)]
            image=image.resize((batch.width,batch.height),Image.Resampling.LANCZOS)
            pixels=torch.from_numpy(np.array(image).copy()).permute(2,0,1).float()/127.5-1
            video=torch.zeros((1,3,batch.num_frames,batch.height,batch.width))
            video[:,:,0]=pixels
            latent=encode_vae(self.vae,video,device=args.device,offload=args.vae_cpu_offload)
            mean=torch.tensor(cfg.latents_mean,device=args.device).view(1,-1,1,1,1)
            std=torch.tensor(cfg.latents_std,device=args.device).view(1,-1,1,1,1)
            latent=((latent-mean)/std).repeat(batch.batch_size,1,1,1,1)
            if pc.task_type.name=='TI2V':
                batch.extra['first_frame_latent']=latent[:,:,:1]
                batch.latents[:,:,:1]=latent[:,:,:1]
            else:
                mask=torch.zeros(1,1,batch.num_frames,batch.height//cfg.scale_factor_spatial,batch.width//cfg.scale_factor_spatial,device=args.device)
                mask[:,:,0]=1
                mask=torch.cat([mask[:,:,:1].repeat(1,1,cfg.scale_factor_temporal,1,1),mask[:,:,1:]],2)
                mask=mask.view(1,shape[2],cfg.scale_factor_temporal,*shape[-2:]).transpose(1,2)
                batch.image_latent=torch.cat([mask.repeat(batch.batch_size,1,1,1,1),latent],1)
        return batch

@register_type
class WanDenoiseNode(GeneratorLoop):
    def setup(self,transformer,scheduler,transformer_2=None):
        self.transformer,self.scheduler,self.transformer_2=transformer,scheduler,transformer_2

    def compute(self,batch,args):
        pc=args.pipeline_config
        if getattr(pc,'is_causal',False):
            from .causal import causal_steps
            yield from causal_steps(self,batch,args)
            return
        self.scheduler=batch.extra['scheduler']
        latents=batch.latents
        dmd=bool(getattr(pc,'dmd_denoising_steps',None))
        for i,t in enumerate(batch.timesteps):
            yield
            low=self.transformer_2 is not None and t.item()<pc.boundary_ratio*self.scheduler.config.num_train_timesteps
            model=self.transformer_2 if low else self.transformer
            other=self.transformer if low else self.transformer_2
            if other is not None and args.dit_cpu_offload:other.to('cpu')
            model.to(args.device)
            dtype=next(model.parameters()).dtype
            x=latents
            if batch.image_latent is not None:x=torch.cat([x,batch.image_latent],1)
            timestep=t.expand(x.shape[0])
            first=batch.extra.get('first_frame_latent')
            if getattr(pc,'expand_timesteps',False):
                mask=torch.ones_like(latents[:,:1])
                if first is not None:mask[:,:,:1]=0
                timestep=(mask[:,0,:,::2,::2]*t).flatten(1)
            def predict(embeds,negative=False):
                batch.is_cfg_negative=negative
                with set_forward_context(current_timestep=i,forward_batch=batch):
                    return model(hidden_states=x.to(dtype),encoder_hidden_states=[e.to(dtype) for e in embeds],timestep=timestep,encoder_hidden_states_image=batch.image_embeds).float()
            velocity=predict(batch.prompt_embeds)
            if not dmd and batch.do_classifier_free_guidance:
                negative=predict(batch.negative_prompt_embeds,True)
                scale=batch.guidance_scale_2 if low else batch.guidance_scale
                velocity=negative+scale*(velocity-negative)
            if dmd:
                index=(self.scheduler.timesteps-t).abs().argmin()
                sigma=self.scheduler.sigmas[index].to(latents.device)
                clean=(latents.double()-sigma.double()*velocity.double()).to(velocity.dtype)
                if i+1<len(batch.timesteps):
                    index=(self.scheduler.timesteps-batch.timesteps[i+1]).abs().argmin()
                    sigma=self.scheduler.sigmas[index].to(latents.device)
                    noise=randn_tensor(latents.shape,generator=batch.generator,device=latents.device,dtype=latents.dtype)
                    latents=(1-sigma)*clean+sigma*noise
                else:latents=clean
            else:
                latents=self.scheduler.step(velocity,t,latents,return_dict=False)[0]
            if first is not None:latents[:,:,:1]=first
            batch.latents=latents

@register_type
class WanDecodeNode(StateNode):
    def setup(self,vae):self.vae=vae
    @operation
    def run(self,batch,args):
        cfg=args.pipeline_config.vae_config.arch_config
        self.vae.to(args.device)
        try:
            mean=torch.tensor(cfg.latents_mean,device=args.device).view(1,-1,1,1,1)
            std=torch.tensor(cfg.latents_std,device=args.device).view(1,-1,1,1,1)
            decoded=self.vae.decode((batch.latents*std+mean).to(next(self.vae.parameters()).dtype))
            if not isinstance(decoded,torch.Tensor):decoded=decoded.sample
            video=(decoded.float()/2+.5).clamp(0,1).permute(0,2,3,4,1).cpu().numpy()
            batch.extra['result']=OutputBatch(output=video,metrics=batch.metrics)
        finally:
            if args.vae_cpu_offload:self.vae.to('cpu')
        return batch

@register_type
class WanRequestGraph(StateGraph):
    def __init__(self,name='inference',inputs=None,outputs=None):
        super().__init__(name,inputs,outputs)
        self.compose([TextEncodingNode('text'),WanPrepareNode('prepare'),WanDenoiseNode('denoise'),WanDecodeNode('decode')])
