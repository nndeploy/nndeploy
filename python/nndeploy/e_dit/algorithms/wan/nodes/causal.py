"""Blockwise self-forcing DMD with request-local KV and text caches.

Adapted from SGLang causal_denoising.py (Apache-2.0).
"""
import torch
from diffusers.utils.torch_utils import randn_tensor
from nndeploy.e_dit.execution_context import set_forward_context

def causal_steps(owner,batch,args):
    model=owner.transformer
    model.to(args.device)
    dtype=next(model.parameters()).dtype
    cfg=model.config
    scheduler=batch.extra['scheduler']
    latents=batch.latents
    b,c,frames,h,w=latents.shape
    frame_tokens=h*w//(cfg.patch_size[-1]*cfg.patch_size[-2])
    block=cfg.num_frames_per_block
    first=model.independent_first_frame
    if (frames-int(first))%block:raise ValueError('Latent frame count does not fit causal block size')
    sizes=([1] if first else [])+[block]*((frames-int(first))//block)
    length=cfg.local_attn_size if cfg.local_attn_size!=-1 else max(frames,cfg.sliding_window_num_frames)
    if length < block + cfg.sink_size:
        raise ValueError('Causal cache must fit one block plus sink frames')
    for layer in model.blocks:
        layer.attn1.max_attention_size=length*frame_tokens
    def zeros(n):return torch.zeros(b,n,cfg.num_attention_heads,cfg.attention_head_dim,device=args.device,dtype=dtype)
    kv=[dict(k=zeros(length*frame_tokens),v=zeros(length*frame_tokens),global_end_index=torch.zeros(1,device=args.device,dtype=torch.long),local_end_index=torch.zeros(1,device=args.device,dtype=torch.long)) for _ in range(cfg.num_layers)]
    cross=[dict(k=zeros(batch.prompt_embeds[0].shape[1]),v=zeros(batch.prompt_embeds[0].shape[1]),is_init=False) for _ in range(cfg.num_layers)]
    times=batch.timesteps
    if args.pipeline_config.warp_denoising_step:
        full=torch.cat([scheduler.timesteps,torch.zeros(1,device=args.device)])
        times=full[1000-times.long()]
    start=0
    try:
        for count in sizes:
            current=latents[:,:,start:start+count]
            def predict(x,t,i):
                with set_forward_context(current_timestep=i,forward_batch=batch):
                    return model(x.to(dtype),[e.to(dtype) for e in batch.prompt_embeds],t.expand(b,1),kv_cache=kv,crossattn_cache=cross,current_start=start*frame_tokens,start_frame=start).float()
            for i,t in enumerate(times):
                yield
                pred=predict(current,t,i)
                sigma=scheduler.sigmas[(scheduler.timesteps-t).abs().argmin()].to(current.device)
                clean=(current.double()-sigma.double()*pred.double()).to(pred.dtype)
                if i+1<len(times):
                    sigma=scheduler.sigmas[(scheduler.timesteps-times[i+1]).abs().argmin()].to(current.device)
                    noise=randn_tensor(current.shape,generator=batch.generator,device=current.device,dtype=current.dtype)
                    current=(1-sigma)*clean+sigma*noise
                else:current=clean
            latents[:,:,start:start+count]=current
            # Refresh each block's KV entries using its clean output.
            predict(current,torch.tensor(getattr(args.pipeline_config,'context_noise',0),device=args.device),0)
            start+=count
        batch.latents=latents
    finally:
        kv.clear()
        cross.clear()
