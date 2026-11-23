"""LTX request fields, kept out of the common diffusion state."""
from dataclasses import dataclass, field
from typing import Any
from nndeploy.e_dit.nodes.schedule_batch import Req

@dataclass(init=False)
class LTX2State(Req):
    audio_latents: Any = None
    audio_prompt_embeds: list = field(default_factory=list)
    negative_audio_prompt_embeds: list = field(default_factory=list)
    trajectory_timesteps: Any = None
    trajectory_latents: Any = None
    trajectory_audio_latents: Any = None
    raw_latent_shape: Any = None
    raw_audio_latent_shape: Any = None
    did_sp_shard_latents: bool = False
    debug: bool = False
    return_trajectory_latents: bool = False
    ltx2_num_image_tokens: int = 0
