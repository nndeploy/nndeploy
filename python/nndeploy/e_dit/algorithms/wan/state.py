from dataclasses import dataclass
from nndeploy.e_dit.nodes.schedule_batch import Req

@dataclass(init=False)
class WanState(Req):
    enable_sequence_shard: bool = False
    enable_teacache: bool = False
    is_cfg_negative: bool = False
    raw_latent_shape: object = None
