"""Native node helpers shared by VideoEdit conditioning and denoising."""
from contextlib import contextmanager
from types import SimpleNamespace
from nndeploy.e_dit.nodes.generator_loop import GeneratorLoop
from nndeploy.e_dit.nodes.text_encoding import TextEncodingNode
from nndeploy.e_dit.nodes.control import service_checkpoint
from nndeploy.e_dit.config.server_args import get_global_server_args

def check_request_timeout(batch):
    service_checkpoint(batch,get_global_server_args(),phase='videoedit')

class VideoEditTextEncoder:
    def __init__(self,encoders,tokenizers):
        self.text_encoders=encoders
        self.tokenizers=tokenizers
    def encode_text(self,text,args,encoder_index=None,return_attention_mask=True,dtype=None):
        batch=SimpleNamespace(max_sequence_length=None,num_outputs_per_prompt=1)
        values=TextEncodingNode.encode(self,text,batch,args)
        return ([v.to(dtype=dtype) for v in values[0]],values[1],values[2])

class VideoEditDenoiseSupport(GeneratorLoop):
    def _manage_device_placement(self,model,other,args):
        model.to(args.device)
    def _maybe_enable_cache_dit(self,steps,batch):
        batch.num_inference_steps = steps
        self.transformer.reset_teacache_state()
    def _build_attn_metadata(self,*args,**kwargs):return None
    @contextmanager
    def progress_bar(self, total):
        yield SimpleNamespace(update=lambda:None)
