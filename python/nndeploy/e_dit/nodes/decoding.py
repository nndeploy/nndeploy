"""Shared decoder residency bindings."""
from .base import StateNode

class DecodingNode(StateNode):
    def setup(self, vae, pipeline=None):
        self.vae = vae
    def load_model(self):
        self.vae.to(self.server_args.device)
    def offload_model(self):
        if self.server_args.vae_cpu_offload:
            for name in ('vae', 'audio_vae', 'vocoder'):
                model = getattr(self, name, None)
                if model is not None: model.to('cpu')
