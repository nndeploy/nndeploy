"""Device-scoped platform queries for model code."""
import torch
from nndeploy.e_dit.layers.attention.backends import AttentionBackendEnum

def get_local_torch_device():
    from nndeploy.e_dit.config.server_args import get_global_server_args
    args = get_global_server_args()
    return torch.device(args.device)

class Platform:
    @property
    def device_type(self):
        return get_local_torch_device().type
    def is_cuda(self):
        return torch.cuda.is_available()
    def is_musa(self):
        return False
    def is_amp_supported(self):
        return self.device_type == "cuda"
    def is_mps(self):
        return self.device_type == 'mps'

current_platform = Platform()
