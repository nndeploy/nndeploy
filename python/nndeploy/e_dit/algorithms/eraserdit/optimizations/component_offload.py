"""Stage-level placement around SGLang DiT hooks and T5 FSDP movement."""
from functools import wraps
from nndeploy.e_dit.memory.phases import residency_scope
from contextlib import nullcontext
import torch
from nndeploy.e_dit.algorithms.eraserdit.optimizations.memory_phase_controller import MemoryPhase


def offload_component(component_name, *, phase=None):
    if phase is None:
        phase = {'text_encoder': MemoryPhase.TEXT_ENCODE,
                 'transformer': MemoryPhase.DENOISE}.get(component_name)

    def decorate(forward):
        @wraps(forward)
        def wrapped(self, batch, server_args, *args, **kwargs):
            controller = batch.extra.get('memory_phase_controller')
            if controller is None:
                if server_args.resolve_resource_policy().enabled:
                    raise RuntimeError('SGLang offload requires a pipeline memory controller')
                return forward(self, batch, server_args, *args, **kwargs)
            device = torch.device(server_args.device)
            scope = torch.cuda.device(device) if device.type == 'cuda' else nullcontext()
            with scope, residency_scope(
                controller, phase, component_name=component_name,
                window_key=(int(batch.extra.get('object_index', -1)),
                            int(batch.extra.get('window_index', -1))),
            ):
                return forward(self, batch, server_args, *args, **kwargs)
        return wrapped
    return decorate
