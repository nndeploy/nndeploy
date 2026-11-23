"""Whole-component residency for eager nodes (not layerwise/FSDP ownership)."""
from contextlib import contextmanager, nullcontext
from threading import Lock, RLock
from weakref import WeakKeyDictionary
import torch

_states = WeakKeyDictionary()
_states_lock = Lock()


@contextmanager
def use_component(model, *, device, offload=False, autocast_dtype=None):
    """Serialize placement/use, preserve weights' dtype, offload on all exits.

    Nested uses must agree on placement and offload policy. The outermost use
    owns cleanup. Do not wrap modules already owned by a sharding/offload manager.
    """
    policy = (str(torch.device(device)), bool(offload))
    with _states_lock:
        state = _states.setdefault(model, {'lock': RLock(), 'depth': 0, 'policy': None})
    with state['lock']:
        outer = state['depth'] == 0
        if not outer and state['policy'] != policy:
            raise ValueError('Nested component use has conflicting residency policies')
        if outer:
            state['policy'] = policy
        state['depth'] += 1
        try:
            if outer:
                model.to(device)
            amp = torch.autocast(torch.device(device).type, dtype=autocast_dtype) if autocast_dtype is not None else nullcontext()
            with amp:
                yield model
        finally:
            state['depth'] -= 1
            if outer:
                state['policy'] = None
                if offload:
                    model.to('cpu')
