"""Request-local model forward metadata, isolated across threads and async tasks."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

@dataclass
class ForwardContext:
    current_timestep: int | None
    attn_metadata: Any = None
    forward_batch: Any = None

_context = ContextVar('dit_forward_context', default=None)

def get_forward_context():
    context = _context.get()
    if context is None:
        raise RuntimeError('No active diffusion forward context')
    return context

@contextmanager
def set_forward_context(current_timestep, attn_metadata=None, forward_batch=None):
    token = _context.set(ForwardContext(current_timestep, attn_metadata, forward_batch))
    try:
        yield _context.get()
    finally:
        _context.reset(token)
