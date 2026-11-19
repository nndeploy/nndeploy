"""Operation-local native module injection and optional CPU dispatch measurements.

A replacement has signature replacement(native, *args, **kwargs), including the
native module's self argument. It must preserve that boundary's tensor/state
contract. No global method replacement or checkpoint-key changes are performed.
"""

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import time

_active = ContextVar("sam3_modules", default=None)


class Modules:
    def __init__(self, replacements=None, measure=False, observe=True):
        self.replacements = dict(replacements or {})
        self.measure = measure
        self.observe = observe
        self.calls = Counter()
        self.cpu_seconds = Counter()
        self.last_shapes = {}

    @contextmanager
    def activate(self):
        token = _active.set(self)
        try:
            yield
        finally:
            _active.reset(token)

    def snapshot(self):
        return {"calls": dict(self.calls), "cpu_wall_seconds_inclusive": dict(self.cpu_seconds), "last_tensor_metadata": dict(self.last_shapes)}


def _tensor_metadata(value, depth=0):
    """Describe tensor structure without reading values or synchronizing CUDA."""
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(getattr(value, "device", "cpu"))}
    if depth < 3 and isinstance(value, dict):
        return {str(k): _tensor_metadata(v, depth + 1) for k, v in list(value.items())[:8]}
    if depth < 3 and isinstance(value, (list, tuple)):
        return [_tensor_metadata(x, depth + 1) for x in value[:8]]
    return type(value).__name__


def boundary(name):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            modules = _active.get()
            if modules is None:
                return function(*args, **kwargs)
            if modules.observe:
                modules.calls[name] += 1
            started = time.perf_counter() if modules.measure and modules.observe else None
            try:
                replacement = modules.replacements.get(name)
                result = replacement(function, *args, **kwargs) if replacement else function(*args, **kwargs)
                if started is not None:
                    modules.last_shapes[name] = {"inputs": _tensor_metadata(args[1:]), "kwargs": _tensor_metadata(kwargs), "outputs": _tensor_metadata(result)}
                return result
            finally:
                if started is not None:
                    modules.cpu_seconds[name] += time.perf_counter() - started
        return wrapped
    return decorate


class PropagationCancelled(Exception):
    """Cooperative stop at a safe frame boundary; the owner must reset the session."""


def check_cancel(state):
    event = state.get("_sam3_workflow_cancel")
    if event is not None and event.is_set():
        raise PropagationCancelled()


def observe_path(name):
    modules = _active.get()
    if modules is not None and modules.observe:
        modules.calls[name] += 1
