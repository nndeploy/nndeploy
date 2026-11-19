"""Optional request control for long-running Python Graph nodes.

Independent of Server: callers may provide a cancellation probe and progress
sink, while native graph execution keeps its existing interruption semantics.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable


@dataclass
class ExecutionContext:
    cancelled: Callable[[], bool] = lambda: False
    publish: Callable[[dict], None] | None = None


_current = ContextVar('nndeploy_graph_execution', default=None)


def current_execution():
    return _current.get()


@contextmanager
def execution_scope(context):
    token = _current.set(context)
    try:
        yield context
    finally:
        _current.reset(token)
