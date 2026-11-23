"""Bounded per-request stage timings, also transportable from process workers.

Repeated stages are summed per request before observation. Nested stages may
overlap; their durations must not be added to estimate request wall time.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import inspect
import time


STAGES = frozenset(('input_decode', 'parameter_update', 'parameter_reinitialize',
                    'compute', 'output_encode', 'output_validate', 'graph_bind',
                    'graph_collect', 'batch_key'))
_samples = ContextVar('server_stage_samples', default=None)


@contextmanager
def stage(name):
    samples = _samples.get()
    if samples is None:
        yield
        return
    if name not in STAGES:
        raise ValueError('Unknown execution stage')
    started, outcome = time.perf_counter(), 'success'
    try:
        yield
    except BaseException:
        outcome = 'error'
        raise
    finally:
        key = name, outcome
        samples[key] = samples.get(key, 0.0) + time.perf_counter() - started


def record(storage, entry, values):
    metrics = getattr(storage, 'metrics', None)
    if metrics is not None:
        for name, outcome, seconds in values:
            metrics.stages.labels(entry, name, outcome).observe(seconds)


@contextmanager
def collection(executor):
    if _samples.get() is not None:
        yield
        return
    samples = {}
    token = _samples.set(samples)
    try:
        yield
    finally:
        _samples.reset(token)
        values = [(name, outcome, seconds) for (name, outcome), seconds in samples.items()]
        sink = getattr(executor.storage, 'stage_sink', None)
        if sink is not None:
            sink(values)
        else:
            record(executor.storage, executor.config.name, values)


def execution_stages(function):
    if inspect.iscoroutinefunction(function):
        @wraps(function)
        async def asynchronous(self, *args, **kwargs):
            with collection(self):
                return await function(self, *args, **kwargs)
        return asynchronous
    @wraps(function)
    def synchronous(self, *args, **kwargs):
        with collection(self):
            return function(self, *args, **kwargs)
    return synchronous


def measured(name):
    def decorate(function):
        if inspect.iscoroutinefunction(function):
            @wraps(function)
            async def asynchronous(*args, **kwargs):
                with stage(name):
                    return await function(*args, **kwargs)
            return asynchronous
        @wraps(function)
        def synchronous(*args, **kwargs):
            with stage(name):
                return function(*args, **kwargs)
        return synchronous
    return decorate
