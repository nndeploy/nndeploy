"""Trusted, application-local algorithm factories; never populated over HTTP."""
import importlib
import inspect
import re
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ServiceAdapter:
    """Trusted factory result for adapters which also mount shared resources.

    The factory validates options without loading models. ``install`` receives
    the existing Server and the entry name and registers through Server.add.
    Fixed-URL adapters may declare an exclusive key; ordinary factories remain
    reusable across any number of named entries.
    """

    install: Callable
    exclusive_key: str | None = None

    def __post_init__(self):
        if not callable(self.install) or inspect.iscoroutinefunction(self.install):
            raise ValueError('Service adapter installer must be a synchronous callable')
        if self.exclusive_key is not None and (not isinstance(self.exclusive_key, str) or not self.exclusive_key):
            raise ValueError('Service adapter exclusive key must be a nonempty string')


class Registry:
    def __init__(self):
        self.factories = {}

    def register(self, name, factory):
        if not isinstance(name, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}', name):
            raise ValueError('Algorithm name must be a simple identifier')
        if name in self.factories:
            raise ValueError(f'Duplicate algorithm name: {name}')
        if isinstance(factory, str):
            module, separator, attribute = factory.partition(':')
            if not separator or not all(part.isidentifier() for part in module.split('.')) or not attribute.isidentifier():
                raise ValueError('Algorithm factory must use module:function syntax')
        elif not callable(factory) or inspect.iscoroutinefunction(factory):
            raise ValueError('Algorithm factory must be a synchronous callable or module:function')
        self.factories[name] = factory

    def create(self, name, options):
        if not isinstance(name, str) or name not in self.factories:
            raise ValueError(f'Unknown registered algorithm: {name}')
        factory = self.factories[name]
        if isinstance(factory, str):
            module, attribute = factory.split(':')
            try:
                factory = getattr(importlib.import_module(module), attribute)
            except (ImportError, AttributeError) as exc:
                raise ValueError(f'Cannot load registered algorithm {name}: {exc}') from exc
        if not callable(factory) or inspect.iscoroutinefunction(factory):
            raise ValueError(f'Algorithm {name} factory must be a synchronous callable')
        if options is not None and not isinstance(options, dict):
            raise ValueError('Algorithm options must be a dictionary')
        import copy
        value = factory(**copy.deepcopy(options or {}))
        if isinstance(value, ServiceAdapter):
            return value
        if inspect.iscoroutine(value):
            value.close()
        allowed = {'workflow', 'graph', 'graph_factory', 'deploy', 'handler_factory',
                   'config', 'input_model', 'output_model', 'router', 'router_prefix', 'router_permissions'}
        if not isinstance(value, dict) or value.keys() - allowed:
            raise ValueError(f'Algorithm {name} must return Server.add arguments or a ServiceAdapter')
        return dict(value)
