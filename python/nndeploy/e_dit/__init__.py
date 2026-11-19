"""DiT inference on native DAGs, with lazily registered algorithm implementations."""
from .registry import register_model, model_names, describe_model
from .params import DiTParams
from .workflow import build_workflow
from .graph import build_graph, server_config

__all__ = ['register_model', 'model_names', 'describe_model', 'DiTParams', 'DiTResources',
           'build_graph', 'build_workflow', 'server_config', 'Req']


def __getattr__(name):
    if name == 'DiTResources':
        from .resources import DiTResources
        return DiTResources
    if name == 'Req':
        from .nodes.schedule_batch import Req
        return Req
    raise AttributeError(name)

# JSON workflows must resolve their saved node keys before any Python factory call.
from .registry import register_builtin_nodes as _register_builtin_nodes
_register_builtin_nodes()
