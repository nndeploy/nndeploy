"""Algorithm-dispatched native graph factories."""
from .params import DiTParams
from .registry import get_model


def build_graph(params=None, name=None):
    params = params if isinstance(params, DiTParams) else DiTParams(**(params or {}))
    return get_model(params.model).build_graph(params, name)


def server_config(*, model="eraserdit", isolation="process"):
    return get_model(model).server_config(isolation=isolation)

