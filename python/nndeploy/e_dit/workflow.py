"""Algorithm-dispatched editor workflows."""
from .params import DiTParams
from .registry import get_model


def build_workflow(params, **kwargs):
    params = params if isinstance(params, DiTParams) else DiTParams(**params)
    return get_model(params.model).build_workflow(params, **kwargs)

