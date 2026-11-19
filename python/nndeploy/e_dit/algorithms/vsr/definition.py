"""VSR algorithm registration without importing model weights or diffusers."""
from nndeploy.e_dit.algorithms.base import AlgorithmAdapter
from .config import VSRConfig


class VSRAdapter(AlgorithmAdapter):
    required_components = ('transformer', 'vae')

    def capabilities(self):
        from nndeploy.e_dit.config.capabilities import AlgorithmCapabilities
        return AlgorithmCapabilities(tasks=('VSR',), devices=('cpu', 'cuda'),
                                     attention_backends=('sdpa',),
                                     features=('component_offload', 'layerwise_offload',
                                               'compile', 'tile_parallel'))

    def build_config(self, params):
        overlap = set(params.options) & set(params.pipeline)
        if overlap:
            raise ValueError(f'Duplicate VSR options: {sorted(overlap)}')
        return VSRConfig(checkpoint_dir=params.model_path, **params.options, **params.pipeline)

    def build_graph(self, params, name=None):
        from .graph import VSRGraph
        return VSRGraph(params, name or 'wan_vsr')

    def build_workflow(self, params, **kwargs):
        from .workflow import build_workflow
        return build_workflow(params, **kwargs)

    def server_config(self, **kwargs):
        from .graph import server_config
        return server_config(**kwargs)


adapter = VSRAdapter()
