"""Lazy model adapters; the framework does not import model-specific configs."""
from importlib import import_module

_COMPONENTS = {
    "EraserDiTLTXVideoTransformer3DModel": "nndeploy.e_dit.algorithms.eraserdit.models.transformer:EraserDiTLTXVideoTransformer3DModel",
    "EraserDiTAutoencoderKLLTXVideo": "nndeploy.e_dit.algorithms.eraserdit.models.vae:EraserDiTAutoencoderKLLTXVideo",
}
_COMPONENTS.update({'LTX2VideoTransformer3DModel': 'nndeploy.e_dit.algorithms.ltx2.models.ltx_2:LTX2VideoTransformer3DModel', 'AutoencoderKLLTX2Video': 'nndeploy.e_dit.algorithms.ltx2.models.ltx_2_vae:AutoencoderKLLTX2Video', 'AutoencoderKLLTX2Audio': 'nndeploy.e_dit.algorithms.ltx2.models.ltx_2_audio:AutoencoderKLLTX2Audio', 'LTX2TextConnectors': 'nndeploy.e_dit.algorithms.ltx2.models.ltx_2_connector:LTX2TextConnectors', 'LTX2Vocoder': 'nndeploy.e_dit.algorithms.ltx2.models.ltx_2_vocoder:LTX2Vocoder', 'LTX2FlowMatchScheduler': 'nndeploy.e_dit.algorithms.ltx2.scheduler:LTX2FlowMatchScheduler', 'LatentUpsampler': 'nndeploy.e_dit.algorithms.ltx2.models.latent_upsampler:LatentUpsampler'})
_COMPONENTS.update({
    'WanTransformer3DModel': 'nndeploy.e_dit.algorithms.wan.models.wanvideo:WanTransformer3DModel',
    'CausalWanTransformer3DModel': 'nndeploy.e_dit.algorithms.wan.models.causal_wanvideo:CausalWanTransformer3DModel',
    'WanVideoEditTransformer3DModel': 'nndeploy.e_dit.algorithms.wan.models.wan_videoedit:WanVideoEditTransformer3DModel',
    'AutoencoderKLWan': 'nndeploy.e_dit.algorithms.wan.models.wanvae:AutoencoderKLWan',
    'FlowUniPCMultistepScheduler': 'nndeploy.e_dit.models.schedulers.scheduling_flow_unipc_multistep:FlowUniPCMultistepScheduler',
    'FlowMatchEulerDiscreteScheduler': 'nndeploy.e_dit.models.schedulers.scheduling_flow_match_euler_discrete:FlowMatchEulerDiscreteScheduler',
    'VideoEditFlowMatchScheduler': 'nndeploy.e_dit.models.schedulers.videoedit_flow_match:VideoEditFlowMatchScheduler',
})
_MODELS = {'eraserdit': 'nndeploy.e_dit.algorithms.eraserdit.definition:adapter',
           'ltx2': 'nndeploy.e_dit.algorithms.ltx2.definition:adapter',
           'ltx2_two_stage': 'nndeploy.e_dit.algorithms.ltx2.definition:two_stage_adapter'}

_MODELS["wan_vsr"] = "nndeploy.e_dit.algorithms.vsr.definition:adapter"

_MODELS.update({name: "nndeploy.e_dit.algorithms.wan.definition:" + name + "_adapter" for name in ['wan_fun', 'wan_videoedit', 'wan_videoedit_dmd', 'wan', 'wan_t2v_720p', 'wan_i2v', 'wan_i2v_720p', 'wan_dmd', 'wan_turbo', 'wan_i2v_dmd', 'wan2_2_ti2v', 'wan2_2_ti2v_dmd', 'wan2_2_t2v', 'wan2_2_i2v', 'wan_causal_dmd']})


def register_model(name, adapter, *, components=None):
    if not isinstance(name, str) or not name or name in _MODELS:
        raise ValueError(f'Duplicate or invalid model name: {name!r}')
    entries = dict(components if components is not None else getattr(adapter, "component_architectures", {}))
    conflicts = {key for key, value in entries.items() if key in _COMPONENTS and _COMPONENTS[key] != value}
    if conflicts:
        raise ValueError(f'Duplicate component architectures: {sorted(conflicts)}')
    for architecture, path in entries.items():
        if not isinstance(architecture, str) or not architecture or not isinstance(path, str) or ':' not in path:
            raise ValueError('Components require architecture names and module:class import paths')
    _COMPONENTS.update(entries)
    _MODELS[name] = adapter


def model_names():
    return tuple(_MODELS)


def get_model(name):
    value = _MODELS[name]
    if isinstance(value, str):
        module, attribute = value.split(':', 1)
        value = getattr(import_module(module), attribute)
    return value


def component_paths():
    """Return lazy component declarations without importing algorithm implementations."""
    return dict(_COMPONENTS)


_NODE_CREATORS = []


def register_lazy_node(key, factory_path):
    """Keep serialized node keys available without importing model implementations."""
    from nndeploy.dag import NodeCreator, register_node

    class LazyCreator(NodeCreator):
        def create_node(self, name, inputs, outputs):
            module, attribute = factory_path.split(':', 1)
            factory = getattr(import_module(module), attribute)
            self.node = factory.create_node(name, inputs, outputs)
            return self.node

    creator = LazyCreator()
    _NODE_CREATORS.append(creator)
    register_node(key, creator)


def register_builtin_nodes():
    register_lazy_node("nndeploy.e_dit.WanVSR", "nndeploy.e_dit.algorithms.vsr.graph:_creator")
    register_lazy_node("nndeploy.e_dit.Wan", "nndeploy.e_dit.algorithms.wan.graph:_creator")
    for key in ("GenerationSamplingInput", "GeneratedVideoOutput"):
        register_lazy_node("nndeploy.e_dit." + key, "nndeploy.e_dit.nodes.media_io:_" + key + "_creator")
    register_lazy_node("nndeploy.e_dit.LTX2", "nndeploy.e_dit.algorithms.ltx2.graph:_creator")
    register_lazy_node('nndeploy.e_dit.EraserDiT',
                       'nndeploy.e_dit.algorithms.eraserdit.graph:_creator')
    for key in ('SamplingInput', 'VideoOutput'):
        register_lazy_node('nndeploy.e_dit.' + key,
                           'nndeploy.e_dit.algorithms.eraserdit.workflow:_' + key + '_creator')


def describe_model(name):
    """Describe capabilities without loading weights or concrete model classes."""
    return {'name': name, **get_model(name).describe()}
