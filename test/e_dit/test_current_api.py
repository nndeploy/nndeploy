"""Current API works without historical exports, env aliases or no-op shims."""
import importlib
import json
from pathlib import Path

import pytest
import torch

import nndeploy.e_dit as edit


@pytest.mark.parametrize('name', ['compatibility', 'state', 'utils.distributed_runtime', 'utils.profiler', 'layers.custom_op'])
def test_removed_modules(name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module('nndeploy.e_dit.' + name)


@pytest.mark.parametrize('module,names', [
    ('nndeploy.e_dit', ('DiTGraph', 'EraserDiTNode', 'DiTState', 'EraserDiTEraseSamplingParams')),
    ('nndeploy.e_dit.graph', ('DiTGraph', 'EraserDiTNode')),
    ('nndeploy.e_dit.workflow', ('SamplingInput', 'VideoOutput')),
])
def test_removed_exports(module, names):
    value = importlib.import_module(module)
    for name in names:
        with pytest.raises(AttributeError):
            getattr(value, name)


def test_current_state_and_environment(monkeypatch):
    from nndeploy.e_dit.nodes.schedule_batch import Req
    from nndeploy.e_dit.utils.env import get_env
    from nndeploy.e_dit.utils import cpu_resources
    assert edit.Req is Req
    monkeypatch.delenv('NNDEPLOY_DIT_CPU_THREADS_PER_RANK', raising=False)
    monkeypatch.setenv('MGERASE_CPU_THREADS_PER_RANK', '9')
    assert get_env('NNDEPLOY_DIT_CPU_THREADS_PER_RANK') is None
    monkeypatch.setenv('NNDEPLOY_DIT_CPU_THREADS_PER_RANK', '2')
    assert get_env('NNDEPLOY_DIT_CPU_THREADS_PER_RANK') == '2'
    assert cpu_resources._physical_gpu_id({'MGERASE_PHYSICAL_GPU_IDS': '7'}, 0) == 0
    assert cpu_resources._physical_gpu_id({'NNDEPLOY_DIT_PHYSICAL_GPU_IDS': '7'}, 0) == 7


@pytest.mark.parametrize('name', edit.model_names())
def test_current_factories_and_json_roundtrip(name, tmp_path):
    from nndeploy.dag import Graph
    params = edit.DiTParams(model=name, device='cpu')
    params.build_config()
    graph = edit.build_graph(params)
    assert graph is not None
    kwargs = dict(video='/tmp/video.mp4', mask='/tmp/mask.mp4', output='/tmp/result.mp4') if name == 'eraserdit' else {}
    workflow = edit.build_workflow(params, **kwargs)
    file = tmp_path / 'workflow.json'
    workflow.save_file(str(file))
    loaded = Graph('loaded')
    loaded.load_file(str(file))
    def compare(a, b, path='nodes'):
        # Native InputStr serialization may repeat required parameter names;
        # deserialization deduplicates this metadata without changing the graph.
        if path.endswith('/required_params_'):
            assert set(a) == set(b), path
            return
        if isinstance(a, dict) and isinstance(b, dict):
            assert a.keys() == b.keys(), path
            for key in a: compare(a[key], b[key], path + '/' + key)
        elif isinstance(a, list) and isinstance(b, list):
            assert len(a) == len(b), path
            for i, (x, y) in enumerate(zip(a, b)): compare(x, y, path + '/' + str(i))
        else:
            assert a == b, (path, a, b)
    compare(json.loads(loaded.serialize())['node_repository_'], json.loads(workflow.serialize())['node_repository_'])


def test_no_legacy_environment_literals():
    import ast
    root = Path(edit.__file__).parent
    for path in root.rglob('*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not node.value.startswith(('MGERASE_', 'SGLANG_VIDEOEDIT_', 'ERASERDIT_DEBUG_DUMP')), path


def test_ltx_transformer_without_placeholder_cache():
    from nndeploy.e_dit.algorithms.ltx2.configs.ltx_2 import LTX2Config
    from nndeploy.e_dit.algorithms.ltx2.models.ltx_2 import LTX2VideoTransformer3DModel
    from nndeploy.e_dit.config.server_args import config_scope
    from nndeploy.e_dit.models.dits.base import BaseDiT
    config = LTX2Config()
    config.update_model_arch(dict(num_attention_heads=2, attention_head_dim=12,
        in_channels=4, out_channels=4, num_layers=1, cross_attention_dim=24, caption_channels=24,
        audio_num_attention_heads=2, audio_attention_head_dim=12,
        audio_in_channels=4, audio_out_channels=4, audio_cross_attention_dim=24, patch_size=(1,1,1)))
    args = edit.DiTParams(model='ltx2', device='cpu').build_config()
    with config_scope(args), torch.inference_mode():
        model = LTX2VideoTransformer3DModel(config, {}).eval()
        assert isinstance(model, BaseDiT)
        assert not hasattr(model, '_init_teacache_state')
        inputs = dict(hidden_states=torch.randn(1,2,4), audio_hidden_states=torch.randn(1,2,4),
            encoder_hidden_states=torch.randn(1,3,24), audio_encoder_hidden_states=torch.randn(1,3,24),
            timestep=torch.tensor([0.5]), num_frames=2, height=1, width=1, audio_num_frames=2,
            return_latents=False)
        result = model(**inputs)
        assert all(torch.isfinite(tensor).all() for tensor in result)
        for a, b in zip(result, model(**inputs)):
            torch.testing.assert_close(a, b)


@pytest.mark.parametrize('name', ['wan', 'wan_dmd'])
def test_wan_native_request_without_profiler_shim(name):
    from types import SimpleNamespace
    import numpy as np
    from nndeploy.e_dit.registry import get_model
    from nndeploy.e_dit.config.server_args import config_scope
    from nndeploy.e_dit.algorithms.wan.models.wanvideo import WanTransformer3DModel
    from nndeploy.e_dit.algorithms.wan.nodes.generation import WanRequestGraph
    from nndeploy.e_dit.algorithms.wan.state import WanState
    from nndeploy.e_dit.models.schedulers.scheduling_flow_unipc_multistep import FlowUniPCMultistepScheduler
    from nndeploy.e_dit.models.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    class Text(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(8, 16)
        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(last_hidden_state=self.emb(input_ids))
    class Tokenizer:
        def __call__(self, texts, **kwargs):
            return dict(input_ids=torch.ones(len(texts),4,dtype=torch.long), attention_mask=torch.ones(len(texts),4,dtype=torch.long))
    class VAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))
        def decode(self, x):
            return torch.nn.functional.interpolate(x[:,:3],scale_factor=(1,8,8)).tanh()
    adapter = get_model(name)
    args = edit.DiTParams(model=name, device='cpu').build_config()
    args.pipeline_config.dit_config.update_model_arch(dict(num_attention_heads=2, attention_head_dim=12,
        in_channels=4,out_channels=4,text_dim=16,freq_dim=16,ffn_dim=48,num_layers=1))
    args.pipeline_config.vae_config.update_model_arch(dict(z_dim=4,latents_mean=[0]*4,latents_std=[1]*4))
    with config_scope(args), torch.inference_mode():
        model = WanTransformer3DModel(args.pipeline_config.dit_config,{})
        scheduler_cls = FlowMatchEulerDiscreteScheduler if name.endswith('dmd') else FlowUniPCMultistepScheduler
        resources = SimpleNamespace(config=args,adapter=adapter,modules=dict(transformer=model,
            scheduler=scheduler_cls(shift=args.pipeline_config.flow_shift),vae=VAE(),text_encoder=Text(),tokenizer=Tokenizer()))
        graph = WanRequestGraph()
        graph.bind(resources)
        graph.init()
        outputs = []
        try:
            for _ in range(2):
                sampling = adapter.sampling(dict(prompt='test',height=32,width=32,num_frames=9,num_inference_steps=2,seed=1))
                batch = WanState(sampling_params=sampling)
                batch.generator = [torch.Generator().manual_seed(1)]
                batch.do_classifier_free_guidance = name == 'wan'
                outputs.append(graph.invoke(batch).extra['result'].output.copy())
            np.testing.assert_array_equal(*outputs)
        finally:
            graph.deinit()
