"""Native graph, request boundaries and deterministic restoration contracts."""
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from nndeploy.e_dit import DiTParams, build_graph, build_workflow, model_names, server_config
from nndeploy.e_dit.algorithms.vsr.config import VSRConfig
from nndeploy.e_dit.algorithms.vsr.model import _resolve_weights
from nndeploy.e_dit.algorithms.vsr.resources import VSRResources
from nndeploy.dag import Graph


def test_registry_and_workflow_roundtrip(tmp_path):
    assert 'wan_vsr' in model_names()
    params = DiTParams('/checkpoint', model='wan_vsr', wan_root='/wan')
    graph = build_workflow(params, video='/input.mp4', output=str(tmp_path / 'output.mp4'),
                           sampling={'target_resolution': '64x96'})
    graph.save_file(str(tmp_path / 'workflow.json'))
    loaded = Graph('loaded')
    loaded.load_file(str(tmp_path / 'workflow.json'))
    model = loaded.get_node('wan_vsr')
    assert model.get_param().to_dict() == params.to_dict()
    assert loaded.get_node('video_input').str_ == '/input.mp4'
    assert loaded.get_node('sampling_input').sampling == {'target_resolution': '64x96'}
    assert server_config(model='wan_vsr')['inputs']['video']['edge'] == 'video'


@pytest.mark.parametrize('settings', [
    {'read_queue': 0}, {'write_queue': -1}, {'tile_t': 32}, {'tile_h': 31},
    {'temporal_overlap': 33}, {'spatial_overlap': 320}, {'color_ref': 'bad'},
    {'target_resolution': '63x64'}, {'crf': 52}, {'preserve_audio': 'false'},
    {'unknown': 1}, {'long_edge': 0},
])
def test_invalid_request(settings):
    with pytest.raises(ValueError):
        VSRConfig().request(settings)


def test_request_defaults_reset():
    config = VSRConfig(target_resolution='64x96')
    assert config.request({'long_edge': 128})['target_resolution'] is None
    assert config.request({'tile_t': 5, 'temporal_overlap': 1})['tile_t'] == 5
    assert config.request({})['tile_t'] == 33


def test_ema_never_pairs_online_decoder(tmp_path):
    (tmp_path / 'transformer_ema').mkdir()
    (tmp_path / 'transformer').mkdir()
    (tmp_path / 'vae_decoder.pt').touch()
    transformer, decoder, variant = _resolve_weights(tmp_path, '/unused')
    assert transformer.name == 'transformer_ema' and decoder is None and variant == 'ema'
    (tmp_path / 'vae_decoder_ema.pt').touch()
    assert _resolve_weights(tmp_path, '/unused')[1].endswith('vae_decoder_ema.pt')


class IdentityRestorer:
    device = torch.device('cpu')
    dtype = torch.float32
    def restore_window(self, window):
        return window


def test_real_dag_video_and_cancel(tmp_path):
    from test_audio import source_video, probe
    from nndeploy.dag.execution import ExecutionContext, execution_scope
    source = source_video(tmp_path)
    params = DiTParams('/checkpoint', model='wan_vsr', wan_root='/wan', device='cpu',
                       pipeline={'tile_t': 5, 'tile_h': 32, 'tile_w': 32,
                                 'temporal_overlap': 1, 'spatial_overlap': 0,
                                 'target_resolution': '32x32', 'color_ref': 'none'})
    def open_fake(resources):
        resources.restorer = IdentityRestorer()
    graph = build_graph(params)
    with patch.object(VSRResources, 'open', open_fake):
        graph.init()
    try:
        output = tmp_path / 'out.mp4'
        events = []
        with execution_scope(ExecutionContext(publish=events.append)):
            output.write_bytes(graph.infer(str(source)))
        streams = probe(output)['streams']
        assert any(s['codec_type'] == 'audio' for s in streams)
        assert graph.resources.last_metrics['frames_written'] == 20
        assert events[-1]['progress']['fraction'] == 1
    finally:
        graph.deinit()


def test_file_protection_and_cancel(tmp_path):
    from nndeploy.dag.execution import ExecutionContext, execution_scope
    from nndeploy.e_dit.nodes.control import RequestCancelled
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'original')
    resources = VSRResources(VSRConfig())
    resources.restorer = IdentityRestorer()
    with pytest.raises(ValueError, match='overwrite'):
        resources.restore(source, source, {})
    with execution_scope(ExecutionContext(cancelled=lambda: True)):
        with pytest.raises(RequestCancelled):
            resources.restore(source, tmp_path / 'out.mp4', {})
    assert resources.lock.acquire(blocking=False)
    resources.lock.release()
    assert source.read_bytes() == b'original'
