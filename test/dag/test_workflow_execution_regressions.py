"""Regressions from editor workflows: legacy YOLO ports and invalid inputs."""
import json

import numpy as np
import pytest
from PIL import Image

import nndeploy.base as base
import nndeploy.codec
import nndeploy.detect
from nndeploy.dag import Edge, Graph
from nndeploy.device import Tensor


@pytest.mark.parametrize('version', [5, 11, 26])
@pytest.mark.parametrize('legacy', [True, False])
def test_yolo_json_output_reaches_draw_node(version, legacy):
    prediction, image, boxes, rendered = [Edge(n) for n in ('prediction', 'image', 'boxes', 'rendered')]
    original = Graph('yolo', [prediction, image], [rendered])
    original.add_node(nndeploy.detect.YoloPostProcess('post', [prediction], [boxes]))
    original.add_node(nndeploy.detect.DrawBox('draw', [image, boxes], [rendered]))
    data = json.loads(original.serialize())
    post = data['node_repository_'][0]
    post['param_'].update(version_=version, num_classes_=1, model_w_=32, model_h_=32)
    post['outputs_'][0]['type_'] = 'DetectResult' if legacy else 'BBoxResult'
    if not legacy:
        data['node_repository_'][1]['key_'] = 'nndeploy::detect::DrawBBox'
        data['node_repository_'][1]['inputs_'][1]['type_'] = 'BBoxResult'
    graph = Graph('loaded')
    assert graph.deserialize(json.dumps(data)) == base.StatusCode.Ok
    assert graph.init() == base.StatusCode.Ok
    try:
        values = np.array([[[16, 16, 16, 16, .9, .9]]], dtype=np.float32)
        if version == 11:
            values = np.array([[[16], [16], [16], [16], [.9]]], dtype=np.float32)
        elif version == 26:
            values = np.array([[[8, 8, 24, 24, .9, 0]]], dtype=np.float32)
        graph.get_edge('prediction').set(Tensor.from_numpy(values))
        graph.get_edge('image').set(np.zeros((32, 32, 3), dtype=np.uint8))
        assert graph.run() == base.StatusCode.Ok
        assert np.asarray(graph.get_edge('rendered').get_graph_output()).any()
        # Saving a loaded legacy workflow must not silently change its port type.
        saved = json.loads(graph.serialize())['node_repository_'][0]
        assert saved['outputs_'][0]['type_'] == post['outputs_'][0]['type_']
    finally:
        graph.deinit()


def image_workflow(path, output, directory=False):
    edge = Edge('image')
    graph = Graph('images')
    decoder = nndeploy.codec.OpenCvImagesDecode if directory else nndeploy.codec.OpenCvImageDecode
    graph.add_node(decoder('decode', [], [edge]))
    graph.add_node(nndeploy.codec.OpenCvImageEncode('encode', [edge], []))
    data = json.loads(graph.serialize())
    data['node_repository_'][0]['path_'] = str(path)
    data['node_repository_'][1]['path_'] = str(output)
    return data


def test_decoder_deserialize_propagates_invalid_directory(tmp_path):
    path = tmp_path / 'input.jpg'
    Image.new('RGB', (8, 8)).save(path)
    graph = Graph('invalid')
    assert graph.deserialize(json.dumps(image_workflow(path, tmp_path / 'out.jpg', True))) != base.StatusCode.Ok


@pytest.mark.parametrize('parallel', ['kParallelTypeNone', 'kParallelTypePipeline'])
def test_directory_decoder_counts_images_and_rejects_empty_directory(tmp_path, parallel):
    node = nndeploy.codec.OpenCvImagesDecode('decode', [], [Edge('image')])
    node.set_parallel_type(base.ParallelType.Pipeline if parallel.endswith('Pipeline') else base.ParallelType.Sequential)
    assert node.set_path(str(tmp_path)) != base.StatusCode.Ok
    for suffix in ('jpg', 'png', 'bmp'):
        Image.new('RGB', (8, 8)).save(tmp_path / ('input.' + suffix))
    assert node.set_path(str(tmp_path)) == base.StatusCode.Ok
    assert node.get_size() == node.get_loop_count() == 3


@pytest.mark.parametrize('directory', [False, True])
def test_editor_path_failure_names_input_node(tmp_path, directory):
    from nndeploy.server.executor import GraphExecutor
    path = tmp_path / 'input.jpg'
    Image.new('RGB', (8, 8)).save(path)
    invalid = str(path) if directory else '\t' + str(path)
    with pytest.raises(ValueError) as caught:
        GraphExecutor(tmp_path).execute(image_workflow(invalid, tmp_path / 'out.jpg', directory), 'test')
    message = str(caught.value)
    assert 'decode' in message and 'path_' in message
    assert ('directory' if directory else '\\t') in message
    assert not (tmp_path / 'out.jpg').exists()
