"""General NumPy Graph ports retain rank and dtype without image coercion."""
import io
import json
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import nndeploy.base
import nndeploy.codec
import nndeploy.dag

from nndeploy.server import Server
from test_arrays import array_graph, graph_config, wire
from test_db import db
from test_redis_queue import backend, redis_service


@pytest.mark.parametrize('isolation', ['thread', 'process'])
@pytest.mark.parametrize('dtype,shape,data', [
    ('float32', [4], [1, 2, 3, 4]),
    ('float16', [2, 2, 1], [0.5, 1.5, 2.5, 3.5]),
    ('int64', [], [2**60]),
    ('uint64', [1, 1, 1, 2], [2**63, 2**64 - 1]),
    ('bool', [1, 1, 1, 1, 1, 1, 1, 2], [True, False]),
    ('int32', [0, 2, 1], []),
])
def test_general_numpy_graph_roundtrip_and_reused_buffers(tmp_path, isolation, dtype, shape, data):
    settings = graph_config('array', dtype=dtype, shape=shape, isolation=isolation)
    body = wire(data, dtype, shape)
    settings['execution']['warmup'] = [{'inputs': {'value': body}}]
    server = Server(graph_factory=array_graph, config=settings, data_dir=tmp_path)
    with TestClient(server.app) as client:
        first = client.post('/v1/server/default/deploy', json={'inputs': {'value': body}})
        assert first.status_code == 200, first.text
        assert first.json()['outputs']['value'] == body
        changed = wire([False if dtype == 'bool' else 0 for _ in data], dtype, shape)
        second = client.post('/v1/server/default/deploy', json={'inputs': {'value': changed}})
        assert second.status_code == 200, second.text
        assert second.json()['outputs']['value'] == changed
        task = client.get('/v1/tasks/' + first.json()['task_id']).json()
        assert task['outputs']['value'] == body


def test_numpy_dynamic_rank_dimensions_and_durable_database_values(db, tmp_path):
    settings = graph_config('array', dtype='int64', shape=[None, None, None, None])
    server = Server(graph_factory=array_graph, config=settings, data_dir=tmp_path,
                    database_url=db.url.render_as_string(hide_password=False))
    with TestClient(server.app) as client:
        for shape in ([1, 2, 3, 1], [2, 1, 2, 2], [1, 0, 3, 1]):
            body = wire(list(range(int(np.prod(shape)))), 'int64', shape)
            response = client.post('/v1/server/default/deploy', json={'inputs': {'value': body}})
            assert response.status_code == 200, response.text
            assert response.json()['outputs']['value'] == body
            assert db.get_task(response.json()['task_id'])['outputs']['value'] == body


def test_shared_process_numpy_graph_keeps_single_channel_dimension(backend, tmp_path):
    database, redis_config = backend
    settings = graph_config('array', dtype='float16', shape=[2, 3, 1], isolation='process')
    server = Server(graph_factory=array_graph, config=settings, data_dir=tmp_path,
                    database_url=database.url.render_as_string(hide_password=False), redis_config=redis_config)
    body = wire([.5, 1, 1.5, 2, 2.5, 3], 'float16', [2, 3, 1])
    with TestClient(server.app) as client:
        response = client.post('/v1/server/default/deploy', json={'inputs': {'value': body}})
        assert response.status_code == 200, response.text
        assert response.json()['outputs']['value'] == body


class VectorImage(nndeploy.dag.Node):
    def __init__(self, name, inputs, outputs):
        super().__init__(name, inputs, outputs)
        self.set_key('test.server.VectorImage')
        self.set_input_type(np.ndarray)
        self.set_output_type(np.ndarray)

    def run(self):
        value = self.get_input(0).get(self)
        assert value.shape == (12,)
        self.get_output(0).set(value.reshape(2, 2, 3))
        return nndeploy.base.Status.ok()


class VectorImageCreator(nndeploy.dag.NodeCreator):
    def create_node(self, name, inputs, outputs):
        self.node = VectorImage(name, inputs, outputs)
        return self.node


vector_image_creator = VectorImageCreator()
nndeploy.dag.register_node('test.server.VectorImage', vector_image_creator)


def vector_image_graph():
    source, target = nndeploy.dag.Edge('input'), nndeploy.dag.Edge('image')
    graph = nndeploy.dag.Graph('vector', [source], [])
    graph.add_node(VectorImage('reshape', [source], [target]))
    graph.add_node(nndeploy.codec.OpenCvImageEncode('encode', [target], []))
    return graph


@pytest.mark.parametrize('source', ['factory', 'json'])
def test_numpy_vector_flows_into_native_opencv_encoder(tmp_path, source):
    workflow = json.loads(vector_image_graph().serialize())
    original = json.dumps(workflow, sort_keys=True)
    options = {'workflow': workflow} if source == 'json' else {'graph_factory': vector_image_graph}
    settings = {'inputs': {'value': {'type': 'array', 'dtype': 'uint8', 'shape': [12], 'edge': 'input'}},
                'outputs': {'image': {'type': 'image', 'node': ['encode']}}}
    server = Server(name='vector', config=settings, data_dir=tmp_path, **options)
    with TestClient(server.app) as client:
        for bgr, rgb in [([0, 0, 255], (255, 0, 0)), ([255, 0, 0], (0, 0, 255))]:
            result = client.post('/v1/server/vector/deploy', json={'inputs': {'value': wire(bgr * 4, 'uint8', [12])}})
            assert result.status_code == 200, result.text
            output = client.get(f"/v1/files/{result.json()['outputs']['image']['file_id']}/content")
            with Image.open(io.BytesIO(output.content)) as image:
                assert image.size == (2, 2)
                assert image.getpixel((0, 0)) == rgb
        assert json.dumps(workflow, sort_keys=True) == original
