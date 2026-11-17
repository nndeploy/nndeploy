"""Media conversion through real resident Graph ports and HTTP files."""
import io
import wave

import nndeploy.base
import nndeploy.dag
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
import pytest

from nndeploy.server import Server
from test_db import db
from test_redis_queue import backend, redis_service
from test_storage import s3_service


class MediaCopy(nndeploy.dag.Node):
    def __init__(self, name, inputs, outputs, kind):
        super().__init__(name, inputs, outputs)
        self.set_key('test.MediaCopy')
        self.set_input_type(kind)
        self.set_output_type(kind)
    def run(self):
        self.get_output(0).set(self.get_input(0).get(self))
        return nndeploy.base.Status.ok()


def media_graph(kind):
    source, target = nndeploy.dag.Edge('input'), nndeploy.dag.Edge('output')
    graph = nndeploy.dag.Graph('media', [source], [target])
    graph.add_node(MediaCopy('copy', [source], [target], kind))
    return graph


def byte_graph():
    return media_graph(bytes)


def pil_graph():
    return media_graph(Image.Image)


def cv_graph():
    return media_graph(np.ndarray)


def config(kind, media_type=None):
    field = {'type': kind}
    if media_type:
        field['media_type'] = media_type
    return {'inputs': {'media': dict(field, edge='input')}, 'outputs': {'media': dict(field, edge='output')}}


def wav_bytes():
    data = io.BytesIO()
    with wave.open(data, 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(b'\x00\x00' * 800)
    return data.getvalue()


@pytest.mark.parametrize('factory', [pil_graph, cv_graph])
def test_image_ports_roundtrip_real_pixels_and_keep_graph_resident(db, tmp_path, factory):
    server = Server(graph_factory=factory, config=config('image'), data_dir=tmp_path,
                    database_url=db.url.render_as_string(hide_password=False))
    with TestClient(server.app) as client:
        graph = server.entries['default']['workers'][0].executor.runner.graph
        references = []
        for color, size in [((7, 32, 199), (5, 7)), ((99, 4, 12), (3, 4))]:
            data = io.BytesIO()
            Image.new('RGB', size, color).save(data, format='PNG')
            uploaded = client.post('/v1/files', files={'file': ('input.png', data.getvalue(), 'image/png')})
            assert uploaded.status_code == 201, uploaded.text
            result = client.post('/v1/server/default/deploy', json={'inputs': {'media': {'file_id': uploaded.json()['file_id']}}})
            assert result.status_code == 200, result.text
            reference = result.json()['outputs']['media']['file_id']
            references.append(reference)
            content = client.get(f'/v1/files/{reference}/content')
            image = Image.open(io.BytesIO(content.content))
            assert image.size == size and image.getpixel((0, 0)) == color
            assert server.entries['default']['workers'][0].executor.runner.graph is graph
        assert references[0] != references[1]
        assert Image.open(io.BytesIO(client.get(f'/v1/files/{references[0]}/content').content)).size == (5, 7)


def test_audio_bytes_ports_validate_mime_and_publish_independent_files(db, tmp_path):
    server = Server(graph_factory=byte_graph, config=config('file', 'audio/wav'), data_dir=tmp_path,
                    database_url=db.url.render_as_string(hide_password=False))
    data = wav_bytes()
    with TestClient(server.app) as client:
        uploaded = client.post('/v1/files', files={'file': ('input.wav', data, 'audio/wav')})
        assert uploaded.status_code == 201, uploaded.text
        ids = []
        for _ in range(2):
            result = client.post('/v1/server/default/deploy', json={'inputs': {'media': {'file_id': uploaded.json()['file_id']}}})
            assert result.status_code == 200, result.text
            identity = result.json()['outputs']['media']['file_id']
            ids.append(identity)
            content = client.get(f'/v1/files/{identity}/content')
            assert content.headers['content-type'].startswith('audio/wav')
            assert content.content == data
        assert ids[0] != ids[1]
        wrong = client.post('/v1/files', files={'file': ('file.bin', data, 'application/octet-stream')})
        rejected = client.post('/v1/server/default/deploy', json={'inputs': {'media': {'file_id': wrong.json()['file_id']}}})
        assert rejected.status_code == 422


class ReadFile(nndeploy.dag.Node):
    def __init__(self, name, inputs, outputs):
        super().__init__(name, inputs, outputs)
        self.set_key('test.ReadFile')
        self.set_input_type(str)
        self.set_output_type(bytes)
    def run(self):
        from pathlib import Path
        self.get_output(0).set(Path(self.get_input(0).get(self)).read_bytes())
        return nndeploy.base.Status.ok()


def path_graph():
    source, target = nndeploy.dag.Edge('input'), nndeploy.dag.Edge('output')
    graph = nndeploy.dag.Graph('media', [source], [target])
    graph.add_node(ReadFile('read', [source], [target]))
    return graph


@pytest.mark.parametrize('isolation', ['thread', 'process'])
@pytest.mark.parametrize('factory', [byte_graph, path_graph])
def test_video_and_text_file_ports_roundtrip_with_explicit_content_type(db, tmp_path, factory, isolation):
    import subprocess
    path = tmp_path / 'small.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=blue:s=16x16:r=5',
                    '-t', '0.4', '-c:v', 'mpeg4', '-y', str(path)], check=True, capture_output=True, timeout=15)
    for media_type, data in [('video/mp4', path.read_bytes()), ('text/plain', b'hello UTF-8')]:
        settings = config('file', media_type)
        settings['execution'] = {'isolation': isolation}
        server = Server(graph_factory=factory, config=settings, data_dir=tmp_path / media_type.replace('/', '-'),
                        database_url=db.url.render_as_string(hide_password=False))
        with TestClient(server.app) as client:
            uploaded = client.post('/v1/files', files={'file': ('input', data, media_type)})
            assert uploaded.status_code == 201, uploaded.text
            for _ in range(2):
                result = client.post('/v1/server/default/deploy', json={'inputs': {'media': {'file_id': uploaded.json()['file_id']}}})
                assert result.status_code == 200, result.text
                output = client.get(f"/v1/files/{result.json()['outputs']['media']['file_id']}/content")
                assert output.headers['content-type'].startswith(media_type)
                assert output.content == data
            local_input = tmp_path / ('local.mp4' if media_type == 'video/mp4' else 'local.txt')
            local_input.write_bytes(data)
            local = client.post('/v1/server/default/deploy', json={'inputs': {'media': str(local_input)}})
            assert local.status_code == 200, local.text
            content = client.get(f"/v1/files/{local.json()['outputs']['media']['file_id']}/content")
            assert content.content == data
            assert client.post('/v1/server/default/deploy',
                json={'inputs': {'media': str(tmp_path / 'missing')}}).status_code == 422


def test_media_output_format_and_original_workflow_stay_independent(db, tmp_path):
    import copy
    from test_server_api import image_graph
    workflow = image_graph()
    original = copy.deepcopy(workflow)
    settings = {'inputs': {'image': {'node': ['decode'], 'type': 'image'}},
                'outputs': {'image': {'node': ['encode'], 'type': 'image', 'media_type': 'image/jpeg'}}}
    server = Server(workflow=workflow, config=settings, data_dir=tmp_path,
                    database_url=db.url.render_as_string(hide_password=False))
    with TestClient(server.app) as client:
        image = io.BytesIO()
        Image.new('RGB', (8, 6), (31, 72, 154)).save(image, format='PNG')
        uploaded = client.post('/v1/files', files={'file': ('image.png', image.getvalue(), 'image/png')}).json()
        response = client.post('/v1/server/default/deploy', json={'inputs': {'image': {'file_id': uploaded['file_id']}}})
        assert response.status_code == 200, response.text
        output = client.get(f"/v1/files/{response.json()['outputs']['image']['file_id']}/content")
        assert output.headers['content-type'] == 'image/jpeg'
        decoded = Image.open(io.BytesIO(output.content))
        assert decoded.format == 'JPEG' and decoded.size == (8, 6)
        assert workflow == original
        assert server.entries['default']['workers'][0].executor.runner.graph.get_node('encode').path_ == 'unused.png'


@pytest.mark.parametrize('value', [
    {'type': 'string', 'media_type': 'text/plain'},
    {'type': 'image', 'media_type': 'audio/wav'},
    {'type': 'file', 'media_type': 'application/x-unknown'},
])
def test_invalid_media_declarations_are_rejected(value):
    from nndeploy.server.config import FieldConfig
    with pytest.raises(ValueError):
        FieldConfig.model_validate(value)


def test_optional_mime_constraint_does_not_change_existing_config_fingerprints():
    from nndeploy.server.config import FieldConfig
    assert 'media_type' not in FieldConfig(type='image').model_dump()
    assert 'media_type' not in FieldConfig(type='file').model_dump()


def test_code_output_reference_must_match_declared_mime_without_deleting_input(tmp_path):
    settings = {'inputs': {'media': {'type': 'file'}},
                'outputs': {'media': {'type': 'file', 'media_type': 'audio/wav'}}}
    server = Server(deploy=lambda values: values, config=settings, data_dir=tmp_path)
    with TestClient(server.app) as client:
        uploaded = client.post('/v1/files', files={'file': ('text.txt', b'not audio', 'text/plain')}).json()
        result = client.post('/v1/server/default/deploy', json={'inputs': {'media': {'file_id': uploaded['file_id']}}})
        assert result.status_code == 500, result.text
        assert client.get(f"/v1/files/{uploaded['file_id']}/content").content == b'not audio'


@pytest.mark.parametrize('factory,kind', [(pil_graph, 'image'), (byte_graph, 'file')])
def test_known_exposed_media_ports_are_discovered_without_extra_configuration(tmp_path, factory, kind):
    server = Server(graph_factory=factory, data_dir=tmp_path)
    with TestClient(server.app) as client:
        schema = server.describe('default')['schema']
        assert schema['inputs']['input']['type'] == kind
        assert schema['outputs']['output']['type'] == kind
        if kind == 'image':
            output = io.BytesIO()
            Image.new('RGB', (4, 3), 'red').save(output, format='PNG')
            data, mime = output.getvalue(), 'image/png'
        else:
            data, mime = b'byte payload', 'application/octet-stream'
        uploaded = client.post('/v1/files', files={'file': ('input', data, mime)}).json()
        result = client.post('/v1/server/default/deploy', json={'inputs': {'input': {'file_id': uploaded['file_id']}}})
        assert result.status_code == 200, result.text
        assert client.get(f"/v1/files/{result.json()['outputs']['output']['file_id']}/content").content == data


class Amplify(MediaCopy):
    def run(self):
        value = self.get_input(0).get(self)
        self.get_output(0).set(value * 100 if value == b'expand' else value)
        return nndeploy.base.Status.ok()


def amplify_graph():
    source, target = nndeploy.dag.Edge('input'), nndeploy.dag.Edge('output')
    graph = nndeploy.dag.Graph('media', [source], [target])
    graph.add_node(Amplify('copy', [source], [target], bytes))
    return graph


def test_file_output_size_limit_fails_without_publication_and_graph_recovers(tmp_path):
    server = Server(graph_factory=amplify_graph, config=config('file'), data_dir=tmp_path,
                    storage_config={'max_bytes': 128})
    with TestClient(server.app) as client:
        for data, expected in [(b'expand', 500), (b'ok', 200)]:
            uploaded = client.post('/v1/files', files={'file': ('input.bin', data, 'application/octet-stream')})
            assert uploaded.status_code == 201, uploaded.text
            result = client.post('/v1/server/default/deploy', json={'inputs': {'media': {'file_id': uploaded.json()['file_id']}}})
            assert result.status_code == expected, result.text
        assert not list((tmp_path / 'tmp' / 'default').glob('request-*'))


class InvalidAudio(MediaCopy):
    def run(self):
        self.get_output(0).set(b'not an audio container')
        return nndeploy.base.Status.ok()


def invalid_audio_graph():
    source, target = nndeploy.dag.Edge('input'), nndeploy.dag.Edge('output')
    graph = nndeploy.dag.Graph('media', [source], [target])
    graph.add_node(InvalidAudio('copy', [source], [target], bytes))
    return graph


def test_file_output_must_validate_actual_media_not_just_extension(tmp_path):
    from sqlalchemy import select, func
    from nndeploy.server.db import files
    server = Server(graph_factory=invalid_audio_graph, config=config('file', 'audio/wav'), data_dir=tmp_path)
    with TestClient(server.app) as client:
        uploaded = client.post('/v1/files', files={'file': ('input.wav', wav_bytes(), 'audio/wav')}).json()
        result = client.post('/v1/server/default/deploy', json={'inputs': {'media': {'file_id': uploaded['file_id']}}})
        assert result.status_code == 422, result.text
        with server.db.engine.connect() as connection:
            assert connection.execute(select(func.count()).select_from(files)).scalar_one() == 1
        assert not list((tmp_path / 'tmp' / 'default').glob('request-*'))


def test_shared_graph_audio_ports_work_through_process_worker(backend, tmp_path):
    database, redis_config = backend
    settings = config('file', 'audio/wav')
    settings['execution'] = {'isolation': 'process'}
    server = Server(graph_factory=byte_graph, config=settings, data_dir=tmp_path,
                    database_url=database.url.render_as_string(hide_password=False), redis_config=redis_config)
    data = wav_bytes()
    with TestClient(server.app) as client:
        uploaded = client.post('/v1/files', files={'file': ('input.wav', data, 'audio/wav')}).json()
        response = client.post('/v1/server/default/deploy', json={'inputs': {'media': {'file_id': uploaded['file_id']}}})
        assert response.status_code == 200, response.text
        output = client.get(f"/v1/files/{response.json()['outputs']['media']['file_id']}/content")
        assert output.content == data and output.headers['content-type'].startswith('audio/wav')
        assert database.get_task(response.json()['task_id'])['state'] == 'succeeded'


class ReadFileCreator(nndeploy.dag.NodeCreator):
    def create_node(self, name, inputs, outputs):
        # Keep the Python trampoline alive until Graph.create_node retains it,
        # matching the repository's Python NodeCreator implementations.
        self.node = ReadFile(name, inputs, outputs)
        return self.node


read_file_creator = ReadFileCreator()
nndeploy.dag.register_node('test.ReadFile', read_file_creator)


def test_workflow_json_file_binding_reuses_existing_input_node_and_restores_value(db, tmp_path):
    import json
    from nndeploy.basic.input import InputStr
    source, target = nndeploy.dag.Edge('path'), nndeploy.dag.Edge('output')
    graph = nndeploy.dag.Graph('files', [], [target])
    node = InputStr('source', [], [source])
    node.str_ = 'unused-original-value'
    graph.add_node(node)
    graph.add_node(ReadFile('reader', [source], [target]))
    path = tmp_path / 'workflow.json'
    original = graph.serialize()
    path.write_text(original)
    settings = config('file', 'audio/wav')
    settings['inputs']['media'].pop('edge')
    settings['inputs']['media']['node'] = ['source']
    server = Server(workflow=path, config=settings, data_dir=tmp_path / 'server',
                    database_url=db.url.render_as_string(hide_password=False))
    data = wav_bytes()
    with TestClient(server.app) as client:
        uploaded = client.post('/v1/files', files={'file': ('input.wav', data, 'audio/wav')}).json()
        for _ in range(2):
            response = client.post('/v1/server/default/deploy', json={'inputs': {'media': {'file_id': uploaded['file_id']}}})
            assert response.status_code == 200, response.text
            assert client.get(f"/v1/files/{response.json()['outputs']['media']['file_id']}/content").content == data
            assert server.entries['default']['workers'][0].executor.runner.graph.get_node('source').str_ == 'unused-original-value'
        assert path.read_text() == original


def test_media_graph_restores_input_and_output_from_object_storage(db, s3_service, tmp_path, monkeypatch):
    import uuid
    endpoint, environment, client = s3_service
    monkeypatch.setenv('NNDEPLOY_MEDIA_TEST_ACCESS', environment['MINIO_ROOT_USER'])
    monkeypatch.setenv('NNDEPLOY_MEDIA_TEST_SECRET', environment['MINIO_ROOT_PASSWORD'])
    storage = dict(backend='s3', bucket='nndeploy-tests', region='us-east-1', endpoint=endpoint,
                   prefix='media-' + uuid.uuid4().hex,
                   access_key_env='NNDEPLOY_MEDIA_TEST_ACCESS', secret_key_env='NNDEPLOY_MEDIA_TEST_SECRET')
    server = Server(graph_factory=byte_graph, config=config('file', 'audio/wav'), data_dir=tmp_path,
                    database_url=db.url.render_as_string(hide_password=False), storage_config=storage)
    data = wav_bytes()
    with TestClient(server.app) as http:
        uploaded = http.post('/v1/files', files={'file': ('input.wav', data, 'audio/wav')}).json()
        cached, _ = server.storage.resolve(uploaded['file_id'])
        cached.unlink()
        response = http.post('/v1/server/default/deploy', json={'inputs': {'media': {'file_id': uploaded['file_id']}}})
        assert response.status_code == 200, response.text
        identity = response.json()['outputs']['media']['file_id']
        cached, record = server.storage.resolve(identity)
        cached.unlink()
        assert http.get(f'/v1/files/{identity}/content').content == data
        key = server.storage.backend.key(record['storage_key'])
        assert client.head_object(Bucket='nndeploy-tests', Key=key)['ContentLength'] == len(data)


def two_file_outputs_graph():
    source = nndeploy.dag.Edge('input')
    first, second = nndeploy.dag.Edge('first'), nndeploy.dag.Edge('second')
    graph = nndeploy.dag.Graph('media', [source], [first, second])
    graph.add_node(MediaCopy('copy1', [source], [first], bytes))
    graph.add_node(MediaCopy('copy2', [source], [second], bytes))
    return graph


def test_invalid_second_output_removes_already_published_first_output(tmp_path):
    from sqlalchemy import select, func
    from nndeploy.server.db import files
    settings = {'inputs': {'media': {'type': 'file', 'edge': 'input'}},
                'outputs': {'first': {'type': 'file', 'edge': 'first'},
                            'second': {'type': 'file', 'edge': 'second', 'media_type': 'audio/wav'}}}
    server = Server(graph_factory=two_file_outputs_graph, config=settings, data_dir=tmp_path)
    with TestClient(server.app) as client:
        uploaded = client.post('/v1/files', files={'file': ('input.bin', b'not audio', 'application/octet-stream')}).json()
        response = client.post('/v1/server/default/deploy', json={'inputs': {'media': {'file_id': uploaded['file_id']}}})
        assert response.status_code == 422, response.text
        with server.db.engine.connect() as connection:
            assert connection.execute(select(func.count()).select_from(files)).scalar_one() == 1
        assert client.get(f"/v1/files/{uploaded['file_id']}/content").content == b'not audio'
        assert not list((tmp_path / 'tmp' / 'default').glob('request-*'))
