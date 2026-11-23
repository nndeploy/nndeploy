"""Resident native codec Graphs must process every frame and reset between files."""
import io
import json
import subprocess
import sys

import cv2
import nndeploy.base
import nndeploy.codec
import nndeploy.dag
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from nndeploy.server import Server
from test_db import db
from test_redis_queue import backend, redis_service


def video_graph():
    edge = nndeploy.dag.Edge('frame')
    graph = nndeploy.dag.Graph('video')
    decoder = nndeploy.codec.OpenCvVideoDecode('decode', [], [edge])
    encoder = nndeploy.codec.OpenCvVideoEncode('encode', [edge], [])
    # Native macOS OpenCV uses AVFoundation; Linux FFmpeg builds provide mp4v.
    encoder.set_fourcc('avc1' if sys.platform == 'darwin' else 'mp4v')
    graph.add_node(decoder)
    graph.add_node(encoder)
    return graph


def image_graph():
    edge = nndeploy.dag.Edge('image')
    graph = nndeploy.dag.Graph('image')
    graph.add_node(nndeploy.codec.OpenCvImageDecode('decode', [], [edge]))
    graph.add_node(nndeploy.codec.OpenCvImageEncode('encode', [edge], []))
    return graph


class ResizeFrames(nndeploy.dag.Node):
    def __init__(self, name, inputs, outputs):
        super().__init__(name, inputs, outputs)
        self.set_input_type(np.ndarray)
        self.set_output_type(np.ndarray)
        self.initializations = self.frames = 0

    def init(self):
        self.initializations += 1
        return nndeploy.base.Status.ok()

    def run(self):
        self.frames += 1
        frame = self.get_input(0).get(self)
        self.get_output(0).set(cv2.resize(255 - frame, (32, 32)))
        return nndeploy.base.Status.ok()


def resize_graph():
    source, target = nndeploy.dag.Edge('source'), nndeploy.dag.Edge('target')
    graph = nndeploy.dag.Graph('resize')
    graph.add_node(nndeploy.codec.OpenCvVideoDecode('decode', [], [source]))
    graph.add_node(ResizeFrames('resize', [source], [target]))
    encoder = nndeploy.codec.OpenCvVideoEncode('encode', [target], [])
    encoder.set_fourcc('avc1' if sys.platform == 'darwin' else 'mp4v')
    graph.add_node(encoder)
    return graph


class ChangingFrames(ResizeFrames):
    def run(self):
        frame = self.get_input(0).get(self)
        # Simulate an algorithm that changes shape partway through a clip.
        width = 16 if self.frames == 0 or frame[:, :, 2].mean() < 200 else 32
        self.frames += 1
        self.get_output(0).set(cv2.resize(frame, (width, width)))
        return nndeploy.base.Status.ok()


def changing_graph():
    source, target = nndeploy.dag.Edge('source'), nndeploy.dag.Edge('target')
    graph = nndeploy.dag.Graph('changing')
    graph.add_node(nndeploy.codec.OpenCvVideoDecode('decode', [], [source]))
    graph.add_node(ChangingFrames('change', [source], [target]))
    encoder = nndeploy.codec.OpenCvVideoEncode('encode', [target], [])
    encoder.set_fourcc('avc1' if sys.platform == 'darwin' else 'mp4v')
    graph.add_node(encoder)
    return graph


def clip(path, color, size, frames, fps=5):
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', f'color=c={color}:s={size}:r={fps}',
                    '-frames:v', str(frames), '-c:v', 'mpeg4', '-y', str(path)],
                   check=True, capture_output=True, timeout=15)
    return path.read_bytes()


def read_frames(path):
    capture = cv2.VideoCapture(str(path))
    try:
        values = []
        for _ in range(100):
            ok, value = capture.read()
            if not ok:
                return values
            values.append(value.copy())
        raise AssertionError('Unexpected video frame count')
    finally:
        capture.release()


@pytest.mark.parametrize('source', ['factory', 'json'])
@pytest.mark.parametrize('isolation', ['thread', 'process'])
def test_native_video_graph_runs_all_frames_and_reuses_models(db, tmp_path, source, isolation):
    workflow = json.loads(video_graph().serialize())
    original = json.dumps(workflow, sort_keys=True)
    settings = {'inputs': {'video': {'type': 'file', 'media_type': 'video/mp4', 'node': ['decode']}},
                'outputs': {'video': {'type': 'file', 'media_type': 'video/mp4', 'node': ['encode']}},
                'execution': {'isolation': isolation}}
    options = {'workflow': workflow} if source == 'json' else {'graph_factory': video_graph}
    server = Server(name='video', config=settings, resources=tmp_path / 'server', data_dir=tmp_path / 'server',
                    database_url=db.url.render_as_string(hide_password=False), **options)
    with TestClient(server.app) as client:
        worker = server.entries['video']['workers'][0]
        original_graph = worker.executor.runner.graph if isolation == 'thread' else None
        references = []
        for index, (color, size, count, shape) in enumerate([('red', '16x16', 3, (16, 16)), ('blue', '24x16', 5, (16, 24))]):
            data = clip(tmp_path / f'input-{index}.mp4', color, size, count)
            upload = client.post('/v1/files', files={'file': ('input.mp4', data, 'video/mp4')})
            assert upload.status_code == 201, upload.text
            result = client.post('/v1/server/video/deploy', json={'inputs': {'video': {'file_id': upload.json()['file_id']}}})
            assert result.status_code == 200, result.text
            identity = result.json()['outputs']['video']['file_id']
            references.append(identity)
            output = client.get(f'/v1/files/{identity}/content')
            path = tmp_path / f'output-{index}.mp4'
            path.write_bytes(output.content)
            frames = read_frames(path)
            assert len(frames) == count
            assert all(frame.shape[:2] == shape for frame in frames)
            channel = 2 if color == 'red' else 0
            assert all(frame[:, :, channel].mean() > 200 for frame in frames)
            assert server.entries['video']['workers'][0] is worker
            if original_graph is not None:
                assert worker.executor.runner.graph is original_graph
        assert len(set(references)) == 2
        assert json.dumps(workflow, sort_keys=True) == original
        assert not list((tmp_path / 'server' / 'tmp' / 'video').glob('request-*'))


@pytest.mark.parametrize('isolation', ['thread', 'process'])
def test_video_frames_pass_through_resident_algorithm(backend, tmp_path, isolation):
    database, redis_config = backend
    server = Server(name='video', graph_factory=resize_graph, resources=tmp_path / 'server', data_dir=tmp_path / 'server',
                    database_url=database.url.render_as_string(hide_password=False),
                    redis_config=redis_config, config={
                        'inputs': {'decode': {'type': 'file', 'media_type': 'video/mp4', 'node': ['decode']}},
                        'outputs': {'encode': {'type': 'file', 'media_type': 'video/mp4', 'node': ['encode']}},
                        'execution': {'isolation': isolation}})
    with TestClient(server.app) as client:
        for count in (3, 5):
            data = clip(tmp_path / 'input.mp4', 'red', '16x16', count)
            upload = client.post('/v1/files', files={'file': ('input.mp4', data, 'video/mp4')})
            assert upload.status_code == 201, upload.text
            result = client.post('/v1/server/video/deploy', json={'inputs': {'decode': {'file_id': upload.json()['file_id']}}})
            assert result.status_code == 200, result.text
            output = client.get(f"/v1/files/{result.json()['outputs']['encode']['file_id']}/content")
            path = tmp_path / 'output.mp4'
            path.write_bytes(output.content)
            frames = read_frames(path)
            assert len(frames) == count
            assert all(frame.shape == (32, 32, 3) for frame in frames)
            # Red was inverted to cyan by the algorithm on every frame.
            assert all(frame[:, :, :2].mean() > 200 and frame[:, :, 2].mean() < 30 for frame in frames)
        if isolation == 'thread':
            node = server.entries['video']['workers'][0].executor.runner.graph.get_node('resize')
            assert node.initializations == 1
            assert node.frames == 8
        assert not list((tmp_path / 'server' / 'tmp' / 'video').glob('request-*'))


def test_native_image_nodes_are_discovered_and_reused(tmp_path):
    with TestClient(Server(graph_factory=image_graph, resources=tmp_path, data_dir=tmp_path).app) as client:
        for color in ['red', 'blue']:
            data = io.BytesIO()
            Image.new('RGB', (5, 7), color).save(data, format='PNG')
            uploaded = client.post('/v1/files', files={'file': ('input.png', data.getvalue(), 'image/png')}).json()
            result = client.post('/v1/server/default/deploy', json={'inputs': {'decode': {'file_id': uploaded['file_id']}}})
            assert result.status_code == 200, result.text
            output = client.get(f"/v1/files/{result.json()['outputs']['encode']['file_id']}/content")
            image = Image.open(io.BytesIO(output.content))
            expected = (255, 0, 0) if color == 'red' else (0, 0, 255)
            assert image.size == (5, 7)
            assert image.getpixel((0, 0)) == expected


def test_native_decoder_does_not_repeat_last_frame_on_early_eof(tmp_path):
    path = tmp_path / 'short.mp4'
    clip(path, 'red', '16x16', 1)
    graph = video_graph()
    try:
        assert graph.init() == nndeploy.base.StatusCode.Ok
        decoder, encoder = graph.get_node('decode'), graph.get_node('encode')
        assert decoder.set_path(str(path)) == nndeploy.base.StatusCode.Ok
        # A damaged container can advertise more frames than can be decoded.
        decoder.set_size(2)
        encoder.set_fps(5)
        encoder.set_size(2)
        encoder.set_path(str(tmp_path / 'result.mp4'))
        assert graph.run() == nndeploy.base.StatusCode.Ok
        assert graph.run() != nndeploy.base.StatusCode.Ok
    finally:
        graph.deinit()


def test_native_video_encoder_rejects_frame_size_changes(tmp_path):
    edge = nndeploy.dag.Edge('frame')
    encoder = nndeploy.codec.OpenCvVideoEncode('encode', [edge], [])
    graph = nndeploy.dag.Graph('encode', [edge], [])
    graph.add_node(encoder)
    try:
        assert graph.init() == nndeploy.base.StatusCode.Ok
        encoder.set_fourcc('avc1' if sys.platform == 'darwin' else 'mp4v')
        encoder.set_fps(5)
        encoder.set_path(str(tmp_path / 'changed.mp4'))
        edge.set(np.zeros((16, 16, 3), dtype=np.uint8))
        assert graph.run() == nndeploy.base.StatusCode.Ok
        edge.set(np.zeros((32, 32, 3), dtype=np.uint8))
        assert graph.run() != nndeploy.base.StatusCode.Ok
    finally:
        graph.deinit()


def test_failed_video_request_discards_partial_output_and_next_request_recovers(tmp_path):
    from sqlalchemy import select, func
    from nndeploy.server.db import files
    server = Server(name='video', graph_factory=changing_graph, resources=tmp_path / 'server', data_dir=tmp_path / 'server')
    with TestClient(server.app) as client:
        for color, expected in [('red', 500), ('blue', 200)]:
            data = clip(tmp_path / 'input.mp4', color, '16x16', 3)
            upload = client.post('/v1/files', files={'file': ('input.mp4', data, 'video/mp4')})
            assert upload.status_code == 201, upload.text
            result = client.post('/v1/server/video/deploy', json={'inputs': {'decode': {'file_id': upload.json()['file_id']}}})
            assert result.status_code == expected, result.text
            assert not list((tmp_path / 'server' / 'tmp' / 'video').glob('request-*'))
            with server.db.engine.connect() as connection:
                assert connection.execute(select(func.count()).select_from(files)).scalar_one() == (1 if color == 'red' else 3)
            if color == 'blue':
                output = client.get(f"/v1/files/{result.json()['outputs']['encode']['file_id']}/content")
                path = tmp_path / 'result.mp4'
                path.write_bytes(output.content)
                assert len(read_frames(path)) == 3


@pytest.mark.parametrize('frames,fps', [(18001, 60), (3, 241)])
@pytest.mark.parametrize('isolation', ['thread', 'process'])
def test_video_request_limits_reject_input_and_allow_next_request(tmp_path, frames, fps, isolation):
    from nndeploy.server.config import suggest_config
    workflow = json.loads(video_graph().serialize())
    settings = suggest_config(workflow)
    settings['execution'] = {'isolation': isolation}
    server = Server(name='video', workflow=workflow, config=settings, resources=tmp_path / 'server', data_dir=tmp_path / 'server')
    with TestClient(server.app) as client:
        for count, rate, expected in [(frames, fps, 422), (3, 5, 200)]:
            data = clip(tmp_path / 'input.mp4', 'red', '16x16', count, rate)
            upload = client.post('/v1/files', files={'file': ('input.mp4', data, 'video/mp4')})
            assert upload.status_code == 201, upload.text
            result = client.post('/v1/server/video/deploy', json={'inputs': {'decode': {'file_id': upload.json()['file_id']}}})
            assert result.status_code == expected, result.text
            if expected == 422:
                assert result.json()['error']['code'] == 'invalid_media'
            assert not list((tmp_path / 'server' / 'tmp' / 'video').glob('request-*'))


@pytest.mark.parametrize('source', ['graph', 'json'])
@pytest.mark.parametrize('nested', [False, True])
def test_native_video_autodetect_preserves_draft_paths_without_opening_them(tmp_path, source, nested):
    draft_input = tmp_path / 'missing-draft.mp4'
    draft_output = tmp_path / 'keep.mp4'
    draft_output.write_bytes(b'original draft output')
    graph = video_graph()
    graph.get_node('decode').reset(str(draft_input))
    graph.get_node('encode').reset(str(draft_output), str(draft_input))
    if nested:
        parent = nndeploy.dag.Graph('outer')
        parent.add_node(graph)
        # A closed subgraph has no external data edges; schedule it through
        # Graph's existing control-order API, just as in ordinary DAG execution.
        assert parent.set_execution_order(['video'])
        graph = parent
    workflow = json.loads(graph.serialize())
    original = json.dumps(workflow, sort_keys=True)
    options = {'graph': graph} if source == 'graph' else {'workflow': workflow}
    server = Server(name='default', resources=tmp_path / 'server', data_dir=tmp_path / 'server', **options)
    with TestClient(server.app) as client:
        data = clip(tmp_path / 'input.mp4', 'red', '16x16', 3)
        upload = client.post('/v1/files', files={'file': ('input.mp4', data, 'video/mp4')})
        assert upload.status_code == 201, upload.text
        field = 'video_decode' if nested else 'decode'
        result = client.post('/v1/server/default/deploy', json={'inputs': {field: {'file_id': upload.json()['file_id']}}})
        assert result.status_code == 200, result.text
        running = server.entries['default']['workers'][0].executor.runner.graph
        if nested:
            running = running.get_node('video')
        for name, path in [('decode', draft_input), ('encode', draft_output)]:
            assert json.loads(running.get_node(name).serialize())['path_'] == str(path)
        assert running.get_node('decode').get_size() == 0
        assert running.get_node('decode').get_loop_count() == 1
        assert draft_output.read_bytes() == b'original draft output'
        assert not draft_input.exists()
        assert json.dumps(workflow, sort_keys=True) == original
