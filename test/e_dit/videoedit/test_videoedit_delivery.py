"""Delivery, original protocol variants, and bounded file processing."""
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from test_videoedit_api import configured, make_inputs, wait


def test_progress_terminal_callback_retry(tmp_path):
    messages = queue.Queue()
    attempts = []
    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            messages.put(body)
            if body['status'] == 'succeeded':
                attempts.append(body)
            self.send_response(503 if body['status'] == 'succeeded' and len(attempts) == 1 else 204)
            self.end_headers()
        def log_message(self, *args):
            pass
    http = ThreadingHTTPServer(('127.0.0.1', 0), Receiver)
    thread = threading.Thread(target=http.serve_forever, daemon=True); thread.start()
    server = configured(tmp_path)
    body = make_inputs(tmp_path/'inputs')
    body.update(prompt='slow', callbackUrl=f'http://127.0.0.1:{http.server_port}/callback')
    try:
        with TestClient(server.app) as client:
            client.post('/v1/videos/repairs', json=body)
            assert messages.get(timeout=5)['status'] == 'running'
            assert wait(client, 'one')['status'] == 'completed'
            for _ in range(200):
                value = client.get('/v1/videos/one/progress').json()
                if value['callback_status'] == 'succeeded':
                    break
                time.sleep(.05)
            assert value['callback_status'] == 'succeeded'
            assert value['callback_attempts'] == 2
            assert attempts[0] == attempts[1]
            assert set(attempts[0]) == {'status', 'progress', 'reason', 'output'}
            assert Path(json.loads(attempts[0]['output'])['gen_video_url']).is_file()
    finally:
        http.shutdown(); http.server_close(); thread.join()


def test_envelope_pagination_health_and_preparation_failure(tmp_path):
    server = configured(tmp_path, contract='backend')
    body = make_inputs(tmp_path/'inputs')
    with TestClient(server.app) as client:
        for identity in ('one', 'two'):
            body['taskId'] = identity
            result = client.post('/v1/videos/repairs', json={'inputs': body, 'parameters': {'seed': 3}})
            assert result.json()['code'] == 0, result.text
            assert wait(client, identity)['status'] == 'completed'
        page = client.get('/v1/videos', params={'order': 'asc', 'after': 'one', 'limit': 1}).json()
        assert [v['id'] for v in page['data']] == ['two']
        assert client.get('/v1/videos', params={'after': 'missing'}).json()['data'] == []
        assert client.get('/health').json()['queue']['completed'] == 2
        assert client.get('/admin/queue', params={'status': 'failed'}).json()['tasks'] == []
        body.update(taskId='missing', video_input_path='/missing/video.mp4')
        result = client.post('/v1/videos/repairs', json=body).json()
        assert result['code'] == 1
        assert client.get('/v1/videos/missing').json()['status'] == 'failed'


def test_encrypted_storage_and_streamed_object_delivery(tmp_path, monkeypatch):
    from nndeploy.e_dit.algorithms.wan.server import service
    uploads = []
    class Client:
        def upload_file(self, path, bucket, key, Callback, ExtraArgs):
            assert ExtraArgs['ContentType'] == 'video/x-msvideo'
            Callback(Path(path).stat().st_size)
            uploads.append((bucket, key))
        def close(self):
            pass
    monkeypatch.setattr(service, 'object_client', lambda config: (Client(), 'https://objects.example'))
    server = configured(tmp_path)
    body = make_inputs(tmp_path/'inputs')
    body.update(outputObjectKey='output/result.mp4', save_crop_only=True,
                minioConfig=dict(endpoint='objects.example', bucket='videos', rootUser='access', rootPass='secret-value'))
    with TestClient(server.app) as client:
        client.post('/v1/videos/repairs', json=body)
        result = wait(client, 'one')
        assert result['status'] == 'completed', result
        assert uploads == [('videos', 'output/result.avi'), ('videos', 'output/result_crop_only.avi')]
        record = server.db.external_task('videoedit', 'one', None, server.config_scope)
        assert 'secret-value' not in json.dumps(record)
        assert result['output_object_key'] == 'output/result.avi'


def test_postprocess_files_incrementally(tmp_path, monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from nndeploy.e_dit.algorithms.wan.server.postprocess import process_files
    from nndeploy.e_dit.algorithms.wan.server.schemas import VideoRepairRequest
    from nndeploy.e_dit.video.postprocess import rife_interpolator, realesrgan_upscaler
    from nndeploy.e_dit.algorithms.wan.video.stream_io import probe_video
    body = make_inputs(tmp_path/'inputs')
    seen = []
    def interpolate(self, frames, **kwargs):
        assert len(frames) == 2
        seen.append(len(frames))
        return [frames[0], frames[0], frames[1]], 2
    monkeypatch.setattr(rife_interpolator.FrameInterpolator, 'interpolate', interpolate)
    monkeypatch.setattr(realesrgan_upscaler.ImageUpscaler, 'upscale',
                        lambda self, frames: [np.repeat(np.repeat(f, 2, 0), 2, 1) for f in frames])
    request = VideoRepairRequest(prompt='edit', enable_frame_interpolation=True, enable_upscaling=True, upscaling_scale=2,
                                 preserve_audio=False)
    path = body['video_input_path']
    process_files([path], request, SimpleNamespace(directory=tmp_path, check=lambda: None, publish=lambda value: None))
    result = probe_video(path)
    assert (result['num_frame'], result['fps'], result['width'], result['height']) == (9, 10, 64, 64)
    assert len(seen) == 4


def test_http_runs_native_transformer_and_vae(tmp_path):
    from test_native_edit import tiny_graph
    from nndeploy.e_dit.algorithms.wan.server import add_videoedit
    from nndeploy.server import Server
    from nndeploy.base import StatusCode
    class Native:
        def init(self):
            self.graph, self.request = tiny_graph('wan_videoedit')
            return StatusCode.Ok
        def infer_files(self, *args, **kwargs):
            return self.graph.infer_files(*args, **kwargs)
        def deinit(self):
            self.graph.model.resources = None
            self.request.deinit()
    server = Server(resources=tmp_path/'server', data_dir=tmp_path/'server')
    add_videoedit(server, {'normal': {'device': 'cpu'}}, graph_factory=lambda variant, params: Native(),
                  execution={'isolation': 'thread'})
    body = make_inputs(tmp_path/'inputs')
    body.update(infer_len=5, overlap=1, refFrameIdx=2, bridgeOverlap=5, num_inference_steps=1,
                guidance_scale=1, enable_teacache=False, dilate_px=0, feather_px=0, preserveAudio=False,
                save_crop_only=True)
    with TestClient(server.app) as client:
        assert client.get('/health').json()['status'] == 'degraded_normal_only'
        accepted = client.post('/v1/videos/repairs', json=body)
        assert accepted.json()['code'] == 0, accepted.text
        result = wait(client, 'one')
        assert result['status'] == 'completed', result
        assert client.get('/v1/videos/one/content').status_code == 200
        record = server.db.external_task('videoedit', 'one', None, server.config_scope)
        assert len(record['outputs']['artifacts']) == 2


def test_normal_survives_dmd_initialization_failure(tmp_path):
    from test_videoedit_api import FakeGraph
    from nndeploy.e_dit.algorithms.wan.server import add_videoedit
    from nndeploy.server import Server
    def factory(variant, params):
        if variant == 'dmd':
            raise RuntimeError('Missing DMD weights')
        return FakeGraph()
    server = Server(resources=tmp_path/'server', data_dir=tmp_path/'server')
    add_videoedit(server, {'normal': {}, 'dmd': {}}, graph_factory=factory, execution={'isolation': 'thread'})
    body = make_inputs(tmp_path/'inputs')
    with TestClient(server.app) as client:
        health = client.get('/health').json()
        assert health['status'] == 'degraded_normal_only'
        assert health['backends'] == {'normal': True, 'dmd': False}
        assert client.post('/v1/videos/repairs', json={**body, 'model': 'dmd'}).status_code == 503
        assert client.post('/v1/videos/repairs', json=body).json()['code'] == 0
        assert wait(client, 'one')['status'] == 'completed'


def test_original_routes_respect_subject_permissions(tmp_path, monkeypatch):
    keys = []
    for name in ('alice', 'bob', 'reader'):
        monkeypatch.setenv('VIDEOEDIT_TEST_' + name, name + '-test-token-' + 'x' * 32)
        keys.append(dict(subject=name, token_env='VIDEOEDIT_TEST_' + name, entries=['videoedit'],
                         permissions=['tasks:read', 'files:read'] + ([] if name == 'reader' else ['deploy', 'tasks:cancel'])))
    server = configured(tmp_path, auth_config={'keys': keys})
    body = make_inputs(tmp_path/'inputs')
    headers = {name: {'Authorization': 'Bearer ' + name + '-test-token-' + 'x' * 32} for name in ('alice', 'bob', 'reader')}
    with TestClient(server.app) as client:
        assert client.post('/v1/videos/repairs', json=body).status_code == 401
        assert client.post('/v1/videos/repairs', json=body, headers=headers['reader']).status_code == 403
        assert client.post('/v1/videos/repairs', json=body, headers=headers['alice']).json()['code'] == 0
        assert wait(client, 'one', headers=headers['alice'])['status'] == 'completed'
        for path in ('/v1/videos/one', '/v1/videos/one/content', '/v1/videos/one/progress'):
            assert client.get(path, headers=headers['bob']).status_code == 404
        assert client.get('/v1/videos', headers=headers['bob']).json()['data'] == []
        assert client.get('/admin/queue', headers=headers['alice']).status_code == 403
        assert client.delete('/v1/videos/one', headers=headers['reader']).status_code == 403
        assert client.get('/v1/videos/one/content', headers=headers['alice']).status_code == 200
