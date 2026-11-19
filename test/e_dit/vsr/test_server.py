"""Compatibility routes on real nndeploy Workers, without production weights."""
import asyncio
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from nndeploy.e_dit import DiTParams
from nndeploy.server import Server
from nndeploy.e_dit.algorithms.vsr.server import add_vsr
from nndeploy.e_dit.algorithms.vsr.server.handler import VSRHandler


class FakeHandler(VSRHandler):
    def init(self):
        with (Path(self.params['wan_root']) / 'model_loads').open('a') as marker:
            marker.write('loaded\n')
        def infer(video, output, **sampling):
            from nndeploy.dag.execution import current_execution
            delay = 1.3 if 'slow' in Path(video).name else .12
            deadline = time.monotonic() + delay
            while time.monotonic() < deadline:
                current_execution().cancelled()
                time.sleep(.01)
            shutil.copyfile(video, output)
            return str(output)
        self.graph = SimpleNamespace(infer_file=infer, deinit=lambda: None)


@pytest.fixture
def video(tmp_path):
    path = tmp_path / 'input.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    'testsrc2=size=32x32:rate=5:duration=1', '-c:v', 'libx264', str(path)], check=True)
    return path


def server(tmp_path, isolation='thread'):
    instance = Server(resources=tmp_path / 'server', data_dir=tmp_path / 'server')
    return add_vsr(instance, DiTParams(model='wan_vsr', wan_root=str(tmp_path)),
                   isolation=isolation, handler_factory=FakeHandler)


def wait(client, identity):
    for _ in range(600):
        value = client.get(f'/v1/videos/{identity}').json()
        if value['status'] in ('completed', 'failed', 'cancelled', 'expired'):
            return value
        time.sleep(.01)
    raise AssertionError(f'Task {identity} did not terminate: {value}')


def submit(client, video, identity='one', **kwargs):
    return client.post('/v1/videos/restorations', json={
        'taskId': identity, 'video_input_path': str(video), 'target_resolution': '32x32', **kwargs})


@pytest.mark.parametrize('isolation', ['thread', 'process'])
def test_lifecycle_and_download(tmp_path, video, isolation):
    service = server(tmp_path, isolation)
    with TestClient(service.app) as client:
        assert client.get('/health').json() == {'status': 'ok'}
        assert submit(client, video).json() == {'code': 0, 'message': 'Task submitted', 'id': 'one'}
        assert submit(client, video, 'busy').json() == {'code': 2, 'message': 'A task is running.'}
        job = wait(client, 'one')
        assert job['status'] == 'completed', job
        assert job['model'] == 'vsr' and job['progress'] == 100
        assert job['size'] == '32x32' and job['seconds'] == '1.0'
        assert client.get('/v1/videos/one/content').content == video.read_bytes()
        record = service.db.external_task('vsr', 'one', None, service.config_scope)
        artifact = record['outputs']['artifacts'][0]['file']
        assert job['file_path'] == artifact['local_path']
        assert Path(job['file_path']).read_bytes() == video.read_bytes()
        assert job['inference_time_s'] > 0
        assert submit(client, video).json() == {'code': 1, 'message': 'taskId already exists', 'id': 'one'}
        assert client.delete('/v1/videos/one').json()['status'] == 'completed'
        assert client.get('/v1/videos?limit=1').json()['data'][0]['id'] == 'one'
        assert client.get('/v1/videos?after=missing').json() == {'data': [], 'object': 'list'}
        assert client.get('/v1/videos?limit=101').status_code == 422
        assert client.get('/v1/videos/missing').json() == {'detail': 'Video not found'}
        assert client.get('/v1/videos/one/progress').json()['callback_attempts'] is None


def test_validation_and_recovery(tmp_path, video):
    with TestClient(server(tmp_path).app) as client:
        for changes in ({'prompt': 'no'}, {'read_queue': 17}, {'tile_t': 6},
                        {'spatial_overlap': 640}, {'callbackUrl': 'ftp://no'},
                        {'videoUrl': 'http://example.com'}, {'taskId': '../bad'},
                        {'target_resolution': '31x32'}, {'outputObjectKey': 'x'}):
            response = submit(client, video, **changes)
            assert response.status_code == 200 and response.json()['code'] == 1, response.text
        assert submit(client, '/missing').json() == {'code': 1, 'message': 'Input video does not exist'}
        assert client.post('/v1/videos/restorations', content='{').json()['code'] == 1
        assert submit(client, video).json()['code'] == 0
        assert wait(client, 'one')['status'] == 'completed'


@pytest.mark.parametrize('operation', ['cancel', 'timeout'])
@pytest.mark.parametrize('isolation', ['thread', 'process'])
def test_cancel_timeout_and_resident_recovery(tmp_path, video, operation, isolation):
    slow = video.with_name('slow.mp4')
    shutil.copyfile(video, slow)
    service = server(tmp_path, isolation)
    with TestClient(service.app) as client:
        assert submit(client, slow, timeout=1 if operation == 'timeout' else -1).json()['code'] == 0
        assert client.get('/v1/videos/one/content').json() == {'detail': 'Generation is still in-progress'}
        if operation == 'cancel':
            for _ in range(200):
                if client.get('/v1/videos/one/progress').json()['progress'] == 1:
                    break
                time.sleep(.01)
            cancelled = client.delete('/v1/videos/one').json()
            assert cancelled['status'] in ('cancelling', 'cancelled')
        job = wait(client, 'one')
        assert job['status'] == ('cancelled' if operation == 'cancel' else 'expired'), job
        assert job['error']['code'] == ('task_cancelled' if operation == 'cancel' else 'execution_timeout')
        assert job['reason'] == job['error']['message']
        assert job['progress'] == 1 and job['completed_at'] is not None
        assert submit(client, video, 'recovered').json()['code'] == 0
        assert wait(client, 'recovered')['status'] == 'completed'
        assert (tmp_path / 'model_loads').read_text() == 'loaded\n'
        internal = service.db.external_task('vsr', 'one', None, service.config_scope)
        assert internal['state'] == ('cancelled' if operation == 'cancel' else 'expired')


@contextmanager
def remote(video):
    received = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(video.read_bytes())
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(500 if len(received) == 1 else 200); self.end_headers()
        def log_message(self, *args):
            pass
    http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{http.server_port}', received
    finally:
        http.shutdown(); http.server_close(); thread.join()


def test_url_and_callback_retry(tmp_path, video, monkeypatch):
    monkeypatch.setenv('NO_PROXY', '127.0.0.1,localhost')
    with remote(video) as (url, received), TestClient(server(tmp_path).app) as client:
        response = client.post('/v1/videos/restorations', json={
            'taskId': 'url', 'videoUrl': url + '/input.mp4', 'callbackUrl': url + '/callback',
            'target_resolution': '32x32'})
        assert response.json()['code'] == 0, response.text
        assert wait(client, 'url')['status'] == 'completed'
        for _ in range(400):
            progress = client.get('/v1/videos/url/progress').json()
            if progress['callback_status'] == 'succeeded':
                break
            time.sleep(.01)
        assert progress['callback_status'] == 'succeeded' and progress['callback_attempts'] == 2
        assert received[0] == received[1]
        assert received[-1]['status'] == 'completed' and received[-1]['model'] == 'vsr'


def test_upload_response_and_failure(tmp_path, video, monkeypatch):
    from nndeploy.e_dit.algorithms.vsr.server import handler as module
    class Storage:
        fail = False
        def __init__(self, config):
            assert config.bucket_name == 'bucket'
            self.bucket_name, self.endpoint_url = 'bucket', 'https://storage'
            self.client = SimpleNamespace(close=lambda: None)
        async def upload_file(self, path, key, bucket_name=None):
            assert key == 'custom.mp4' and bucket_name == 'other'
            if self.fail:
                raise RuntimeError('upload failed')
            return 'https://storage/other/custom.mp4'
    monkeypatch.setattr(module, 'RequestCloudStorage', Storage)
    options = dict(minioConfig={'endpoint': 'storage', 'bucketName': 'bucket',
        'accessKey': 'key', 'secretKey': 'secret'}, outputObjectKey='custom.mp4', output_bucket='other')
    instance = server(tmp_path)
    with TestClient(instance.app) as client:
        assert submit(client, video, **options).json()['code'] == 0
        job = wait(client, 'one')
        assert job['status'] == 'completed' and job['file_path'] is None
        response = client.get('/v1/videos/one/content')
        assert response.status_code == 400 and job['url'] in response.json()['detail']
        Storage.fail = True
        assert submit(client, video, 'fail', **options).json()['code'] == 0
        assert wait(client, 'fail')['reason'] == 'upload failed'
        record = instance.db.external_task('vsr', 'fail', None, instance.config_scope)
        assert record['state'] == 'failed' and record['outputs'] is None
        assert record['inputs']['storage_secret'] and 'secretKey' not in json.dumps(record['inputs']['request'])
        assert 'minio_config' not in record['inputs']['request']


def test_history_artifacts_and_pending_callback_survive_restart(tmp_path, video, monkeypatch):
    from nndeploy.server import callbacks
    from nndeploy.e_dit.algorithms.vsr.server.callbacks import task_callback
    delivered = []
    monkeypatch.setattr(callbacks, 'send', lambda record: 503)
    first = server(tmp_path)
    with TestClient(first.app) as client:
        assert submit(client, video, callbackUrl='http://127.0.0.1:9000/callback').json()['code'] == 0
        assert wait(client, 'one')['status'] == 'completed'
        record = first.db.external_task('vsr', 'one', None, first.config_scope)
        identity = record['task_id']
        assert record['callback']['state'] == 'pending'
        assert first.db.latest_progress(identity)['stage'] == 'restoring'
    def send(record):
        delivered.append(task_callback(record))
        return 200
    monkeypatch.setattr(callbacks, 'send', send)
    second = server(tmp_path)
    with TestClient(second.app) as client:
        assert client.get('/v1/videos/one').json()['status'] == 'completed'
        assert client.get('/v1/videos/one/content').content == video.read_bytes()
        assert submit(client, video).json()['message'] == 'taskId already exists'
        for _ in range(500):
            status = client.get('/v1/videos/one/progress').json()
            if status['callback_status'] == 'succeeded':
                break
            time.sleep(.01)
        assert status['callback_status'] == 'succeeded'
        assert delivered and delivered[-1]['id'] == 'one' and delivered[-1]['status'] == 'completed'
        assert second.db.external_task('vsr', 'one', None, second.config_scope)['task_id'] == identity


@pytest.mark.parametrize('reverse', [False, True])
def test_videoedit_and_vsr_share_routes_and_resolve_duplicate_ids(tmp_path, video, reverse, monkeypatch):
    from nndeploy.server import Server
    from nndeploy.e_dit.algorithms.vsr.server import add_vsr
    from nndeploy.e_dit.algorithms.wan.server import add_videoedit
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'videoedit'))
    from test_videoedit_api import graph_factory, make_inputs
    instance = Server(resources=tmp_path/'combined', data_dir=tmp_path/'combined')
    def add_restore():
        add_vsr(instance, DiTParams(model='wan_vsr', wan_root=str(tmp_path)),
                isolation='thread', handler_factory=FakeHandler)
    def add_edit():
        add_videoedit(instance, {'normal': {'model_path': 'unused'}}, graph_factory=graph_factory,
                      execution={'isolation': 'thread'})
    for add in ([add_restore, add_edit] if reverse else [add_edit, add_restore]):
        add()
    with TestClient(instance.app) as client:
        paths = instance.app.openapi()['paths']
        assert '/v1/videos/restorations' in paths and '/v1/videos/repairs' in paths
        assert set(client.get('/health').json()['services']) == {'videoedit', 'vsr'}
        assert submit(client, video).json()['code'] == 0
        assert wait(client, 'one')['status'] == 'completed'
        assert client.post('/v1/videos/repairs', json=make_inputs(tmp_path/'edit-inputs')).json()['code'] == 0
        for _ in range(300):
            edited = client.get('/v1/videos/one?entry=videoedit').json()
            if edited['status'] == 'completed':
                break
            time.sleep(.01)
        assert edited['status'] == 'completed', edited
        assert client.get('/v1/videos/one').status_code == 409
        assert client.get('/v1/videos/one/content').status_code == 409
        assert client.delete('/v1/videos/one').status_code == 409
        values = client.get('/v1/videos?order=asc').json()['data']
        assert [item['entry'] for item in values] == ['vsr', 'videoedit']
        assert len({item['task_id'] for item in values}) == 2
        assert client.get('/v1/videos?after=one').status_code == 409
        page = client.get('/v1/videos', params={'after': values[0]['task_id'], 'order': 'asc', 'limit': 1}).json()
        assert page['data'] == [values[1]]
        assert client.get('/v1/videos?entry=vsr').json()['data'] == [values[0]]
        assert client.get('/v1/videos/one/content?entry=vsr').content == video.read_bytes()
        assert client.get('/v1/videos/' + values[0]['task_id'] + '/content').content == video.read_bytes()
        assert client.get('/v1/videos/one?entry=missing').status_code == 404


def test_video_ownership_applies_to_every_resource_route(tmp_path, video, monkeypatch):
    from nndeploy.server import Server
    from nndeploy.e_dit.algorithms.vsr.server import add_vsr
    tokens = {name: name * 32 for name in ('alice', 'bob', 'reader')}
    for name, token in tokens.items():
        monkeypatch.setenv('VIDEO_KEY_' + name, token)
    auth = {'keys': [dict(subject=name, token_env='VIDEO_KEY_' + name, entries=['vsr'],
        permissions=['tasks:read', 'files:read'] if name == 'reader' else
        ['deploy', 'tasks:read', 'tasks:cancel', 'files:read']) for name in tokens]}
    instance = Server(resources=tmp_path/'owned', data_dir=tmp_path/'owned', auth_config=auth, admin_token='admin-token')
    add_vsr(instance, DiTParams(model='wan_vsr', wan_root=str(tmp_path)), isolation='thread', handler_factory=FakeHandler)
    with TestClient(instance.app) as client:
        assert submit(client, video).status_code == 401
        ids = {}
        for name in ('alice', 'bob'):
            client.headers['Authorization'] = 'Bearer ' + tokens[name]
            assert submit(client, video).json()['code'] == 0
            assert wait(client, 'one')['status'] == 'completed'
            record = instance.db.external_task('vsr', 'one', name, instance.config_scope)
            ids[name] = record['task_id']
            assert record['owner'] == name
            assert instance.db.get_file(record['outputs']['artifacts'][0]['file']['file_id'])['owner'] == name
        assert ids['alice'] != ids['bob']
        assert len(client.get('/v1/videos').json()['data']) == 1
        for suffix in ('', '/progress', '/content'):
            assert client.get('/v1/videos/' + ids['alice'] + suffix).status_code == 404
        assert client.delete('/v1/videos/' + ids['alice']).status_code == 404
        alice = instance.db.get_task(ids['alice'])
        file_id = alice['outputs']['artifacts'][0]['file']['file_id']
        assert client.get('/v1/files/' + file_id + '/content').status_code == 404
        client.headers['Authorization'] = 'Bearer ' + tokens['reader']
        assert submit(client, video).status_code == 403
        assert client.delete('/v1/videos/one?entry=vsr').status_code == 403
        assert client.get('/v1/videos').json()['data'] == []
        client.headers['Authorization'] = 'Bearer admin-token'
        values = client.get('/v1/videos').json()['data']
        assert len(values) == 2
        assert {value['task_id'] for value in values} == set(ids.values())
        assert client.get('/v1/videos/' + ids['alice']).status_code == 200


def test_video_memory_estimator_is_persisted_and_bounded(tmp_path, video):
    calls = []
    def estimate(request):
        calls.append(dict(request))
        amount = 101 if request['task_id'] == 'too-large' else 30
        request['target_resolution'] = 'invalid-mutation'
        return {'cpu': amount}
    service = Server(resources=tmp_path / 'server', data_dir=tmp_path / 'server', device_limits={
        'devices': {'cpu': {'max_workers': 1, 'memory_bytes': 200}}})
    add_vsr(service, DiTParams(model='wan_vsr', wan_root=str(tmp_path)), isolation='thread',
            handler_factory=FakeHandler, memory_estimator=estimate,
            execution={'devices': ['cpu'], 'memory_bytes': {'cpu': 100}, 'temporary_memory_bytes': {'cpu': 100}})
    with TestClient(service.app) as client:
        assert submit(client, video).json()['code'] == 0
        assert wait(client, 'one')['status'] == 'completed'
        record = service.db.external_task('vsr', 'one', None, service.config_scope)
        assert record['service']['temporary_memory_bytes'] == {'cpu': 30}
        assert record['inputs']['request']['target_resolution'] == '32x32'
        assert 'minio_config' not in calls[0]
        assert submit(client, video, 'too-large').json()['code'] == 1
        assert service.db.external_task('vsr', 'too-large', None, service.config_scope) is None
