"""Editor Run and resident Deploy share configured device admission."""
import json
import uuid
from types import SimpleNamespace
from contextlib import ExitStack

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from nndeploy.server import Server
from nndeploy.server.app import _EditorRuntime
from nndeploy.server.task_queue import ExecutionStatus
from test_editor_history import location
from test_editor_queue_recovery import wait, live
from test_device_execution import limits, activity
from test_server_api import text_graph, image_graph
from test_shared_server import entry_config, shared_deploy


def saved(client, server, graph, **execution):
    response = client.post('/api/workflow/save', json={'businessContent': graph})
    assert response.status_code == 200, response.text
    identity = response.json()['result']['id']
    path = server.db.get_workflow_path(identity).with_suffix('.server.json')
    path.write_text(json.dumps({'execution': dict(devices=['cpu'], **execution)}))
    return identity, path


def test_editor_budget_admission_rejects_missing_or_invalid_settings(tmp_path, location):
    server = Server(SimpleNamespace(resources=str(tmp_path / 'editor'), debug=True), database_url=location,
                    device_limits=limits())
    with TestClient(server.app) as client:
        graph = text_graph()
        response = client.post('/api/queue', json=graph)
        assert response.status_code == 422 and response.json()['error']['accepted'] is False
        identity, path = saved(client, server, graph)
        for content in ('{', json.dumps({'execution': {'devices': ['unknown']}}),
                        json.dumps({'execution': {'devices': ['cpu'], 'workers': 2}})):
            path.write_text(content)
            response = client.post('/api/queue', params={'workflow_id': identity}, json=graph)
            assert response.status_code == 422, response.text
        assert server.queue.get_current_queue()['PENDING'] == []


def test_budget_snapshot_is_retained_across_retries_and_terminal_history(tmp_path, location):
    server = Server(SimpleNamespace(resources=str(tmp_path / 'editor'), debug=True), database_url=location,
                    device_limits=limits())
    with TestClient(server.app) as client:
        graph = text_graph()
        identity, path = saved(client, server, graph, priority=3)
        key = str(uuid.uuid4())
        options = dict(params={'workflow_id': identity}, json=graph, headers={'Idempotency-Key': key})
        assert client.post('/api/queue', **options).status_code == 202
        path.unlink()
        assert client.post('/api/queue', **options).status_code == 202
        idx, payload = server.queue.get(timeout=1)
        assert payload['_editor_execution']['execution']['priority'] == 3
        assert payload['_editor_execution']['name'] == 'workflow-' + identity
        server.queue.mark_dispatched(idx)
        server.queue.task_done(idx, ExecutionStatus(True, 'Completed'), {}, {})
        assert client.post('/api/queue', **options).status_code == 202
        changed = dict(options, params={'workflow_id': str(uuid.uuid4())})
        assert client.post('/api/queue', **changed).status_code == 409


def test_missing_exit_evidence_does_not_admit_or_reserve_budget(tmp_path, location, monkeypatch):
    from nndeploy.server import control_guard
    from nndeploy.server.devices import EditorDeviceRun
    from nndeploy.server.errors import ServerError
    server = Server(SimpleNamespace(resources=str(tmp_path / 'editor'), debug=True), database_url=location,
                    device_limits=limits())
    with TestClient(server.app) as client:
        graph = text_graph()
        identity, _ = saved(client, server, graph)
        monkeypatch.setattr(control_guard, '_boot_id', lambda: None)
        response = client.post('/api/queue', params={'workflow_id': identity}, json=graph)
        assert response.status_code == 422 and response.json()['error']['accepted'] is False
        with pytest.raises(ServerError, match='process-exit evidence'):
            EditorDeviceRun({}, None)
        assert server.queue.get_current_queue()['PENDING'] == []
        assert server.devices.snapshot()['cpu']['used']['workers'] == 0


def test_pending_budget_snapshot_survives_server_restart_and_sidecar_removal(tmp_path, location):
    args = SimpleNamespace(resources=str(tmp_path / 'editor'), debug=True)
    server = Server(args, database_url=location, device_limits=limits())
    graph, key = text_graph(), str(uuid.uuid4())
    with TestClient(server.app) as client:
        identity, path = saved(client, server, graph, priority=-2)
        options = dict(params={'workflow_id': identity}, json=graph, headers={'Idempotency-Key': key})
        assert client.post('/api/queue', **options).status_code == 202
        path.unlink()
    restored = Server(args, database_url=location, device_limits=limits())
    with TestClient(restored.app) as client:
        assert client.post('/api/queue', **options).status_code == 202
        idx, payload = restored.queue.get(timeout=1)
        assert payload['id'] == key and payload['_editor_execution']['execution']['priority'] == -2
        assert restored.queue.get(timeout=.01) is None


def lost_receipt_worker(*args, **kwargs):
    from nndeploy.server.worker import run
    from nndeploy.server.devices import Devices
    reserve, begin = Devices._reserve, Devices._begin
    missed = set()
    def lost(operation, key, *values, **options):
        result = operation(*values, **options)
        if key not in missed:
            missed.add(key)
            raise OSError('Controlled lost device commit receipt')
        return result
    Devices._reserve = lambda *values, **options: lost(reserve, 'reserve', *values, **options)
    Devices._begin = lambda *values, **options: lost(begin, 'begin', *values, **options)
    run(*args, **kwargs)


@pytest.mark.parametrize('outcome', ['complete', 'cancel', 'timeout', 'kill', 'running_cancel', 'lost_receipts'])
@pytest.mark.parametrize('separate_server', [False, True])
def test_real_editor_run_shares_resident_slot_until_physical_exit(tmp_path, location, monkeypatch, outcome, separate_server):
    if outcome == 'lost_receipts':
        from nndeploy.server import app
        monkeypatch.setattr(app, 'worker_run', lost_receipt_worker)
    resources = tmp_path / 'resources'
    plugins = resources / 'plugin'
    plugins.mkdir(parents=True)
    (plugins / 'budget_image.py').write_text('''
import json
import time
from pathlib import Path
import nndeploy.dag
from nndeploy.codec.pil import PILImageEncodec
class BudgetImage(PILImageEncodec):
    def deserialize(self, value):
        data = json.loads(value)
        self.marker, self.release = Path(data['marker_']), Path(data['release_'])
        return super().deserialize(value)
    def init(self):
        status = super().init()
        self.marker.touch()
        deadline = time.monotonic() + 20
        while not self.release.exists():
            if time.monotonic() >= deadline:
                raise RuntimeError('Editor test did not release initialization')
            time.sleep(.02)
        return status
class Creator(nndeploy.dag.NodeCreator):
    def create_node(self, name, inputs, outputs):
        self.node = BudgetImage(name, inputs, outputs)
        self.node.set_key('test.editor.BudgetImage')
        return self.node
creator = Creator()
nndeploy.dag.register_node('test.editor.BudgetImage', creator)
''')
    source = resources / 'input.png'
    Image.new('RGB', (4, 5), (10, 20, 30)).save(source)
    marker, release, editor_marker, editor_release, next_marker = (tmp_path / name for name in
        ('resident-start', 'resident-release', 'editor-start', 'editor-release', 'next-start'))
    graph = image_graph()
    graph['node_repository_'][0]['path_'] = str(source)
    graph['node_repository_'][1].update(key_='test.editor.BudgetImage', path_=str(resources / 'result.png'),
                                       marker_=str(editor_marker), release_=str(editor_release),
                                       node_type_='Output', io_type_='Image', io_params_=['path_'])
    args = SimpleNamespace(resources=str(resources), log=str(tmp_path / 'editor.log'), debug=True)
    runtime = _EditorRuntime(args)
    settings = limits(host=uuid.uuid4().hex if separate_server else None)
    server = runtime.start(lambda jobs, plugins, cancellations: Server(args, jobs, plugins, cancellations,
                            database_url=location, device_limits=settings))
    peer = Server(data_dir=tmp_path / 'peer', database_url=server.db.url.render_as_string(hide_password=False),
                  device_limits=settings) if separate_server else server
    config = entry_config()
    config['execution'].update(isolation='thread', devices=['cpu'])
    peer.add(name='resident', deploy=shared_deploy, config=config)
    with TestClient(server.app) as client, ExitStack() as peers:
        deploy_client = peers.enter_context(TestClient(peer.app)) if separate_server else client
        try:
            identity, _ = saved(client, server, graph, queue_timeout_seconds=.3 if outcome == 'timeout' else 30)
            first = deploy_client.post('/v1/server/resident/deploy', json={'mode': 'async',
                'inputs': {'marker': str(marker), 'release': str(release)}})
            assert first.status_code == 202, first.text
            wait(marker.exists)
            response = client.post('/api/queue', params={'workflow_id': identity}, json=graph)
            assert response.status_code == 202, response.text
            task_id = response.json()['result']['task_id']
            def task():
                return client.get('/api/queue/task/' + task_id).json()['result']
            if outcome == 'timeout':
                wait(lambda: task()['state'] == 'FAILED')
                assert not editor_marker.exists()
            else:
                wait(lambda: activity(server) == {'running': 1, 'waiting': 1})
                assert not editor_marker.exists()
                if outcome == 'cancel':
                    assert client.post('/api/queue/cancel/' + task_id).status_code == 200
                    wait(lambda: task()['state'] == 'CANCELLED')
                    assert not editor_marker.exists()
                else:
                    release.touch()
                    wait(editor_marker.exists)
                    worker = runtime.worker
                    worker_pid = worker.pid
                    next_task = deploy_client.post('/v1/server/resident/deploy', json={'mode': 'async',
                        'inputs': {'marker': str(next_marker)}})
                    assert next_task.status_code == 202, next_task.text
                    wait(lambda: activity(server) == {'running': 1, 'waiting': 1})
                    server.devices.recover()
                    assert not next_marker.exists() and worker.is_alive()
                    if outcome == 'running_cancel':
                        assert client.post('/api/queue/cancel/' + task_id).status_code == 200
                        assert task()['state'] not in ('SUCCEEDED', 'FAILED', 'CANCELLED')
                        server.devices.recover()
                        assert not next_marker.exists() and worker.is_alive()
                    if outcome == 'kill':
                        worker.kill()
                    else:
                        editor_release.touch()
                    expected = 'FAILED' if outcome == 'kill' else 'CANCELLED' if outcome == 'running_cancel' else 'SUCCEEDED'
                    wait(lambda: task()['state'] == expected)
                    wait(next_marker.exists)
                    assert not live(worker_pid)
                    if outcome in ('complete', 'lost_receipts'):
                        assert len(task()['output_files']) == 1
            def reclaimed():
                server.devices.recover()
                return server.devices.snapshot()['cpu']['used']['workers'] == 1
            wait(reclaimed)
        finally:
            release.touch()
            editor_release.touch()
            runtime.close()
