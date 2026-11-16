"""Execution deadlines are fenced outside a potentially GIL-blocked Worker."""
import io
import copy
import json
import os
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import select

from nndeploy.server import Server, app
from nndeploy.server.db import files
from nndeploy.server.task_queue import ExecutionStatus
from test_editor_history import location
from test_editor_queue_recovery import wait, live
from test_editor_devices import saved
from test_device_execution import limits, activity
from test_shared_server import entry_config, shared_deploy
from test_server_api import image_graph, text_graph


@pytest.mark.parametrize('reason', ['execution', 'queue', 'cancel'])
def test_manager_refuses_expired_results_until_owner_exit(monkeypatch, reason):
    now = [100.0]
    monkeypatch.setattr(app, 'time', SimpleNamespace(time=lambda: now[0], monotonic=lambda: now[0]))
    jobs = app._EditorJobQueue()
    reader = jobs.new_reader()
    payload = {'_editor_execution': {'execution': {'queue_timeout_seconds': 5, 'execution_timeout_seconds': 2}},
               '_editor_submitted': 100.0}
    jobs.put((1, payload))
    jobs.get(reader=reader)
    assert jobs.claim(1, reader)
    assert not jobs.begin_execution(1, 'other')
    if reason == 'execution':
        assert jobs.begin_execution(1, reader)
        now[0] += 1
        assert jobs.begin_execution(1, reader)  # Lost reply cannot extend the deadline.
        now[0] += 2
    elif reason == 'queue':
        now[0] += 6
        assert not jobs.begin_execution(1, reader)
    else:
        jobs.cancel(1)
        assert not jobs.begin_execution(1, reader)
    # The late result is rejected even before the parent has polled expiry.
    assert not jobs.finish(1, reader, ExecutionStatus(True, 'late'), {'bad': 'result'}, {})
    assert jobs.outcomes() == []
    assert jobs.execution_expired(reader) and not jobs.execution_expired('other')
    with pytest.raises(RuntimeError, match='exit'):
        jobs.new_reader()
    assert jobs.retire_reader(reader) is True
    assert jobs.retire_reader(reader) is True
    _, status, outputs, _ = jobs.outcomes()[0]
    assert status.str == ('cancelled' if reason == 'cancel' else 'failed')
    assert not outputs and (reason == 'cancel' or 'timed out' in status.messages[0])
    assert not jobs.finish(1, reader, ExecutionStatus(True), {}, {})


def capture_stall_worker(*args, **kwargs):
    import ctypes
    from pathlib import Path
    from nndeploy.server import editor_outputs
    from nndeploy.server.worker import run
    capture = editor_outputs.capture_worker_files
    def stalled(payload, options):
        capture(payload, options)
        graph = payload['graph_json']
        if graph.get('capture_stall_'):
            Path(graph['capture_stall_']).touch()
            ctypes.PyDLL(None).sleep(60)  # Hold the GIL: a child Python timer cannot enforce this.
    editor_outputs.capture_worker_files = stalled
    run(*args, **kwargs)


def test_accepted_result_is_preserved_while_exit_deadline_is_enforced(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(app, 'time', SimpleNamespace(time=lambda: now[0], monotonic=lambda: now[0]))
    jobs = app._EditorJobQueue()
    reader = jobs.new_reader()
    jobs.put((1, {'_editor_execution': {'execution': {'execution_timeout_seconds': 2}}}))
    jobs.get(reader=reader)
    assert jobs.claim(1, reader) and jobs.begin_execution(1, reader)
    status = ExecutionStatus(True, 'accepted')
    assert jobs.finish(1, reader, status, {'text': 'original'}, {})
    now[0] += 3
    assert jobs.execution_expired(reader)
    jobs.retire_reader(reader)
    assert jobs.outcomes() == [(1, status, {'text': 'original'}, {})]
    assert not jobs.execution_expired(reader)
    current = jobs.new_reader()
    assert not jobs.execution_expired(current)


def exit_stall_worker(*args, **kwargs):
    import ctypes
    from pathlib import Path
    from nndeploy.server.worker import run
    result_queue = args[1]
    resources = Path(args[6])
    class Results:
        def put(self, value):
            result_queue.put(value)
            if (resources / 'stall-exit').exists():
                (resources / 'exit-started').touch()
                ctypes.PyDLL(None).sleep(60)
    run(args[0], Results(), *args[2:], **kwargs)


def test_waiting_for_device_does_not_consume_execution_deadline(tmp_path, location):
    args = SimpleNamespace(resources=str(tmp_path / 'resources'), log=str(tmp_path / 'editor.log'), debug=True)
    runtime = app._EditorRuntime(args)
    server = runtime.start(lambda jobs, plugins, cancellations: Server(args, jobs, plugins, cancellations,
                            database_url=location, device_limits=limits()))
    config = entry_config()
    config['execution'].update(isolation='thread', devices=['cpu'])
    server.add(name='resident', deploy=shared_deploy, config=config)
    marker, release = tmp_path / 'resident-start', tmp_path / 'release'
    with TestClient(server.app) as client:
        try:
            graph = text_graph()
            identity, _ = saved(client, server, graph, isolation='process', execution_timeout_seconds=1,
                                queue_timeout_seconds=15)
            result = client.post('/v1/server/resident/deploy', json={'mode': 'async',
                'inputs': {'marker': str(marker), 'release': str(release)}})
            assert result.status_code == 202
            wait(marker.exists)
            response = client.post('/api/queue', params={'workflow_id': identity}, json=graph)
            assert response.status_code == 202, response.text
            task_id = response.json()['result']['task_id']
            def task():
                return client.get('/api/queue/task/' + task_id).json()['result']
            wait(lambda: activity(server)['waiting'] == 1)
            time.sleep(1.25)  # Deliberately exceed the declared execution time while waiting.
            assert task()['state'] not in ('FAILED', 'CANCELLED', 'SUCCEEDED')
            release.touch()
            wait(lambda: task()['state'] in ('FAILED', 'SUCCEEDED'))
            assert task()['state'] == 'SUCCEEDED', task()
        finally:
            release.touch()
            runtime.close()


@pytest.mark.skipif(os.name != 'posix', reason='Native GIL-blocking sleep uses POSIX libc')
@pytest.mark.parametrize('stage', ['init', 'run', 'deinit', 'capture', 'exit'])
@pytest.mark.parametrize('budgeted', [False, True])
def test_deadline_kills_native_work_and_preserves_other_entry_and_next_run(tmp_path, location, monkeypatch, stage, budgeted):
    resources = tmp_path / 'resources'
    plugins = resources / 'plugin'
    plugins.mkdir(parents=True)
    (plugins / 'deadline_image.py').write_text('''
import ctypes
import json
from pathlib import Path
import nndeploy.dag
from nndeploy.codec.pil import PILImageEncodec
class DeadlineImage(PILImageEncodec):
    def deserialize(self, value):
        data = json.loads(value)
        self.stage, self.marker = data['stage_'], data['marker_']
        return super().deserialize(value)
    def stall(self, stage):
        if self.stage == stage:
            Path(self.marker).touch()
            ctypes.PyDLL(None).sleep(60)
    def init(self):
        self.stall('init')
        return super().init()
    def run(self):
        self.stall('run')
        return super().run()
    def deinit(self):
        self.stall('deinit')
        return super().deinit()
class Creator(nndeploy.dag.NodeCreator):
    def create_node(self, name, inputs, outputs):
        self.node = DeadlineImage(name, inputs, outputs)
        self.node.set_key('test.editor.DeadlineImage')
        return self.node
creator = Creator()
nndeploy.dag.register_node('test.editor.DeadlineImage', creator)
''')
    source = resources / 'input.png'
    Image.new('RGB', (4, 5), (10, 20, 30)).save(source)
    marker, resident_marker = tmp_path / 'native-start', tmp_path / 'resident-start'
    if stage == 'exit':
        (resources / 'stall-exit').touch()
        marker = resources / 'exit-started'
        monkeypatch.setattr(app, 'worker_run', exit_stall_worker)
    graph = image_graph()
    graph['node_repository_'][0]['path_'] = str(source)
    graph['node_repository_'][1].update(key_='test.editor.DeadlineImage', path_=str(resources / 'result.png'),
        stage_=stage, marker_=str(marker), node_type_='Output', io_type_='Image', io_params_=['path_'])
    if stage == 'capture':
        graph['capture_stall_'] = str(marker)
        monkeypatch.setattr(app, 'worker_run', capture_stall_worker)
    args = SimpleNamespace(resources=str(resources), log=str(tmp_path / 'editor.log'), debug=True)
    runtime = app._EditorRuntime(args)
    server = runtime.start(lambda jobs, plugins, cancellations: Server(args, jobs, plugins, cancellations,
                database_url=location, device_limits=limits() if budgeted else None))
    config = entry_config()
    config['execution'].update(isolation='thread', devices=['cpu'] if budgeted else [])
    server.add(name='resident', deploy=shared_deploy, config=config)
    with TestClient(server.app) as client:
        try:
            identity, path = saved(client, server, graph)
            path.write_text(json.dumps({'execution': {'devices': ['cpu'] if budgeted else [],
                'isolation': 'process', 'execution_timeout_seconds': 2, 'queue_timeout_seconds': 30}}))
            worker_pid = runtime.worker.pid
            response = client.post('/api/queue', params={'workflow_id': identity}, json=graph)
            assert response.status_code == 202, response.text
            task_id = response.json()['result']['task_id']
            def task():
                return client.get('/api/queue/task/' + task_id).json()['result']
            wait(marker.exists)
            if stage != 'exit':
                assert task()['state'] not in ('SUCCEEDED', 'FAILED', 'CANCELLED')
            expired_id = None
            if not budgeted and stage == 'init':
                expired_graph = copy.deepcopy(graph)
                expired_marker = tmp_path / 'expired-must-not-initialize'
                expired_graph['node_repository_'][1]['marker_'] = str(expired_marker)
                original = path.read_text()
                settings = json.loads(original)
                settings['execution']['queue_timeout_seconds'] = .1
                path.write_text(json.dumps(settings))
                expired = client.post('/api/queue', params={'workflow_id': identity}, json=expired_graph)
                assert expired.status_code == 202
                expired_id = expired.json()['result']['task_id']
                path.write_text(original)
            other = client.post('/v1/server/resident/deploy', json={'mode': 'async', 'inputs': {'marker': str(resident_marker)}})
            assert other.status_code == 202
            if budgeted:
                wait(lambda: activity(server)['waiting'] == 1)
                server.devices.recover()
                assert not resident_marker.exists() and live(worker_pid)
            if stage == 'exit':
                wait(lambda: task()['state'] == 'SUCCEEDED')
                wait(lambda: not live(worker_pid))
                assert task()['state'] == 'SUCCEEDED' and len(task()['output_files']) == 1
                (resources / 'stall-exit').unlink()
            else:
                wait(lambda: task()['state'] == 'FAILED')
                assert 'Execution timed out' in str(task()['status']) and task()['output_files'] == []
            assert not live(worker_pid)
            wait(resident_marker.exists)
            if expired_id:
                def expired_task():
                    return client.get('/api/queue/task/' + expired_id).json()['result']
                wait(lambda: expired_task()['state'] == 'FAILED')
                assert 'Queue timed out' in str(expired_task()['status']) and not expired_marker.exists()
            server.storage.cleanup()
            with server.db.engine.connect() as connection:
                owned = list(connection.execute(select(files.c.file_id).where(files.c.owner == 'editor:' + server.config_scope)))
                assert len(owned) == (1 if stage == 'exit' else 0)
            # The replacement Worker handles an ordinary request using the same saved settings.
            graph['node_repository_'][1]['stage_'] = 'none'
            graph.pop('capture_stall_', None)
            result = client.post('/api/queue', params={'workflow_id': identity}, json=graph)
            assert result.status_code == 202, result.text
            next_id = result.json()['result']['task_id']
            def next_result():
                return client.get('/api/queue/task/' + next_id).json()['result']
            wait(lambda: next_result()['state'] in ('SUCCEEDED', 'FAILED'))
            assert next_result()['state'] == 'SUCCEEDED', next_result()
            image = client.get(next_result()['output_files'][0]['url'])
            assert Image.open(io.BytesIO(image.content)).getpixel((0, 0)) == (10, 20, 30)
        finally:
            runtime.close()
