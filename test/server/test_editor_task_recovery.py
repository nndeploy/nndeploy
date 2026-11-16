"""Persist editor admission and recover only after proving execution has exited."""
import json
import multiprocessing
import queue
import threading
import uuid
import os
import signal
import subprocess
import sys

import pytest
import requests
from fastapi.testclient import TestClient

from nndeploy.server.control_guard import ControlGuard
from nndeploy.server.task_queue import ExecutionStatus
from test_editor_history import location, editor

from test_cli import unused_port
from test_editor_queue_recovery import LAUNCH, live, wait
from test_server_api import text_graph


PENDING_LAUNCH = LAUNCH.replace('app.main()', '''
original_scheduler = app.start_scheduler
def scheduler(queue, jobs, stop):
    if not (root / 'dispatch').exists():
        thread = threading.Thread(target=lambda: stop.wait(), daemon=True)
        thread.start()
        return thread
    return original_scheduler(queue, jobs, stop)
app.start_scheduler = scheduler
app.main()
''')


@pytest.mark.skipif(os.name != 'posix', reason='Requires POSIX process signals')
def test_editor_recovers_accepted_unpublished_work_after_parent_kill(tmp_path):
    address = f'http://127.0.0.1:{unused_port()}'
    resources = tmp_path / 'resources'
    command = [sys.executable, '-c', PENDING_LAUNCH, '--host', '127.0.0.1',
               '--port', address.rsplit(':', 1)[1], '--debug', '--resources', str(resources),
               '--log', str(tmp_path / 'editor.log')]
    environment = dict(os.environ, NNDEPLOY_EDITOR_RECOVERY_TEST=str(tmp_path))
    children = set()
    console = tmp_path / 'console.log'
    with console.open('w') as log:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=log)
        def ready():
            assert process.poll() is None, console.read_text()
            try:
                return requests.get(address + '/health/ready', timeout=.5).status_code == 200
            except requests.RequestException:
                return False
        def remember():
            records = json.loads((tmp_path / 'pids.json').read_text())
            children.update(pid for group in records.values() for pid in group)
        try:
            wait(ready, 30)
            remember()
            graph = text_graph()
            graph['node_repository_'][0]['str_'] = 'restored request'
            output = tmp_path / 'output.txt'
            graph['node_repository_'][1]['frontend_params'].update(save_to_file=True, file_path=str(output))
            response = requests.post(address + '/api/queue', json=graph, timeout=3)
            assert response.status_code == 202, response.text
            identity = response.json()['result']['task_id']
            assert not output.exists()
            process.kill()
            process.wait(timeout=5)
            wait(lambda: not any(live(pid) for pid in children))
            (tmp_path / 'dispatch').touch()
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=log)
            wait(ready, 30)
            remember()
            def completed():
                rows = requests.get(address + '/api/queue/history', timeout=3).json()['result']['items']
                return next((record for record in rows if record['task_id'] == identity), None)
            record = wait(completed, 8)
            assert record['state'] == 'SUCCEEDED', record
            assert output.read_text() == 'restored request\n'
            process.terminate()
            process.wait(timeout=15)
            wait(lambda: not any(live(pid) for pid in children))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            if (tmp_path / 'pids.json').exists():
                remember()
            for pid in children:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def hold_guard(path, ready):
    import time
    guard = ControlGuard.join(path)
    ready.set()
    time.sleep(60)
    guard.close()


@pytest.mark.parametrize('state', ['pending', 'dispatched', 'running'])
@pytest.mark.parametrize('cancel', [False, True])
def test_editor_recovery_preserves_pending_and_never_replays_uncertain_tasks(tmp_path, location, state, cancel):
    path = tmp_path / 'editor'
    first = editor(path, location)
    with TestClient(first.app) as client:
        identity = client.post('/api/queue', json=text_graph()).json()['result']['task_id']
        old_guard = first.queue.history_store.guard.path
        if state != 'pending':
            idx, payload = first.queue.get(timeout=1)
            first.queue.dispatch(idx, payload, queue.Queue())
            if state == 'running':
                first.queue.mark_started(identity, worker_pid=123)
        if cancel:
            # The parent can die after persisting intent but before sending IPC.
            first.queue.history_store.cancel(identity)
    restored = editor(path, location)
    with TestClient(restored.app) as client:
        if state == 'pending' and not cancel:
            idx, payload = restored.queue.get(timeout=.1)
            assert payload['id'] == identity
            restored.queue.task_done(idx, ExecutionStatus(True), {}, {})
        else:
            assert restored.queue.get(timeout=.01) is None
            record = restored.queue.get_task_by_id(identity)
            assert record['state'] == ('CANCELLED' if cancel else 'FAILED')
        with pytest.raises(RuntimeError, match='revoked'):
            ControlGuard.join(old_guard)
        assert client.get('/api/queue/history').json()['result']['total'] == 1


def test_editor_recovery_waits_for_real_child_exit(tmp_path, location):
    path = tmp_path / 'editor'
    first = editor(path, location)
    context = multiprocessing.get_context('spawn')
    ready = context.Event()
    child = None
    try:
        with TestClient(first.app) as client:
            running = client.post('/api/queue', json=text_graph()).json()['result']['task_id']
            idx, payload = first.queue.get(timeout=1)
            first.queue.dispatch(idx, payload, queue.Queue())
            first.queue.mark_started(running, worker_pid=123)
            pending = client.post('/api/queue', json=text_graph()).json()['result']['task_id']
            child = context.Process(target=hold_guard, args=(str(first.queue.history_store.guard.path), ready))
            child.start()
            assert ready.wait(5)
        restored = editor(path, location)
        with TestClient(restored.app) as client:
            assert child.is_alive()
            assert restored.queue.get(timeout=.01) is None
            assert client.get('/api/queue/history').json()['result']['total'] == 0
            status = client.get('/api/queue/info').json()['result']
            assert any(row['task']['id'] == running for row in status['running'])
            assert any(row['task']['id'] == pending for row in status['pending'])
            response = client.post('/api/queue/cancel/' + running)
            assert response.json()['result']['state'] == 'RUNNING'
            assert response.json()['result']['cancel_requested']
            restored.queue.recover_editor()
            assert client.get('/api/queue/history').json()['result']['total'] == 0
            child.kill()
            child.join(timeout=5)
            assert not child.is_alive()
            restored.queue.recover_editor()
            assert restored.queue.get_task_by_id(running)['state'] == 'CANCELLED'
            idx, payload = restored.queue.get(timeout=.1)
            assert payload['id'] == pending
            restored.queue.task_done(idx, ExecutionStatus(True), {}, {})
    finally:
        if child is not None:
            if child.is_alive():
                child.kill()
            child.join(timeout=5)
            child.close()


def test_editor_dispatch_database_failure_never_publishes_unrecorded_work(tmp_path, location, monkeypatch):
    from sqlalchemy.exc import SQLAlchemyError
    from nndeploy.server.app import start_scheduler
    server = editor(tmp_path / 'editor', location)
    with TestClient(server.app) as client:
        identity = client.post('/api/queue', json=text_graph()).json()['result']['task_id']
        original = server.queue.history_store.dispatch
        entered, release, stop = threading.Event(), threading.Event(), threading.Event()
        def dispatch(task_id):
            if not release.is_set():
                entered.set()
                raise SQLAlchemyError('Controlled dispatch database outage')
            return original(task_id)
        monkeypatch.setattr(server.queue.history_store, 'dispatch', dispatch)
        jobs = queue.Queue()
        thread = start_scheduler(server.queue, jobs, stop)
        try:
            assert entered.wait(3)
            assert jobs.empty()
            assert thread.is_alive()
            release.set()
            idx, payload = jobs.get(timeout=3)
            assert payload['id'] == identity
            server.queue.task_done(idx, ExecutionStatus(True), {}, {})
            assert jobs.empty()
        finally:
            release.set()
            stop.set()
            thread.join(timeout=3)
            assert not thread.is_alive()


def test_editor_admission_lost_receipt_is_recovered_with_original_task_id(tmp_path, location, monkeypatch):
    from sqlalchemy.exc import SQLAlchemyError
    server = editor(tmp_path / 'editor', location)
    with TestClient(server.app) as client:
        original = server.queue.history_store.admit
        def admit(*args):
            original(*args)
            raise SQLAlchemyError('Controlled admission receipt loss')
        monkeypatch.setattr(server.queue.history_store, 'admit', admit)
        response = client.post('/api/queue', json=text_graph())
        assert response.status_code == 503, response.text
        error = response.json()['error']
        assert error['code'] == 'editor_admission_unconfirmed'
        assert 'accepted' not in error
        identity = error['task_id']
        assert server.queue.get(timeout=.01) is None
        monkeypatch.setattr(server.queue.history_store, 'admit', original)
        server.queue.recover_editor()
        idx, payload = server.queue.get(timeout=.1)
        assert payload['id'] == identity
        server.queue.task_done(idx, ExecutionStatus(True), {}, {})
        assert server.queue.get(timeout=.01) is None


def test_editor_restores_all_startup_batches_in_priority_and_admission_order(tmp_path, location):
    path = tmp_path / 'editor'
    first = editor(path, location)
    expected = []
    with TestClient(first.app):
        for index in range(105):
            identity = str(uuid.uuid4())
            priority = index % 3
            first.queue.put({'id': identity, 'graph_json': text_graph()}, priority)
            expected.append((priority, index, identity))
    restored = editor(path, location)
    with TestClient(restored.app):
        observed = []
        for _ in expected:
            idx, payload = restored.queue.get(timeout=.1)
            observed.append(payload['id'])
            restored.queue.task_done(idx, ExecutionStatus(True), {}, {})
        assert observed == [item[2] for item in sorted(expected)]
        assert restored.queue.get(timeout=.01) is None


@pytest.mark.parametrize('proof', ['missing', 'copied'])
def test_editor_does_not_recover_without_matching_physical_exit_evidence(tmp_path, location, proof):
    path = tmp_path / 'editor'
    first = editor(path, location)
    with TestClient(first.app) as client:
        identity = client.post('/api/queue', json=text_graph()).json()['result']['task_id']
        guard = first.queue.history_store.guard.path
    original = guard.with_suffix('.original')
    guard.rename(original)
    if proof == 'copied':
        guard.write_bytes(original.read_bytes())
    restored = editor(path, location)
    with TestClient(restored.app) as client:
        restored.queue.recover_editor()
        assert restored.queue.get(timeout=.01) is None
        assert restored.queue.get_task_by_id(identity)['state'] == 'PENDING'
        assert client.get('/api/queue/history').json()['result']['total'] == 0


def test_editor_concurrent_recovery_assigns_each_request_once(tmp_path, location):
    from concurrent.futures import ThreadPoolExecutor
    path = tmp_path / 'editor'
    first = editor(path, location)
    with TestClient(first.app):
        identities = [str(uuid.uuid4()) for _ in range(8)]
        for identity in identities:
            first.queue.put({'id': identity, 'graph_json': text_graph()})
    with ThreadPoolExecutor(max_workers=2) as pool:
        servers = list(pool.map(lambda _: editor(path, location), range(2)))
    with TestClient(servers[0].app), TestClient(servers[1].app):
        observed = []
        for server in servers:
            while True:
                item = server.queue.get(timeout=.01)
                if item is None:
                    break
                idx, payload = item
                observed.append(payload['id'])
                server.queue.task_done(idx, ExecutionStatus(True), {}, {})
        assert sorted(observed) == sorted(identities)


def test_real_editor_worker_holds_exit_evidence_while_graph_is_running(tmp_path, monkeypatch):
    from test_editor_task_control import running_editor, held_graph, submit, entered, terminal
    with running_editor(tmp_path, monkeypatch) as runtime:
        control = tmp_path / 'control'
        identity = submit(runtime, held_graph(control))
        entered(runtime, identity, control)
        store = runtime.server.queue.history_store
        # Drop only the parent's lock, as would happen on its physical exit.
        store.guard.close()
        with ControlGuard.recover(store.directory, store.token, store.holder) as exited:
            assert not exited, 'The real computing Worker must independently hold the evidence'
        (control / 'release').touch()
        assert terminal(runtime, identity)['state'] == 'SUCCEEDED'


def test_editor_startup_failure_releases_recovered_task_ownership(tmp_path, location, monkeypatch):
    from nndeploy.server.template import WorkflowTemplateManager
    path = tmp_path / 'editor'
    first = editor(path, location)
    with TestClient(first.app) as client:
        identity = client.post('/api/queue', json=text_graph()).json()['result']['task_id']
    original = WorkflowTemplateManager.init_templates
    def fail():
        raise RuntimeError('Controlled editor startup failure')
    monkeypatch.setattr(WorkflowTemplateManager, 'init_templates', fail)
    with pytest.raises(RuntimeError, match='Controlled editor startup'):
        editor(path, location)
    monkeypatch.setattr(WorkflowTemplateManager, 'init_templates', original)
    restored = editor(path, location)
    with TestClient(restored.app):
        item = restored.queue.get(timeout=.01)
        assert item is not None, 'A failed Server constructor must release its recovered admission'
        idx, payload = item
        assert payload['id'] == identity
        restored.queue.task_done(idx, ExecutionStatus(True), {}, {})


def test_editor_remote_cancel_reaches_tasks_beyond_first_collection_batch(tmp_path, location):
    from nndeploy.server.app import _EditorJobQueue
    path = tmp_path / 'editor'
    first = editor(path, location)
    jobs = _EditorJobQueue()
    first.queue._job_q = jobs
    with TestClient(first.app):
        identities = []
        for _ in range(20):
            identity = str(uuid.uuid4())
            identities.append(identity)
            first.queue.put({'id': identity, 'graph_json': text_graph()})
            idx, payload = first.queue.get(timeout=.1)
            first.queue.dispatch(idx, payload, jobs)
        second = editor(path, location)
        with TestClient(second.app) as client:
            response = client.post('/api/queue/cancel/' + identities[-1])
            assert response.status_code == 200, response.text
            assert response.json()['result']['state'] == 'DISPATCHED'
            first.queue.recover_editor()
            first.queue.collect_results()
            assert first.queue.get_task_by_id(identities[-1])['state'] == 'CANCELLED'
            assert all(first.queue.get_task_by_id(identity)['state'] == 'DISPATCHED' for identity in identities[:-1])
