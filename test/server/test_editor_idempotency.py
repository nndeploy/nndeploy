"""Editor retries keep the original Task across owners, phases and lost receipts."""
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from nndeploy.server.task_queue import ExecutionStatus
from test_editor_history import editor, location
from test_server_api import text_graph


def headers():
    return {'Idempotency-Key': str(uuid.uuid4())}


@pytest.mark.parametrize('phase', ['pending', 'dispatched', 'running', 'success', 'cancelled'])
def test_editor_same_request_replays_every_phase_and_after_restart(tmp_path, location, phase):
    path = tmp_path / 'editor'
    key, graph = headers(), text_graph()
    server = editor(path, location)
    with TestClient(server.app) as client:
        response = client.post('/api/queue', json=graph, headers=key)
        assert response.status_code == 202, response.text
        identity = response.json()['result']['task_id']
        assert identity == key['Idempotency-Key']
        if phase in ('dispatched', 'running', 'success'):
            idx, _ = server.queue.get(timeout=1)
            server.queue.mark_dispatched(idx)
            if phase != 'dispatched':
                server.queue.mark_started(identity, worker_pid=123)
            if phase == 'success':
                server.queue.task_done(idx, ExecutionStatus(True, 'Done'), {}, {})
        if phase == 'cancelled':
            client.post('/api/queue/cancel/' + identity)
        # An existing request must be replayable even when new admission is closed.
        server.queue.pause_editor()
        server.queue.capacity = 0
        assert client.post('/api/queue', json=graph, headers=key).json() == response.json()
    recreated = editor(path, location)
    with TestClient(recreated.app) as client:
        replay = client.post('/api/queue', json=graph, headers=key)
        assert replay.status_code == 202, replay.text
        assert replay.json() == response.json()
        expected_pending = 1 if phase == 'pending' else 0
        assert len(recreated.queue.get_current_queue()['PENDING']) == expected_pending
        modified = dict(graph, name_='different')
        conflict = client.post('/api/queue', json=modified, headers=key)
        assert conflict.status_code == 409, conflict.text


def test_editor_concurrent_owners_do_not_enqueue_the_same_request_twice(tmp_path, location):
    path = tmp_path / 'editor'
    first, second = editor(path, location), editor(path, location)
    key, graph = headers(), text_graph()
    with TestClient(first.app) as a, TestClient(second.app) as b:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit((a if i % 2 else b).post, '/api/queue', json=graph, headers=key)
                       for i in range(16)]
            replies = [future.result() for future in futures]
        assert all(reply.status_code == 202 for reply in replies), [r.text for r in replies]
        assert {r.json()['result']['task_id'] for r in replies} == {key['Idempotency-Key']}
        assert len(first.queue._pq) + len(second.queue._pq) == 1


@pytest.mark.parametrize('committed', [False, True])
def test_editor_retry_after_missing_database_receipt(tmp_path, location, monkeypatch, committed):
    server = editor(tmp_path / 'editor', location)
    key, graph = headers(), text_graph()
    store = server.queue.history_store
    original = store.admit
    def fail(*args, **kwargs):
        if committed:
            original(*args, **kwargs)
        raise OperationalError('admit', {}, RuntimeError('lost receipt'))
    with TestClient(server.app) as client:
        monkeypatch.setattr(store, 'admit', fail)
        first = client.post('/api/queue', json=graph, headers=key)
        assert first.status_code == 503
        assert first.json()['error']['task_id'] == key['Idempotency-Key']
        monkeypatch.setattr(store, 'admit', original)
        retry = client.post('/api/queue', json=graph, headers=key)
        assert retry.status_code == 202, retry.text
        assert len(server.queue._pq) == 1
        assert server.queue._pq[0][3]['id'] == key['Idempotency-Key']


def test_editor_key_validation_and_legacy_requests(tmp_path, location):
    with TestClient(editor(tmp_path / 'editor', location).app) as client:
        for invalid in ['', 'not-a-uuid', 'a' * 500, 'A0000000-0000-4000-8000-000000000000']:
            assert client.post('/api/queue', json=text_graph(), headers={'Idempotency-Key': invalid}).status_code == 422
        a = client.post('/api/queue', json=text_graph()).json()['result']['task_id']
        b = client.post('/api/queue', json=text_graph()).json()['result']['task_id']
        assert a != b


def test_editor_new_admission_rejection_does_not_reserve_identity(tmp_path, location):
    server = editor(tmp_path / 'editor', location)
    key, graph = headers(), text_graph()
    with TestClient(server.app) as client:
        server.queue.capacity = 0
        full = client.post('/api/queue', json=graph, headers=key)
        assert full.status_code == 429, full.text
        assert full.json()['error']['accepted'] is False
        assert client.get('/api/queue/task/' + key['Idempotency-Key']).status_code == 404
        server.queue.capacity = None
        server.queue.pause_editor()
        paused = client.post('/api/queue', json=graph, headers=key)
        assert paused.status_code == 503
        assert paused.json()['error']['accepted'] is False
        server.queue.resume_editor(None)
        assert client.post('/api/queue', json=graph, headers=key).status_code == 202
        server.queue.capacity = 0
        assert client.post('/api/queue', json=graph, headers=key).status_code == 202
        assert len(server.queue._pq) == 1


def test_editor_idempotency_is_scoped_to_the_workspace(tmp_path, location):
    key, graph = headers(), text_graph()
    with TestClient(editor(tmp_path / 'first', location).app) as a, TestClient(editor(tmp_path / 'second', location).app) as b:
        assert a.post('/api/queue', json=graph, headers=key).status_code == 202
        other = dict(graph, name_='different')
        assert b.post('/api/queue', json=other, headers=key).status_code == 202


def test_editor_native_graph_executes_once_when_run_requests_are_replayed(tmp_path, monkeypatch):
    from test_editor_task_control import running_editor, held_graph, entered, terminal
    with running_editor(tmp_path, monkeypatch) as runtime:
        directory = tmp_path / 'control'
        graph, key = held_graph(directory), headers()
        identity = key['Idempotency-Key']
        response = runtime.client.post('/api/queue', json=graph, headers=key)
        assert response.status_code == 202, response.text
        entered(runtime, identity, directory)
        for _ in range(4):
            assert runtime.client.post('/api/queue', json=graph, headers=key).json() == response.json()
        (directory / 'release').touch()
        assert terminal(runtime, identity)['state'] == 'SUCCEEDED'
        for _ in range(4):
            assert runtime.client.post('/api/queue', json=graph, headers=key).json() == response.json()
        assert not runtime.server.queue._pq
        assert (directory / 'entered').read_text().count('run') == 1
