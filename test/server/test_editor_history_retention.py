"""History expiration removes output bytes without replaying an old editor request."""
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update

from nndeploy.server import Server
from nndeploy.server.db import editor_history, editor_tasks
from nndeploy.server.editor_outputs import capture_files
from nndeploy.server.task_queue import ExecutionStatus
from test_editor_history import location
from test_server_api import text_graph


def make_server(path, location, retention=None):
    return Server(SimpleNamespace(resources=str(path), debug=True), database_url=location,
                  editor_history_retention_seconds=retention)


def finish(client, server, path):
    graph = text_graph()
    graph['node_repository_'][0]['str_'] = 'private input body'
    key = {'Idempotency-Key': str(uuid.uuid4())}
    response = client.post('/api/queue', json=graph, headers=key)
    assert response.status_code == 202, response.text
    identity = response.json()['result']['task_id']
    idx, payload = server.queue.get(timeout=1)
    path.write_text('saved output')
    file_graph = {'node_repository_': [{'name_': 'file', 'node_type_': 'Output', 'io_type_': 'Text',
                  'io_param_': ['path'], 'path': str(path)}]}
    file = capture_files(file_graph, server.storage, payload['_editor_scope'], identity, payload['_editor_token'])[0]
    server.queue.task_done(idx, ExecutionStatus(True, 'Done'), {'sink': {'String': ['saved text']}}, {})
    return identity, key, graph, file


def age(server, identity):
    with server.db.engine.begin() as conn:
        conn.execute(update(editor_history).where(editor_history.c.scope == server.config_scope,
            editor_history.c.task_id == identity).values(finished_at=server.db.clock(conn) - 120))


def test_expired_history_releases_outputs_but_preserves_exact_request_identity(tmp_path, location):
    path = tmp_path / 'editor'
    server = make_server(path, location, 60)
    with TestClient(server.app) as client:
        identity, key, graph, file = finish(client, server, tmp_path / 'output.txt')
        age(server, identity)
        result = client.post('/api/queue/history/cleanup')
        assert result.status_code == 200, result.text
        assert result.json()['result']['deleted_history'] == 1
        assert client.get('/api/queue/history').json()['result']['items'] == []
        status = client.get('/api/queue/task/' + identity).json()['result']
        assert status['state'] == 'SUCCEEDED' and status['outputs_expired'] is True
        assert status['outputs'] == {} and status['output_files'] == []
        assert client.get(file['url']).status_code == 404
        server.storage.cleanup()
        assert server.db.get_file(file['file_id']) is None
        assert client.post('/api/queue', json=graph, headers=key).status_code == 202
        assert server.queue.get_current_queue()['PENDING'] == []
        assert client.post('/api/queue', json=dict(graph, name_='changed'), headers=key).status_code == 409
        assert client.post('/api/queue/history/cleanup').json()['result']['deleted_history'] == 0
        with server.db.engine.connect() as conn:
            payload = conn.execute(select(editor_tasks.c.payload).where(editor_tasks.c.scope == server.config_scope,
                editor_tasks.c.task_id == identity)).scalar_one()
            assert 'private input body' not in payload and 'saved text' not in payload
            assert len(payload) < 2048
    with TestClient(make_server(path, location).app) as client:
        assert client.post('/api/queue', json=graph, headers=key).status_code == 202
        assert client.get('/api/queue/task/' + identity).json()['result']['outputs_expired'] is True


def test_history_cleanup_is_opt_in_and_does_not_touch_active_or_other_workspaces(tmp_path, location):
    server = make_server(tmp_path / 'editor', location)
    with TestClient(server.app) as client:
        identity, _, _, file = finish(client, server, tmp_path / 'output.txt')
        age(server, identity)
        assert client.post('/api/queue/history/cleanup').json()['result']['deleted_history'] == 0
        pending = client.post('/api/queue', json=text_graph()).json()['result']['task_id']
        assert client.get(file['url']).status_code == 200
        server.editor_history_retention_seconds = 60
        assert client.post('/api/queue/history/cleanup').json()['result']['deleted_history'] == 1
        assert client.get('/api/queue/task/' + pending).json()['result']['state'] == 'PENDING'
    with TestClient(make_server(tmp_path / 'other', location, 60).app) as client:
        assert client.get('/api/queue/task/' + identity).status_code == 404
        assert client.post('/api/queue/history/cleanup').json()['result']['deleted_history'] == 0


def test_expiration_by_another_instance_clears_cached_results(tmp_path, location):
    path = tmp_path / 'editor'
    first, second = make_server(path, location), make_server(path, location, 60)
    with TestClient(first.app) as a, TestClient(second.app) as b:
        identity, key, graph, file = finish(a, first, tmp_path / 'output.txt')
        age(first, identity)
        assert identity in first.queue._hist
        assert b.post('/api/queue/history/cleanup').json()['result']['deleted_history'] == 1
        assert a.get('/api/queue/task/' + identity).json()['result']['outputs_expired'] is True
        assert a.get(file['url']).status_code == 404
        assert a.post('/api/queue', json=graph, headers=key).status_code == 202
        assert first.queue.get_current_queue()['PENDING'] == []
        assert identity not in first.queue._hist


def test_lost_completion_receipt_cannot_resurrect_expired_history(tmp_path, location, monkeypatch):
    server = make_server(tmp_path / 'editor', location, 60)
    with TestClient(server.app) as client:
        key = {'Idempotency-Key': str(uuid.uuid4())}
        identity = client.post('/api/queue', json=text_graph(), headers=key).json()['result']['task_id']
        idx, _ = server.queue.get(timeout=1)
        save = server.queue.history_store.save
        def lost(*args):
            save(*args)
            raise OSError('Lost completion receipt')
        monkeypatch.setattr(server.queue.history_store, 'save', lost)
        result, status = {'sink': {'String': ['large original output']}}, ExecutionStatus(True, 'Done')
        with pytest.raises(OSError, match='Lost completion'):
            server.queue.task_done(idx, status, result, {})
        age(server, identity)
        assert client.post('/api/queue/history/cleanup').json()['result']['deleted_history'] == 1
        monkeypatch.setattr(server.queue.history_store, 'save', save)
        server.queue.task_done(idx, status, result, {})
        assert client.get('/api/queue/task/' + identity).json()['result']['outputs_expired'] is True
        assert client.get('/api/queue/history').json()['result']['items'] == []
        assert identity not in server.queue._hist


def test_concurrent_cleaners_and_retries_keep_one_receipt(tmp_path, location):
    from concurrent.futures import ThreadPoolExecutor
    path = tmp_path / 'editor'
    first, second = make_server(path, location, 60), make_server(path, location, 60)
    with TestClient(first.app) as a, TestClient(second.app) as b:
        identity, key, graph, file = finish(a, first, tmp_path / 'output.txt')
        age(first, identity)
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit((a if i % 2 else b).post, '/api/queue/history/cleanup') for i in range(4)]
            retries = [pool.submit((a if i % 2 else b).post, '/api/queue', json=graph, headers=key) for i in range(8)]
            assert sum(f.result().json()['result']['deleted_history'] for f in futures) == 1
            assert all(f.result().status_code == 202 for f in retries)
        assert not first.queue.get_current_queue()['PENDING'] and not second.queue.get_current_queue()['PENDING']
        assert a.get('/api/queue/task/' + identity).json()['result']['outputs_expired'] is True


@pytest.mark.parametrize('value', [0, 59, True, float('nan'), float('inf')])
def test_history_retention_rejects_invalid_policy(tmp_path, value):
    with pytest.raises(ValueError, match='retention'):
        make_server(tmp_path, None, value)


def test_history_cleanup_rolls_back_file_release_and_receipt_together(tmp_path, location):
    from sqlalchemy import event
    server = make_server(tmp_path / 'editor', location, 60)
    with TestClient(server.app) as client:
        identity, key, graph, file = finish(client, server, tmp_path / 'output.txt')
        age(server, identity)
        def fail(conn, cursor, statement, parameters, context, many):
            if statement.lstrip().startswith('DELETE FROM server_editor_history'):
                raise OSError('Controlled cleanup transaction failure')
        event.listen(server.db.engine, 'after_cursor_execute', fail)
        try:
            with pytest.raises(OSError, match='Controlled cleanup'):
                server.queue.cleanup_history(60)
        finally:
            event.remove(server.db.engine, 'after_cursor_execute', fail)
        assert client.get(file['url']).content == b'saved output'
        assert client.get('/api/queue/task/' + identity).json()['result']['outputs'] == {'sink': 'saved text'}
        assert client.post('/api/queue', json=graph, headers=key).status_code == 202
        server.storage.cleanup()
        assert server.db.get_file(file['file_id']) is not None
        assert client.post('/api/queue/history/cleanup').json()['result']['deleted_history'] == 1


def test_retention_preserves_inflight_download_until_reader_releases_it(tmp_path, location):
    server = make_server(tmp_path / 'editor', location, 60)
    with TestClient(server.app) as client:
        identity, _, _, file = finish(client, server, tmp_path / 'output.txt')
        age(server, identity)
        with server.storage.open(file['file_id']) as content:
            assert client.post('/api/queue/history/cleanup').json()['result']['deleted_history'] == 1
            server.storage.cleanup()
            assert content.read() == b'saved output'
            assert server.db.get_file(file['file_id']) is not None
        server.storage.cleanup()
        assert server.db.get_file(file['file_id']) is None


def test_retention_is_applied_automatically_after_restart(tmp_path, location):
    from test_editor_queue_recovery import wait
    path = tmp_path / 'editor'
    server = make_server(path, location)
    with TestClient(server.app) as client:
        identity, key, graph, file = finish(client, server, tmp_path / 'output.txt')
        age(server, identity)
    server = make_server(path, location, 60)
    with TestClient(server.app) as client:
        wait(lambda: client.get('/api/queue/task/' + identity).json()['result'].get('outputs_expired'))
        wait(lambda: server.db.get_file(file['file_id']) is None)
        assert client.post('/api/queue', json=graph, headers=key).status_code == 202
        assert not server.queue.get_current_queue()['PENDING']


def test_pre_fingerprint_terminal_record_and_history_only_record_keep_idempotency(tmp_path, location):
    from sqlalchemy import delete
    server = make_server(tmp_path / 'editor', location, 60)
    with TestClient(server.app) as client:
        identities = []
        for history_only in (False, True):
            key = {'Idempotency-Key': str(uuid.uuid4())}
            graph = text_graph()
            identity = client.post('/api/queue', json=graph, headers=key).json()['result']['task_id']
            idx, _ = server.queue.get(timeout=1)
            server.queue.task_done(idx, ExecutionStatus(True, 'Done'), {}, {})
            with server.db.engine.begin() as conn:
                condition = (editor_tasks.c.scope == server.config_scope) & (editor_tasks.c.task_id == identity)
                if history_only:
                    conn.execute(delete(editor_tasks).where(condition))
                else:
                    conn.execute(update(editor_tasks).where(condition).values(payload='{}'))
            if not history_only:
                assert client.post('/api/queue', json=graph, headers=key).status_code == 202
            age(server, identity)
            identities.append((identity, key, graph))
        assert client.post('/api/queue/history/cleanup').json()['result']['deleted_history'] == 2
        for identity, key, graph in identities:
            assert client.post('/api/queue', json=graph, headers=key).status_code == 202
            assert client.get('/api/queue/task/' + identity).json()['result']['outputs_expired'] is True
        assert not server.queue.get_current_queue()['PENDING']


def test_host_clock_skew_does_not_expire_a_just_completed_task(tmp_path, location, monkeypatch):
    server = make_server(tmp_path / 'editor', location, 60)
    with TestClient(server.app) as client:
        save = server.queue.history_store.save
        def skewed(identity, record):
            return save(identity, dict(record, ts_finish=1))
        monkeypatch.setattr(server.queue.history_store, 'save', skewed)
        before = server.db.clock()
        identity, _, _, file = finish(client, server, tmp_path / 'output.txt')
        record = client.get('/api/queue/task/' + identity).json()['result']
        assert before <= record['ts_finish'] <= server.db.clock()
        assert client.post('/api/queue/history/cleanup').json()['result']['deleted_history'] == 0
        assert client.get(file['url']).status_code == 200
