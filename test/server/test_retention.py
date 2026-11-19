"""Retention is a durable task policy, independent of the cleaning process."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import uuid

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import event, select, update

from nndeploy.server import Server
from nndeploy.server.config import ExecutionConfig
from nndeploy.server.db import tasks, file_refs
from nndeploy.server.errors import ServerError
from nndeploy.server.streaming import TaskEventResponse
from test_db import db


def expire(database, *identities):
    with database.engine.begin() as connection:
        connection.execute(update(tasks).where(tasks.c.task_id.in_(identities)).values(purge_after=database.clock(connection) - 1))


def create(database, *, retention=60, name=None, inputs=None, key=None):
    identity = 'task_' + uuid.uuid4().hex
    database.create_task(identity, name or uuid.uuid4().hex, time.time(), inputs, {},
                         idempotency_key=key, retention_seconds=retention)
    return identity


def test_cleanup_atomically_releases_results_events_idempotency_and_task_file_refs(db):
    file = 'file_' + uuid.uuid4().hex
    db.create_file(dict(file_id=file, storage_key=file, media_type='text/plain', size=1, created_at=1))
    name, inputs = uuid.uuid4().hex, {'file': {'file_id': file}}
    identity = create(db, name=name, inputs=inputs, key='retry')
    db.start_task(identity, time.time())
    db.append_task_event(identity, {'value': 1})
    before = db.clock()
    db.finish_task(identity, 'succeeded', -1000, outputs=inputs)
    record = db.get_task(identity)
    assert before + 60 <= record['purge_after'] <= db.clock() + 60
    assert db.find_task(name, 'retry', inputs, {})['task_id'] == identity
    assert db.cleanup_tasks() == 0
    with pytest.raises(ServerError, match='referenced'):
        db.mark_file_deleting(file)
    expire(db, identity)
    assert db.cleanup_tasks() == 1
    assert db.get_task(identity) is None
    assert db.read_task_events(identity) == []
    with db.engine.connect() as connection:
        assert connection.execute(select(file_refs).where(file_refs.c.task_id == identity)).first() is None
    assert db.find_task(name, 'retry', inputs, {}) is None
    # A result expiring is not authorization to destroy its file content.
    assert db.get_file(file)['state'] == 'ready'
    replacement = create(db, name=name, inputs=inputs, key='retry')
    assert replacement != identity
    db.finish_task(replacement, 'cancelled', time.time())
    expire(db, replacement)
    assert db.cleanup_tasks() == 1
    db.mark_file_deleting(file)
    db.delete_file(file)


def test_pending_running_cancel_requested_and_unexpired_records_survive(db):
    pending = create(db)
    running = create(db)
    db.start_task(running, time.time())
    db.request_cancel(running)
    # Even an invalid old deadline must never permit deleting active execution.
    expire(db, pending, running)
    recent = create(db)
    permanent = create(db, retention=None)
    for identity in (recent, permanent):
        db.finish_task(identity, 'succeeded', 0, {})
    assert db.get_task(permanent)['purge_after'] is None
    assert db.cleanup_tasks() == 0
    for identity in (pending, running, recent, permanent):
        assert db.get_task(identity) is not None
    for identity in (pending, running):
        db.finish_task(identity, 'cancelled', time.time())
        assert db.get_task(identity)['purge_after'] > db.clock()


def test_cleanup_is_bounded_concurrent_and_uses_all_terminal_states(db):
    identities = []
    for state in ('succeeded', 'failed', 'cancelled', 'expired') * 5:
        identity = create(db)
        db.finish_task(identity, state, time.time())
        identities.append(identity)
    expire(db, *identities)
    assert db.cleanup_tasks(limit=3) == 3
    barrier, lock, selections = threading.Barrier(4), threading.Lock(), 0
    def collide(connection, cursor, statement, parameters, context, executemany):
        nonlocal selections
        if statement.startswith('SELECT server_tasks.task_id') and 'purge_after' in statement and 'LIMIT' in statement:
            with lock:
                selections += 1
                first_wave = selections <= 4
            if first_wave:
                barrier.wait(timeout=5)
    event.listen(db.engine, 'after_cursor_execute', collide)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            counts = list(pool.map(lambda _: db.cleanup_tasks(limit=5), range(8)))
    finally:
        event.remove(db.engine, 'after_cursor_execute', collide)
    # Competing cleaners can select the same rows and later delete zero. A
    # fixed number of concurrent attempts does not promise to drain the table.
    remaining = sum(db.get_task(identity) is not None for identity in identities)
    assert min(counts) >= 0 and max(counts) <= 5 and sum(counts) + remaining == 17
    for _ in range(4):
        counts.append(db.cleanup_tasks(limit=5))
    assert max(counts) <= 5 and sum(counts) == 17
    assert all(db.get_task(identity) is None for identity in identities)
    assert db.cleanup_tasks() == 0


def test_failed_cleanup_rolls_back_events_and_references(db):
    file = 'file_' + uuid.uuid4().hex
    db.create_file(dict(file_id=file, storage_key=file, media_type='text/plain', size=1, created_at=1))
    identity = create(db, key='retained-on-error', inputs={'file_id': file})
    db.finish_task(identity, 'succeeded', time.time(), {'text': 'still here'})
    expire(db, identity)
    def fail_after_delete(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith('DELETE FROM server_tasks'):
            raise RuntimeError('injected post-delete error')
    event.listen(db.engine, 'after_cursor_execute', fail_after_delete)
    try:
        with pytest.raises(RuntimeError, match='injected'):
            db.cleanup_tasks()
    finally:
        event.remove(db.engine, 'after_cursor_execute', fail_after_delete)
    assert db.get_task(identity)['outputs'] == {'text': 'still here'}
    assert [row['kind'] for row in db.read_task_events(identity)] == ['result', 'done']
    with pytest.raises(ServerError, match='referenced'):
        db.mark_file_deleting(file)
    assert db.cleanup_tasks() == 1
    db.delete_file(file)


@pytest.mark.parametrize('seconds', [0, 59, -1, float('inf'), float('nan'), True])
def test_retention_rejects_invalid_periods(seconds):
    with pytest.raises(ValueError):
        ExecutionConfig(task_retention_seconds=seconds)


def test_http_cleanup_requires_management_and_expiry_allows_new_idempotent_request(tmp_path):
    calls = []
    def deploy(inputs):
        calls.append(inputs)
        return inputs
    server = Server(name='echo', deploy=deploy, resources=tmp_path, data_dir=tmp_path, admin_token='admin',
                    config={'inputs': {'text': {'type': 'string'}}, 'execution': {'task_retention_seconds': 60.0}})
    headers = {'Idempotency-Key': 'same'}
    body = {'inputs': {'text': 'hello'}}
    with TestClient(server.app) as client:
        first = client.post('/v1/server/echo/deploy', json=body, headers=headers).json()
        identity = first['task_id']
        row = client.get('/v1/tasks/' + identity).json()
        assert row['purge_after'] >= row['finished_at'] + 59
        assert client.get('/v1/server/echo/schema').json()['task_retention_seconds'] == 60
        assert client.post('/v1/server/echo/deploy', json=body, headers=headers).json() == first
        expire(server.db, identity)
        assert client.post('/api/server/cleanup-tasks').status_code == 401
        assert server.db.get_task(identity) is not None
        result = client.post('/api/server/cleanup-tasks', headers={'Authorization': 'Bearer admin'})
        assert result.status_code == 200 and result.json() == {'deleted_tasks': 1}
        assert client.get('/v1/tasks/' + identity).status_code == 404
        assert client.get('/v1/tasks/' + identity + '/events').status_code == 404
        assert server.entries['echo']['queue'].get_history() == {}
        repeated = client.post('/v1/server/echo/deploy', json=body, headers=headers).json()
        assert repeated['task_id'] != identity
        assert len(calls) == 2


def test_restart_cleans_retained_tasks_with_original_policy(tmp_path):
    server = Server(name='echo', deploy=lambda inputs: inputs, resources=tmp_path, data_dir=tmp_path,
                    config={'execution': {'task_retention_seconds': 60.0}})
    with TestClient(server.app) as client:
        identity = client.post('/v1/server/echo/deploy', json={}).json()['task_id']
        expire(server.db, identity)
    # No entry or retention configuration on the replacement process.
    reopened = Server(resources=tmp_path, data_dir=tmp_path)
    with TestClient(reopened.app):
        deadline = time.monotonic() + 5
        while reopened.db.get_task(identity) is not None:
            assert time.monotonic() < deadline
            time.sleep(.01)


def test_shutdown_waits_for_current_cleanup_transaction(tmp_path, monkeypatch):
    server = Server(resources=tmp_path, data_dir=tmp_path)
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    original_close = server.db.close
    def cleanup():
        entered.set()
        assert release.wait(5)
        assert not closed.is_set()
        return 0
    def close():
        closed.set()
        original_close()
    monkeypatch.setattr(server.db, 'cleanup_tasks', cleanup)
    monkeypatch.setattr(server.db, 'close', close)
    async def scenario():
        await server.start()
        assert await asyncio.to_thread(entered.wait, 5)
        stopping = asyncio.create_task(server.close())
        try:
            await asyncio.sleep(.03)
            assert not stopping.done() and not closed.is_set()
        finally:
            release.set()
            await asyncio.wait_for(stopping, 5)
        assert closed.is_set()
    asyncio.run(scenario())


def test_background_cleanup_recovers_after_database_error(tmp_path, monkeypatch, caplog):
    server = Server(resources=tmp_path, data_dir=tmp_path)
    attempts = []
    original = server.db.cleanup_tasks
    def cleanup():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError('temporary database failure')
        return original()
    monkeypatch.setattr(server.db, 'cleanup_tasks', cleanup)
    async def scenario():
        server.task_maintenance = asyncio.create_task(server._maintain_tasks(interval=.01))
        try:
            async def recovered():
                while len(attempts) < 2:
                    await asyncio.sleep(.01)
            await asyncio.wait_for(recovered(), 5)
        finally:
            await server.close()
    asyncio.run(scenario())
    assert 'Task retention cleanup failed' in caplog.text


def test_sse_reports_retention_loss_without_fabricating_task_terminal_state(tmp_path):
    server = Server(resources=tmp_path, data_dir=tmp_path)
    async def scenario():
        response = TaskEventResponse(server, 'gone', reserved=True)
        chunks = [chunk async for chunk in response.events('gone', 0)]
        assert b'"code":"task_not_found"' in b''.join(chunks)
        assert b'"terminal":false' in b''.join(chunks)
        assert b'event: done' not in b''.join(chunks)
        await server.close()
    asyncio.run(scenario())
