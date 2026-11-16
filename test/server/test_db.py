"""The same transactions run against SQLite and explicitly supplied local DBs."""
import json
import os
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import insert, select, update
from sqlalchemy.exc import IntegrityError

from nndeploy.server.db import DB, tasks, workflows
from nndeploy.server.errors import ServerError

URLS = ['sqlite'] + json.loads(os.environ.get('NNDEPLOY_TEST_DATABASE_URLS', '[]'))


@pytest.fixture(params=URLS, ids=lambda url: url.split(':')[0])
def db(request, tmp_path):
    value = DB(tmp_path / 'server.db' if request.param == 'sqlite' else request.param,
               backup_dir=tmp_path / 'db-backups').init_schema()
    yield value
    value.close()


def test_concurrent_idempotency_and_different_payload_conflict(db):
    name = uuid.uuid4().hex
    def create(index):
        return db.create_task(uuid.uuid4().hex, name, time.time(), {'text': '中文'}, {}, idempotency_key='same')
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(create, range(20)))
    assert len(set(ids)) == 1
    with pytest.raises(ServerError) as conflict:
        db.create_task(uuid.uuid4().hex, name, time.time(), {'text': 'different'}, {}, idempotency_key='same')
    assert conflict.value.status_code == 409
    row = db.get_task(ids[0])
    assert row['inputs'] == {'text': '中文'}
    assert row['state'] == 'pending'


def test_terminal_and_stale_attempt_cannot_overwrite_result(db):
    identity = uuid.uuid4().hex
    db.create_task(identity, 'test', time.time())
    db.start_task(identity, time.time())
    with db.engine.begin() as connection:
        connection.execute(update(tasks).where(tasks.c.task_id == identity).values(attempt=2))
    with pytest.raises(RuntimeError):
        db.finish_task(identity, 'succeeded', time.time(), {'stale': True})
    db.finish_task(identity, 'succeeded', time.time(), {'good': True}, attempt=2)
    with pytest.raises(RuntimeError):
        db.finish_task(identity, 'failed', time.time(), attempt=2)
    assert db.get_task(identity)['outputs'] == {'good': True}
    assert not db.request_cancel(identity)['cancel_requested']


def test_stream_events_are_owned_bounded_and_end_with_one_atomic_terminal(db):
    identity = uuid.uuid4().hex
    db.create_task(identity, 'stream', time.time())
    db.start_task(identity, time.time())
    # More than MySQL TEXT's capacity, within the declared event limit.
    assert db.append_task_event(identity, {'text': 'x' * 70000}, max_bytes=100000) == 2
    with pytest.raises(ServerError) as stale:
        db.append_task_event(identity, {}, worker_token='wrong-owner')
    assert stale.value.code == 'execution_ownership_lost'
    with pytest.raises(ServerError) as size:
        db.append_task_event(identity, {'text': 'x' * 100}, max_bytes=64)
    assert size.value.code == 'stream_event_too_large'
    assert db.get_task(identity)['event_seq'] == 2
    db.request_cancel(identity)
    with pytest.raises(ServerError) as cancelled:
        db.append_task_event(identity, {})
    assert cancelled.value.code == 'task_cancelled'
    db.finish_task(identity, 'cancelled', time.time())
    with pytest.raises(RuntimeError):
        db.finish_task(identity, 'succeeded', time.time())
    events = db.read_task_events(identity)
    assert [(event['sequence'], event['kind']) for event in events] == [(1, 'start'), (2, 'delta'), (3, 'error'), (4, 'done')]
    assert events[-1]['data']['state'] == db.get_task(identity)['state'] == 'cancelled'


def test_workflow_concurrent_upsert_preserves_identity(db, tmp_path):
    path = tmp_path / (uuid.uuid4().hex + '.json')
    path.write_text('{}')
    with ThreadPoolExecutor(max_workers=8) as pool:
        identities = list(pool.map(lambda _: db.upsert_workflow_by_path(path, '中文'), range(16)))
    assert len(set(identities)) == 1
    assert db.get_workflow_path(identities[0]) == path
    assert db.get_workflow_name(identities[0]) == '中文'
    assert db.delete_workflow(identities[0])
    assert db.get_workflow_path(identities[0]) is None


def test_transaction_rollback_and_file_reconnect(db):
    identity = uuid.uuid4().hex
    with pytest.raises(IntegrityError):
        with db.engine.begin() as connection:
            connection.execute(insert(workflows).values(id=identity, path=identity))
            connection.execute(insert(workflows).values(id=identity, path='duplicate'))
    assert db.get_workflow_path(identity) is None
    record = dict(file_id=identity, storage_key=identity, media_type='image/png', size=12, created_at=time.time())
    db.create_file(record)
    db.engine.dispose()
    assert all(db.get_file(identity)[key] == value for key, value in record.items())
    db.delete_file(identity)
    assert db.get_file(identity) is None


def test_migrate_existing_sqlite_preserves_workflow_and_early_tasks(tmp_path):
    path = tmp_path / 'legacy.db'
    with sqlite3.connect(path) as conn:
        conn.executescript('''
            CREATE TABLE workflows (id TEXT PRIMARY KEY, path TEXT NOT NULL UNIQUE, name TEXT,
              ext TEXT, size INTEGER, cover TEXT, requirements TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
            INSERT INTO workflows(id,path,name) VALUES('original','/tmp/original.json','Original');
            CREATE TABLE server_tasks (task_id TEXT PRIMARY KEY, name TEXT NOT NULL, state TEXT NOT NULL,
              created_at REAL NOT NULL, started_at REAL, finished_at REAL, outputs TEXT, error TEXT);
            INSERT INTO server_tasks(task_id,name,state,created_at,outputs)
              VALUES('old-task','echo','succeeded',1,'{"text":"preserved"}');
        ''')
    db = DB(path).init_schema()
    assert db.get_workflow_path('original').as_posix() == '/tmp/original.json'
    assert db.get_task('old-task')['outputs'] == {'text': 'preserved'}
    assert db.get_task('old-task')['attempt'] == 1
    db.close()
    reopened = DB(path).init_schema()
    assert reopened.get_workflow_name('original') == 'Original'
    reopened.close()


def test_http_deploy_and_restart_queries_use_selected_database(db, tmp_path):
    from fastapi.testclient import TestClient
    from nndeploy.server import Server
    name = 'entry-' + uuid.uuid4().hex
    location = db.url.render_as_string(hide_password=False)
    with TestClient(Server(name=name, deploy=lambda inputs: inputs, data_dir=tmp_path / 'http', database_url=location).app) as client:
        first = client.post(f'/v1/server/{name}/deploy', json={'inputs': {'text': 'persisted'}}, headers={'Idempotency-Key': 'http'})
        assert first.status_code == 200, first.text
        again = client.post(f'/v1/server/{name}/deploy', json={'inputs': {'text': 'persisted'}}, headers={'Idempotency-Key': 'http'})
        assert first.json() == again.json()
        file = client.post('/v1/files', files={'file': ('test.txt', b'persisted file', 'text/plain')})
        assert file.status_code == 201
    with TestClient(Server(data_dir=tmp_path / 'http', database_url=location).app) as client:
        row = client.get('/v1/tasks/' + first.json()['task_id']).json()
        assert row['state'] == 'succeeded' and row['inputs'] == {'text': 'persisted'}
        assert client.get('/v1/files/' + file.json()['file_id'] + '/content').content == b'persisted file'


def test_reference_and_delete_race_never_loses_a_task_file(db):
    import threading
    identity = 'file_' + uuid.uuid4().hex
    task = 'task_' + uuid.uuid4().hex
    db.create_file(dict(file_id=identity, storage_key=identity, media_type='image/png', size=1, created_at=time.time()))
    barrier = threading.Barrier(2)
    def admit():
        barrier.wait(timeout=5)
        try:
            db.create_task(task, 'files', time.time(), {'image': {'file_id': identity}})
            return 'accepted'
        except ServerError:
            return 'rejected'
    def remove():
        barrier.wait(timeout=5)
        try:
            db.mark_file_deleting(identity)
            return 'deleting'
        except ServerError:
            return 'referenced'
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.submit(admit), pool.submit(remove)
        accepted, removed = first.result(), second.result()
    assert (accepted, removed) in (('accepted', 'referenced'), ('rejected', 'deleting'))
    if accepted == 'accepted':
        assert db.get_file(identity)['state'] == 'ready'
        with pytest.raises(ServerError) as refusal:
            db.delete_file(identity)
        assert refusal.value.code == 'file_in_use'


def _migrate_in_process(location):
    database = DB(location).init_schema()
    try:
        return database.database_identity()
    finally:
        database.close()


def test_multiple_processes_share_one_migration_and_database_identity(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn')) as pool:
        identities = list(pool.map(_migrate_in_process, [str(tmp_path / 'concurrent.db')] * 3))
    assert len(set(identities)) == 1


def test_existing_sqlite_is_backed_up_before_schema_upgrade(tmp_path):
    import stat
    path = tmp_path / 'original.db'
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE original_data (id TEXT PRIMARY KEY, value TEXT)')
        connection.execute("INSERT INTO original_data VALUES('preserved', 'before-upgrade')")
    database = DB(path).init_schema()
    database.close()
    backups = list(tmp_path.glob('original.db.backup-*.sqlite3'))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as connection:
        assert connection.execute('SELECT value FROM original_data').fetchone()[0] == 'before-upgrade'
        assert not connection.execute("SELECT name FROM sqlite_master WHERE name='server_tasks'").fetchone()
    if os.name != 'nt':
        assert stat.S_IMODE(backups[0].stat().st_mode) == 0o600
    DB(path).init_schema().close()
    assert list(tmp_path.glob('original.db.backup-*.sqlite3')) == backups
