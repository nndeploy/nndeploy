"""Task quotas use durable admission, independent of API replica or queue."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import uuid

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, func, text, update

from nndeploy.server import Server
from nndeploy.server.db import DB, tasks
from nndeploy.server.errors import ServerError
from test_auth import credentials
from test_db import db
from test_redis_queue import redis_service, backend


def test_concurrent_admission_across_connections_and_entries_never_exceeds_identity_limit(db):
    owner = 'subject-' + uuid.uuid4().hex
    databases = [DB(db.url) for _ in range(3)]
    barrier = threading.Barrier(12)

    def create(i):
        barrier.wait(timeout=10)
        try:
            return databases[i % 3].create_task(uuid.uuid4().hex, 'entry-' + str(i), time.time(),
                                               owner=owner, max_tasks_per_subject=3)
        except ServerError as exc:
            assert exc.code == 'subject_tasks_full'
            return None

    try:
        with ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(create, range(12)))
        admitted = [identity for identity in results if identity]
        assert len(admitted) == 3
        # Cancellation intent, API restart and a different queue do not free
        # capacity while the original execution remains unconfirmed.
        db.request_cancel(admitted[0])
        restarted = DB(db.url)
        try:
            with pytest.raises(ServerError) as full:
                restarted.create_task(uuid.uuid4().hex, 'new-entry', time.time(), owner=owner, max_tasks_per_subject=3)
            assert full.value.code == 'subject_tasks_full'
            db.finish_task(admitted[0], 'cancelled', time.time())
            restarted.create_task(uuid.uuid4().hex, 'new-entry', time.time(), owner=owner, max_tasks_per_subject=3)
        finally:
            restarted.close()
        # Case-sensitive subjects must stay distinct even on MySQL.
        db.create_task(uuid.uuid4().hex, 'other', time.time(), owner=owner.upper(), max_tasks_per_subject=1)
    finally:
        for database in databases:
            database.close()


def test_quota_idempotency_and_failed_admission_rollback(db):
    owner, name = uuid.uuid4().hex, uuid.uuid4().hex
    with pytest.raises(ServerError) as missing:
        db.create_task(uuid.uuid4().hex, name, time.time(), {'file_id': 'file_missing'}, {},
                       owner=owner, max_tasks_per_subject=1)
    assert missing.value.code == 'invalid_file'

    def create(i):
        return db.create_task(uuid.uuid4().hex, name, time.time(), {}, {}, owner=owner,
                              idempotency_key='same', max_tasks_per_subject=1)
    with ThreadPoolExecutor(max_workers=8) as pool:
        identities = list(pool.map(create, range(16)))
    assert len(set(identities)) == 1
    with pytest.raises(ServerError) as conflict:
        db.create_task(uuid.uuid4().hex, name, time.time(), {'changed': True}, {},
                       owner=owner, idempotency_key='same', max_tasks_per_subject=1)
    assert conflict.value.status_code == 409
    db.finish_task(identities[0], 'failed', time.time())
    db.create_task(uuid.uuid4().hex, name, time.time(), {}, {}, owner=owner, max_tasks_per_subject=1)


def test_http_running_cancel_retains_capacity_and_terminal_releases_it(db, tmp_path, credentials):
    config, headers = credentials
    config['max_inflight_tasks_per_subject'] = 1
    # The SQL fixture is shared by other integration tests; isolate subjects.
    for key in config['keys']:
        key['subject'] += '-' + uuid.uuid4().hex
        key['entries'].append('other')
    entered, release = threading.Event(), threading.Event()

    def deploy(values):
        entered.set()
        assert release.wait(timeout=15)
        return values

    server = Server(name='echo', deploy=deploy, auth_config=config, data_dir=tmp_path,
                    database_url=db.url.render_as_string(hide_password=False))
    server.add(name='other', deploy=lambda x: x)
    with TestClient(server.app) as client:
        try:
            first = client.post('/v1/server/echo/deploy', json={'inputs': {}, 'mode': 'async'},
                                headers=dict(headers['alice'], **{'Idempotency-Key': 'original'}))
            assert first.status_code == 202, first.text
            identity = first.json()['task_id']
            assert entered.wait(timeout=5)
            replay = client.post('/v1/server/echo/deploy', json={'inputs': {}, 'mode': 'async'},
                                 headers=dict(headers['alice'], **{'Idempotency-Key': 'original'}))
            assert replay.json()['task_id'] == identity
            assert client.get('/v1/tasks/' + identity, headers=headers['alice']).status_code == 200
            cancelled = client.post('/v1/tasks/' + identity + '/cancel', headers=headers['alice'])
            assert cancelled.status_code == 200 and cancelled.json()['state'] == 'running'
            rejected = client.post('/v1/server/other/deploy', json={'inputs': {}}, headers=headers['alice'])
            assert rejected.status_code == 429, rejected.text
            assert rejected.json()['error']['code'] == 'subject_tasks_full'
            assert client.post('/v1/server/other/deploy', json={'inputs': {}}, headers=headers['bob']).status_code == 200
            release.set()
            deadline = time.monotonic() + 5
            while db.get_task(identity)['state'] not in ('succeeded', 'cancelled'):
                assert time.monotonic() < deadline
                time.sleep(.02)
            assert client.post('/v1/server/other/deploy', json={'inputs': {}}, headers=headers['alice']).status_code == 200
        finally:
            release.set()


def test_shared_http_replicas_use_one_identity_task_budget(backend, tmp_path, credentials):
    from test_shared_server import shared_deploy, entry_config, until

    database, redis = backend
    auth, headers = credentials
    auth['max_inflight_tasks_per_subject'] = 1
    for key in auth['keys']:
        key['subject'] += '-' + uuid.uuid4().hex
    servers = [Server(name='echo', deploy=shared_deploy, config=entry_config(),
                      data_dir=tmp_path / 'shared', database_url=database.url.render_as_string(hide_password=False),
                      redis_config=redis, auth_config=auth) for _ in range(2)]
    marker, release = tmp_path / 'started', tmp_path / 'release'

    async def scenario():
        try:
            for server in servers:
                await server.start()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=servers[0].app), base_url='http://test') as a, \
                       httpx.AsyncClient(transport=httpx.ASGITransport(app=servers[1].app), base_url='http://test') as b:
                payload = {'inputs': {'marker': str(marker), 'release': str(release)}, 'mode': 'async'}
                request_headers = dict(headers['alice'], **{'Idempotency-Key': 'shared'})
                response = await a.post('/v1/server/echo/deploy', json=payload, headers=request_headers)
                assert response.status_code == 202, response.text
                identity = response.json()['task_id']
                await until(marker.exists)
                replay = await b.post('/v1/server/echo/deploy', json=payload, headers=request_headers)
                assert replay.json()['task_id'] == identity
                denied = await b.post('/v1/server/echo/deploy', json={'inputs': {}, 'mode': 'async'}, headers=headers['alice'])
                assert denied.status_code == 429, denied.text
                assert denied.json()['error']['code'] == 'subject_tasks_full'
                release.write_text('released')
                await until(lambda: database.get_task(identity)['state'] == 'succeeded')
                assert (await b.post('/v1/server/echo/deploy', json={'inputs': {}}, headers=headers['alice'])).status_code == 200
                with database.engine.connect() as connection:
                    assert connection.execute(select(func.count()).select_from(tasks).where(
                        tasks.c.queue_key == database.get_task(identity)['queue_key'])).scalar_one() == 2
        finally:
            release.write_text('released')
            await asyncio.gather(*(server.close() for server in servers))
    asyncio.run(scenario())


def test_migration_counts_old_unfinished_tasks_and_is_resumable(db):
    owner = 'old-' + uuid.uuid4().hex
    identity = db.create_task(uuid.uuid4().hex, uuid.uuid4().hex, time.time(), owner=owner)
    with db.engine.begin() as connection:
        connection.execute(update(tasks).where(tasks.c.task_id == identity).values(owner_key=None))
        connection.execute(text("UPDATE alembic_version SET version_num = '0025_editor_tasks'"))
    upgraded = DB(db.url).init_schema()
    try:
        assert upgraded.get_task(identity)['owner_key'] == DB.subject_key(owner)
        with pytest.raises(ServerError) as full:
            upgraded.create_task(uuid.uuid4().hex, 'after-upgrade', time.time(), owner=owner, max_tasks_per_subject=1)
        assert full.value.code == 'subject_tasks_full'
        upgraded.finish_task(identity, 'expired', time.time())
        upgraded.create_task(uuid.uuid4().hex, 'after-upgrade', time.time(), owner=owner, max_tasks_per_subject=1)
    finally:
        upgraded.close()
