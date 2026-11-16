"""Shared queue fault tests using a real, disposable Redis process."""
import asyncio
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import uuid

import pytest
from sqlalchemy import update

from nndeploy.server.db import DB, tasks
from nndeploy.server.errors import ServerError
from nndeploy.server.task_queue import RedisTaskQueue

URLS = ['sqlite'] + json.loads(os.environ.get('NNDEPLOY_TEST_DATABASE_URLS', '[]'))


@pytest.fixture(scope='module')
def redis_service(tmp_path_factory):
    if os.environ.get('NNDEPLOY_TEST_REDIS') != '1':
        pytest.skip('Set NNDEPLOY_TEST_REDIS=1 for real Redis integration tests')
    import redis
    executable = shutil.which('redis-server')
    if not executable:
        pytest.fail('redis-server is required for requested integration tests')
    directory = tmp_path_factory.mktemp('redis')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    class Service:
        url = f'redis://127.0.0.1:{port}/0'
        process = None
        def start(self):
            with (directory / 'redis.log').open('a') as log:
                self.process = subprocess.Popen([executable, '--bind', '127.0.0.1', '--port', str(port),
                    '--save', '', '--appendonly', 'no', '--dir', str(directory)], stdout=log, stderr=log)
            client = redis.Redis.from_url(self.url, socket_timeout=.2)
            try:
                for _ in range(100):
                    if self.process.poll() is not None:
                        pytest.fail('Redis exited: ' + (directory / 'redis.log').read_text())
                    try:
                        if client.ping():
                            return
                    except redis.RedisError:
                        pass
                    time.sleep(.05)
                pytest.fail('Redis startup timed out')
            finally:
                client.close()
        def stop(self):
            if self.process and self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=10)
    service = Service()
    service.start()
    yield service
    service.stop()


@pytest.fixture(params=URLS, ids=lambda url: url.split(':')[0])
def backend(request, redis_service, tmp_path, monkeypatch):
    db = DB(tmp_path / 'queue.db' if request.param == 'sqlite' else request.param).init_schema()
    monkeypatch.setenv('NNDEPLOY_REDIS_TEST_URL', redis_service.url)
    config = dict(url_env='NNDEPLOY_REDIS_TEST_URL', namespace='test-' + uuid.uuid4().hex,
                  lease_seconds=1, socket_timeout_seconds=.2, max_inflight_tasks=4)
    yield db, config
    db.close()


def test_shared_workers_route_by_entry_and_configuration(backend):
    db, config = backend
    async def scenario():
        a = RedisTaskQueue(db, config, 'a', 'same-config', 2)
        a2 = RedisTaskQueue(db, config, 'a', 'same-config', 2)
        b = RedisTaskQueue(db, config, 'b', 'same-config', 2)
        changed = RedisTaskQueue(db, config, 'a', 'different-config', 2)
        queues = [a, a2, b, changed]
        try:
            for queue in queues:
                await queue.start()
            first = await a.put({'model': 'a'})
            second = await b.put({'model': 'b'})
            # Even a misrouted Redis message cannot make B execute an A task.
            await b.client.xadd(b.stream, {'task_id': first['task_id']})
            received_b = await b.get('worker-b')
            assert received_b['task_id'] == second['task_id']
            assert await b.get('worker-b') is None
            assert await changed.get('wrong-config-worker') is None
            received_a = await a2.get('worker-a')
            assert received_a['task_id'] == first['task_id']
            await a2.finish(received_a, outputs={'ok': 'a'})
            await b.finish(received_b, outputs={'ok': 'b'})
            assert db.queue_usage(a.group_key)['inflight'] == 0
            assert await a.client.xlen(a.stream) == 0
        finally:
            for queue in queues:
                await queue.close()
    asyncio.run(scenario())


def test_shared_capacity_and_idempotency_survive_concurrent_submitters(backend):
    db, config = backend
    async def scenario():
        a = RedisTaskQueue(db, config, 'capacity', 'config', 2)
        b = RedisTaskQueue(db, config, 'capacity', 'config', 2)
        try:
            await a.start()
            await b.start()
            results = await asyncio.gather(*(queue.put({'index': index}) for index, queue in enumerate([a, b, a, b])), return_exceptions=True)
            accepted = [row for row in results if isinstance(row, dict)]
            assert len(accepted) == 2
            assert all(isinstance(row, ServerError) and row.status_code == 429 for row in results if not isinstance(row, dict))
            for worker in ('worker-1', 'worker-2'):
                record = await a.get(worker)
                await a.finish(record, outputs={})
            duplicates = await asyncio.gather(*(queue.put({'same': True}, idempotency_key='one') for queue in [a, b, a, b]))
            assert len({row['task_id'] for row in duplicates}) == 1
            assert db.queue_usage(a.key)['inflight'] == 1
            with pytest.raises(ServerError) as conflict:
                await b.put({'same': False}, idempotency_key='one')
            assert conflict.value.code == 'idempotency_conflict'
            record = await b.get('worker-final')
            await b.finish(record, outputs={})
            await b.finish(record, outputs={})  # Lost ACK retry is harmless.
            assert db.queue_usage(a.key)['inflight'] == 0
        finally:
            await a.close()
            await b.close()
    asyncio.run(scenario())


def test_redis_outage_retains_db_outbox_and_recovers(backend, redis_service):
    db, config = backend
    async def scenario():
        queue = RedisTaskQueue(db, config, 'outbox', 'config', 2)
        try:
            await queue.start()
            await asyncio.to_thread(redis_service.stop)
            accepted = await queue.put({'text': 'recover'})
            assert accepted['dispatch_pending']
            assert db.get_task(accepted['task_id'])['state'] == 'pending'
            assert db.queue_usage(queue.key)['inflight'] == 1
            await asyncio.to_thread(redis_service.start)
            await queue.start()
            assert (await queue.reconcile())['published'] == 1
            record = await queue.get('recovered-worker')
            assert record['inputs'] == {'text': 'recover'}
            await queue.finish(record, outputs={'ok': True})
            assert db.queue_usage(queue.key)['inflight'] == 0
        finally:
            if redis_service.process.poll() is not None:
                await asyncio.to_thread(redis_service.start)
            await queue.close()
    asyncio.run(scenario())


def test_lost_ack_redelivery_does_not_execute_completed_task(backend, redis_service):
    from redis.exceptions import RedisError
    db, config = backend
    async def scenario():
        queue = RedisTaskQueue(db, config, 'ack', 'config', 2)
        try:
            await queue.start()
            accepted = await queue.put({'value': 1})
            record = await queue.get('original-worker')
            await asyncio.to_thread(redis_service.stop)
            with pytest.raises(RedisError):
                await queue.finish(record, outputs={'value': 2})
            assert db.get_task(accepted['task_id'])['state'] == 'succeeded'
            assert db.queue_usage(queue.key)['inflight'] == 0
            await asyncio.to_thread(redis_service.start)
            await queue.start()
            await queue.client.xadd(queue.stream, {'task_id': accepted['task_id']})
            assert await queue.get('another-worker') is None
            assert db.get_task(accepted['task_id'])['outputs'] == {'value': 2}
        finally:
            if redis_service.process.poll() is not None:
                await asyncio.to_thread(redis_service.start)
            await queue.close()
    asyncio.run(scenario())


def test_expired_lease_never_releases_running_capacity_or_reexecutes(backend):
    db, config = backend
    async def scenario():
        queue = RedisTaskQueue(db, config, 'lease', 'config', 1)
        second = RedisTaskQueue(db, config, 'lease', 'config', 1)
        try:
            await queue.start()
            await second.start()
            accepted = await queue.put({'block': True})
            record = await queue.get('live-worker')
            with db.engine.begin() as connection:
                connection.execute(update(tasks).where(tasks.c.task_id == record['task_id']).values(lease_expires_at=time.time() - 1))
            await queue.client.xclaim(queue.stream, queue.group, 'stale-owner', min_idle_time=0,
                                      message_ids=[record['_message_id']], idle=2000)
            assert await second.get('replacement-worker') is None
            assert (await second.reconcile())['stalled'] == [accepted['task_id']]
            with pytest.raises(ServerError) as full:
                await second.put({'another': True})
            assert full.value.status_code == 429
            assert db.queue_usage(queue.key)['inflight'] == 1
            assert await queue.heartbeat(record)
            await queue.finish(record, outputs={'finished': True})
            assert db.queue_usage(queue.key)['inflight'] == 0
        finally:
            await queue.close()
            await second.close()
    asyncio.run(scenario())


def test_cancel_expire_and_stale_worker_result(backend):
    db, config = backend
    async def scenario():
        queue = RedisTaskQueue(db, config, 'states', 'config', 2)
        try:
            await queue.start()
            cancelled = await queue.put({'cancel': True})
            db.request_cancel(cancelled['task_id'])
            assert await queue.get('worker') is None
            assert db.get_task(cancelled['task_id'])['state'] == 'cancelled'
            expired = await queue.put({'expire': True})
            with db.engine.begin() as connection:
                connection.execute(update(tasks).where(tasks.c.task_id == expired['task_id']).values(expires_at=time.time() - 1))
            assert await queue.get('worker') is None
            assert db.get_task(expired['task_id'])['state'] == 'expired'
            await queue.put({'ownership': True})
            record = await queue.get('old-worker')
            with db.engine.begin() as connection:
                connection.execute(update(tasks).where(tasks.c.task_id == record['task_id']).values(attempt=2, worker_token='new-worker'))
            with pytest.raises(RuntimeError):
                await queue.finish(record, outputs={'stale': True})
            assert not await queue.heartbeat(record)
            assert db.queue_usage(queue.key)['inflight'] == 1
            fresh = db.get_task(record['task_id'])
            fresh['_message_id'] = record['_message_id']
            await queue.finish(fresh, outputs={'fresh': True})
            assert db.get_task(record['task_id'])['outputs'] == {'fresh': True}
        finally:
            await queue.close()
    asyncio.run(scenario())


def test_namespace_rejects_different_database(backend, tmp_path):
    db, config = backend
    other = DB(tmp_path / 'other-database.db').init_schema()
    async def scenario():
        original = RedisTaskQueue(db, config, 'same', 'same', 1)
        invalid = RedisTaskQueue(other, config, 'same', 'same', 1)
        try:
            await original.start()
            with pytest.raises(ServerError) as failure:
                await invalid.start()
            assert failure.value.code == 'queue_database_mismatch'
        finally:
            await original.close()
            await invalid.close()
    try:
        asyncio.run(scenario())
    finally:
        other.close()


def test_default_and_explicit_equivalent_settings_match_the_same_queue():
    from nndeploy.server.config import ServerConfig
    from nndeploy.server.redis import entry_fingerprint
    source = {'identity': 'workflow-fingerprint', 'input_model': None, 'output_model': None}
    default = ServerConfig()
    explicit = ServerConfig.model_validate({'execution': {'queue_timeout_seconds': 30, 'startup_timeout_seconds': 60}})
    changed = ServerConfig.model_validate({'execution': {'queue_timeout_seconds': 31}})
    assert entry_fingerprint(source, default) == entry_fingerprint(source, explicit)
    assert entry_fingerprint(source, default) != entry_fingerprint(source, changed)


def test_shared_task_preserves_callback_and_checks_idempotency(backend):
    db, config = backend
    async def scenario():
        queue = RedisTaskQueue(db, config, 'callback', 'config', 2)
        await queue.start()
        try:
            options = dict(idempotency_key='callback', callback_url='http://127.0.0.1:9000/callback', callback_scope='workspace')
            accepted = await queue.put({'text': 'hello'}, **options)
            same = await queue.put({'text': 'hello'}, **options)
            assert same['task_id'] == accepted['task_id']
            with pytest.raises(ServerError) as error:
                await queue.put({'text': 'hello'}, **dict(options, callback_url='http://127.0.0.1:9001/callback'))
            assert error.value.code == 'idempotency_conflict'
            record = db.get_task(accepted['task_id'])
            assert record['callback_url'] == options['callback_url'] and record['callback_scope'] == 'workspace'
            received = await queue.get('callback-worker')
            await queue.finish(received, outputs={'ok': True})
            from nndeploy.server.callbacks import claim
            assert claim(db, 'workspace')['state'] == 'succeeded'
        finally:
            await queue.close()
    asyncio.run(scenario())
