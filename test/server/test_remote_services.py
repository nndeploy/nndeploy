"""Opt-in remote checks; never run destructive local-service fault tests remotely.

Enable NNDEPLOY_TEST_REMOTE_POSTGRES / _REDIS / _COS individually with value 1.
Connections use DATABASE_URL, REDIS_URL and COS_* environment variables. All
writes use a generated PostgreSQL schema, Redis namespace or COS object prefix.
No credentials or signed URLs should be printed by the test runner.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import os
import time
import uuid

import pytest

from nndeploy.server.db import DB
from nndeploy.server.errors import ServerError
from nndeploy.server.storage import LocalStorage
from nndeploy.server.storage.objects import StorageConfig


def enabled(service):
    if os.environ.get('NNDEPLOY_TEST_REMOTE_' + service) != '1':
        pytest.skip('Opt-in remote ' + service + ' validation is disabled')


@pytest.fixture
def remote_database(tmp_path):
    enabled('POSTGRES')
    import psycopg
    from psycopg import sql
    from sqlalchemy.engine import make_url
    schema = 'nndeploy_probe_' + uuid.uuid4().hex
    admin = psycopg.connect(os.environ['DATABASE_URL'], connect_timeout=5, autocommit=True)
    database = None
    created = False
    try:
        admin.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
        created = True
        original = make_url(os.environ['DATABASE_URL'])
        options = original.query.get('options', '') + ' -csearch_path=' + schema
        url = original.update_query_dict({'options': options})
        database = DB(url.render_as_string(hide_password=False), timeout_seconds=5, backup_dir=tmp_path / 'backup')
        with database.engine.connect() as connection:
            assert connection.exec_driver_sql('SELECT current_schema()').scalar_one() == schema
        database.init_schema()
        yield database
    finally:
        if database is not None:
            database.close()
        try:
            if created:
                admin.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
        finally:
            admin.close()


def test_remote_postgres_transactions_idempotency_and_durable_events(remote_database):
    database = remote_database
    name = 'remote-probe'
    def submit(index):
        return database.create_task('task_' + uuid.uuid4().hex, name, time.time(),
                                    {'text': '中文'}, {}, idempotency_key='same')
    with ThreadPoolExecutor(max_workers=4) as pool:
        identities = list(pool.map(submit, range(8)))
    assert len(set(identities)) == 1
    identity = identities[0]
    with pytest.raises(ServerError) as error:
        database.create_task('task_' + uuid.uuid4().hex, name, time.time(),
                             {'text': 'changed'}, {}, idempotency_key='same')
    assert error.value.code == 'idempotency_conflict'
    database.start_task(identity, time.time())
    database.append_task_event(identity, {'text': 'delta'})
    database.finish_task(identity, 'succeeded', time.time(), {'text': 'done'})
    database.engine.dispose()
    assert database.get_task(identity)['outputs'] == {'text': 'done'}
    assert database.read_task_events(identity)[-1]['data']['state'] == 'succeeded'


@pytest.fixture
def remote_redis(tmp_path):
    enabled('REDIS')
    import redis
    from nndeploy.server.redis import RedisConfig
    # Never adopt the application's existing namespace or notification keys.
    namespace = os.environ.get('REDIS_PREFIX', 'nndeploy') + ':probe-' + uuid.uuid4().hex[:16]
    config = RedisConfig(url_env='REDIS_URL', namespace=namespace,
        event_stream_prefix=os.environ.get('REDIS_STREAM_PREFIX'), socket_timeout_seconds=3)
    client = redis.Redis.from_url(os.environ['REDIS_URL'], decode_responses=True,
                                  socket_timeout=3, socket_connect_timeout=3)
    database = None
    try:
        assert client.ping()
        database = DB(tmp_path / 'redis-metadata.db').init_schema()
        yield database, config
    finally:
        if database is not None:
            database.close()
            patterns = [namespace + ':*']
            if config.event_stream_prefix:
                patterns.append(config.event_stream_prefix + namespace + ':*')
            for pattern in patterns:
                for key in client.scan_iter(match=pattern, count=100):
                    client.delete(key)
        client.close()


def test_remote_redis_queue_and_database_backed_event_replay(remote_redis):
    from nndeploy.server.redis import RedisEvents
    from nndeploy.server.task_queue import RedisTaskQueue
    database, config = remote_redis
    async def scenario():
        queues = [RedisTaskQueue(database, config.model_dump(), 'probe', 'same', 2) for _ in range(2)]
        events = RedisEvents(config, database.database_identity())
        try:
            for queue in queues:
                await queue.start()
            accepted = await asyncio.gather(*(queue.put({'text': 'once'}, idempotency_key='same') for queue in queues))
            assert accepted[0]['task_id'] == accepted[1]['task_id']
            record = await queues[1].get('remote-probe-worker')
            assert record['task_id'] == accepted[0]['task_id']
            await queues[1].finish(record, outputs={'text': 'done'})
            await queues[1].finish(record, outputs={'text': 'done'})
            assert database.queue_usage(queues[0].key)['inflight'] == 0
            await asyncio.to_thread(events.notify, record['task_id'], 2)
            assert await events.wait(record['task_id'], 1)
            await events.reader.delete(events.prefix + record['task_id'])
            assert database.get_task(record['task_id'])['outputs'] == {'text': 'done'}
        finally:
            await events.close()
            for queue in queues:
                await queue.close()
    asyncio.run(scenario())


@pytest.fixture
def remote_cos(tmp_path):
    enabled('COS')
    prefix = 'nndeploy-validation/' + uuid.uuid4().hex
    config = StorageConfig(backend='cos', bucket=os.environ.get('COS_BUCKET_NAME') or os.environ['COS_BUCKET'],
        region=os.environ['COS_REGION'], prefix=prefix,
        access_key_env='COS_ACCESS_KEY_ID', secret_key_env='COS_ACCESS_KEY_SECRET')
    database = DB(tmp_path / 'cos-metadata.db').init_schema()
    storage = LocalStorage(tmp_path / 'cache-a', database, config=config)
    backend = storage.backend
    try:
        yield storage, config
    finally:
        try:
            # The random prefix belongs to this fixture only. Also clear any
            # failed multipart upload if an assertion or transport error occurs.
            uploads = backend.client.list_multipart_uploads(Bucket=config.bucket, Prefix=prefix + '/').get('Upload', [])
            for upload in uploads:
                assert upload['Key'].startswith(prefix + '/')
                backend.client.abort_multipart_upload(Bucket=config.bucket, Key=upload['Key'], UploadId=upload['UploadId'])
            objects = backend.client.list_objects(Bucket=config.bucket, Prefix=prefix + '/').get('Contents', [])
            for record in objects:
                assert record['Key'].startswith(prefix + '/')
                backend.client.delete_object(Bucket=config.bucket, Key=record['Key'])
            assert not backend.client.list_objects(Bucket=config.bucket, Prefix=prefix + '/').get('Contents')
            assert not backend.client.list_multipart_uploads(Bucket=config.bucket, Prefix=prefix + '/').get('Upload')
        finally:
            storage.close()
            database.close()


@pytest.mark.parametrize('size', [37, 9 * 1024**2])
def test_remote_cos_roundtrip_signed_url_and_cold_cache(remote_cos, tmp_path, size):
    import requests
    storage, config = remote_cos
    chunk = b'nndeploy-check-'
    payload = (chunk * (size // len(chunk) + 1))[:size]
    record = storage.save(io.BytesIO(payload), 'application/octet-stream', owner='probe')
    cold = LocalStorage(tmp_path / 'cache-b', storage.db, config=config)
    try:
        with cold.open(record['file_id']) as stream:
            assert stream.read() == payload
        stored = storage.db.get_file(record['file_id'])
        url = storage.backend.signed_get(stored['storage_key'], 60)
        with requests.get(url, timeout=30) as response:
            assert response.status_code == 200
            assert response.content == payload
        assert not storage.backend.client.list_multipart_uploads(Bucket=config.bucket,
            Prefix=config.prefix + '/').get('Upload')
        storage.remove(record['file_id'])
        assert storage.db.get_file(record['file_id']) is None
        with pytest.raises(Exception) as error:
            storage.backend.client.head_object(Bucket=config.bucket, Key=storage.backend.key(stored['storage_key']))
        assert error.value.get_status_code() == 404
    finally:
        cold.close()


def test_remote_cos_failed_part_is_aborted(remote_cos, monkeypatch):
    storage, config = remote_cos
    def fail(**kwargs):
        raise RuntimeError('Injected upload failure')
    monkeypatch.setattr(storage.backend.client, 'upload_part', fail)
    with pytest.raises(RuntimeError, match='Injected upload failure'):
        storage.save(io.BytesIO(b'x' * (9 * 1024**2)), 'application/octet-stream')
    assert not storage.backend.client.list_multipart_uploads(Bucket=config.bucket,
        Prefix=config.prefix + '/').get('Upload')
    assert not storage.backend.client.list_objects(Bucket=config.bucket,
        Prefix=config.prefix + '/').get('Contents')


def test_remote_cos_metadata_failure_rolls_back_object(remote_cos, monkeypatch):
    storage, config = remote_cos
    def fail(*args, **kwargs):
        raise RuntimeError('Injected metadata failure')
    monkeypatch.setattr(storage.db, 'create_file', fail)
    with pytest.raises(RuntimeError, match='Injected metadata failure'):
        storage.save(io.BytesIO(b'remote-rollback-check'), 'application/octet-stream')
    assert not storage.backend.client.list_objects(Bucket=config.bucket,
        Prefix=config.prefix + '/').get('Contents')


def test_remote_cos_detects_corrupt_content_before_caching(remote_cos):
    storage, config = remote_cos
    record = storage.save(io.BytesIO(b'original-data'), 'application/octet-stream')
    path, stored = storage.resolve(record['file_id'])
    path.unlink()
    storage.backend.client.put_object(Bucket=config.bucket, Key=storage.backend.key(stored['storage_key']),
                                      Body=b'modified-data')
    with pytest.raises(ServerError) as error:
        storage.resolve(record['file_id'])
    assert error.value.code == 'content_mismatch'
    assert not path.exists()
    assert not list(storage.root.glob('.download-*'))
    storage.remove(record['file_id'])


def test_remote_cos_server_graph_outputs_survive_restart(remote_cos, tmp_path):
    from fastapi.testclient import TestClient
    from nndeploy.server import Server
    from test_server_api import image_graph
    from test_storage import picture
    storage, config = remote_cos
    options = dict(workflow=image_graph(), name='image', admin_token='remote-probe-test-token',
        database_url=storage.db.url.render_as_string(hide_password=False), storage_config=config.model_dump())
    first = Server(resources=tmp_path / 'first', data_dir=tmp_path / 'first', **options)
    with TestClient(first.app) as client:
        upload = client.post('/v1/files', files={'file': ('input.png', picture(), 'image/png')})
        assert upload.status_code == 201
        result = client.post('/v1/server/image/deploy', json={'inputs': {'decode': {'file_id': upload.json()['file_id']}}})
        assert result.status_code == 200, result.text
        identity = result.json()['task_id']
        output = result.json()['outputs']['encode']['file_id']
        content = client.get('/v1/files/' + output + '/content')
        assert content.status_code == 200
        expected = content.content
        assert client.delete('/v1/files/' + output,
            headers={'Authorization': 'Bearer remote-probe-test-token'}).status_code == 409
    second = Server(resources=tmp_path / 'second', data_dir=tmp_path / 'second', **options)
    with TestClient(second.app) as client:
        assert client.get('/v1/tasks/' + identity).json()['state'] == 'succeeded'
        content = client.get('/v1/files/' + output + '/content')
        assert content.status_code == 200
        assert content.content == expected


def test_remote_postgres_redis_cos_two_server_graph_roundtrip(
        remote_database, remote_redis, remote_cos, tmp_path):
    import httpx
    from PIL import Image
    from nndeploy.server import Server
    from test_server_api import image_graph
    from test_storage import picture
    _, redis_config = remote_redis
    _, storage_config = remote_cos
    options = dict(name='remote-image', workflow=image_graph(),
        database_url=remote_database.url.render_as_string(hide_password=False),
        redis_config=redis_config.model_dump(), storage_config=storage_config.model_dump())
    servers = [Server(resources=tmp_path / name, data_dir=tmp_path / name, **options) for name in ('replica-a', 'replica-b')]

    async def scenario():
        try:
            for server in servers:
                await server.start()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=servers[0].app), base_url='http://test') as first, \
                    httpx.AsyncClient(transport=httpx.ASGITransport(app=servers[1].app), base_url='http://test') as second:
                uploaded = await first.post('/v1/files', files={'file': ('input.png', picture(), 'image/png')})
                assert uploaded.status_code == 201
                body = {'inputs': {'decode': {'file_id': uploaded.json()['file_id']}}}
                headers = {'Idempotency-Key': 'remote-once'}
                response = await first.post('/v1/server/remote-image/deploy', json=body, headers=headers)
                assert response.status_code == 200, response.text
                task = response.json()
                repeated = await second.post('/v1/server/remote-image/deploy', json=body, headers=headers)
                assert repeated.status_code == 200
                assert repeated.json()['task_id'] == task['task_id']
                identity = task['outputs']['encode']['file_id']
                stored = remote_database.get_file(identity)
                (servers[1].storage.root / stored['storage_key']).unlink(missing_ok=True)
                content = await second.get('/v1/files/' + identity + '/content')
                assert content.status_code == 200
                with Image.open(io.BytesIO(content.content)) as result:
                    assert result.size == (7, 9)
                    assert result.getpixel((0, 0)) == (24, 60, 91)
                events = remote_database.read_task_events(task['task_id'])
                assert events[-1]['data']['state'] == 'succeeded'
                for server in servers:
                    assert remote_database.queue_usage(server.entries['remote-image']['queue'].key)['inflight'] == 0
        finally:
            results = await asyncio.gather(*(server.close() for server in servers), return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    raise result
    asyncio.run(scenario())
