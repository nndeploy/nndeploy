"""Two Server instances, real Redis and independent model worker processes."""
import asyncio
import os
from pathlib import Path
import time

import httpx
import pytest

from nndeploy.server import Server
from nndeploy.server.errors import ServerError
from test_redis_queue import redis_service, backend
from test_auth import credentials
from test_streaming import output_stream, stream_config, parse_events
from test_engine import build_engine, engine_config
from test_dynamic_batching import VectorBatch, BatchOutput, batch_config


def test_shared_stopped_workflow_does_not_warm_until_explicitly_resumed(backend, tmp_path):
    import uuid
    from sqlalchemy import select, func
    from nndeploy.server.db import tasks
    from test_server_api import text_graph
    from test_warmup import text_config
    db, redis_config = backend
    name = 'warmup-' + uuid.uuid4().hex
    marker = tmp_path / 'executions.txt'
    workflow = text_graph()
    workflow['node_repository_'][1]['frontend_params'].update(save_to_file=True, file_path=str(marker))
    options = dict(resources=tmp_path / 'server', data_dir=tmp_path / 'server', database_url=db.url.render_as_string(hide_password=False), redis_config=redis_config)
    server = Server(name=name, workflow=workflow, config=text_config(), **options)
    async def scenario():
        await server.start()
        try:
            assert marker.read_text().splitlines() == ['warm']
            with db.engine.connect() as connection:
                assert connection.execute(select(func.count()).select_from(tasks).where(tasks.c.name == name)).scalar_one() == 0
            await server.stop_entry(name)
        finally:
            await server.close()
        restored = Server(**options)
        await restored.start()
        try:
            assert restored.describe(name)['status'] == 'stopped'
            assert marker.read_text().splitlines() == ['warm']
            await restored.start_entry(name)
            assert marker.read_text().splitlines() == ['warm', 'warm']
            assert (await restored.deploy(name, {}))['outputs']['text'] == 'default'
        finally:
            await restored.close()
    asyncio.run(scenario())


def test_shared_graph_outputs_are_owned_before_result_commit_and_survive_until_retention(backend, tmp_path):
    import io
    from PIL import Image
    from test_server_api import image_graph
    from test_storage import picture
    from test_file_cleanup import expire as expire_file
    from test_retention import expire as expire_task
    db, redis_config = backend
    server = Server(resources=tmp_path, data_dir=tmp_path, database_url=db.url.render_as_string(hide_password=False), redis_config=redis_config)
    file = server.storage.save(io.BytesIO(picture()), 'image/png')['file_id']
    server.add(name='images', workflow=image_graph(), config={
        'inputs': {'image': {'type': 'image', 'node': ['decode']}},
        'outputs': {'image': {'type': 'image', 'node': ['encode']}}})
    async def scenario():
        await server.start()
        try:
            result = await server.deploy('images', {'image': {'file_id': file}})
            output = result['outputs']['image']['file_id']
            assert db.get_file(output)['producer_task_id'] == result['task_id']
            assert db.get_task(result['task_id'])['worker_token'] is not None
            with server.storage.open(output) as content:
                with Image.open(content) as image:
                    assert image.size == (7, 9) and image.getpixel((0, 0)) == (24, 60, 91)
            expire_file(db, output)
            assert (await server.cleanup_files())['deleted'] == 0
            expire_task(db, result['task_id'])
            await server.cleanup_tasks()
            assert (await server.cleanup_files())['deleted'] == 1
        finally:
            await server.close()
    asyncio.run(scenario())


def test_shared_retention_survives_replay_and_another_server_can_clean(backend, tmp_path):
    from test_retention import expire
    db, redis_config = backend
    config = entry_config()
    config['execution']['task_retention_seconds'] = 60.0
    server = Server(name='retained', deploy=shared_deploy, config=config, resources=tmp_path / 'worker', data_dir=tmp_path / 'worker',
                    database_url=db.url.render_as_string(hide_password=False), redis_config=redis_config)
    observer = Server(resources=tmp_path / 'observer', data_dir=tmp_path / 'observer', database_url=db.url.render_as_string(hide_password=False))
    async def request():
        _, future = await server.submit('retained', {'text': 'once'}, idempotency_key='same')
        return await asyncio.shield(future)
    async def scenario():
        await server.start()
        await observer.start()
        try:
            first = await request()
            identity = first['task_id']
            row = db.get_task(identity)
            assert row['retention_seconds'] == 60 and row['purge_after'] > db.clock()
            again = await request()
            assert again == first
            assert await observer.cleanup_tasks() == 0
            expire(db, identity)
            assert await observer.cleanup_tasks() == 1
            assert db.get_task(identity) is None and db.read_task_events(identity) == []
            assert db.queue_usage(server.entries['retained']['queue'].key)['inflight'] == 0
            repeated = await request()
            assert repeated['task_id'] != identity
            # A reader reconnecting after retention must not wait indefinitely.
            _, future = server._record_future(dict(task_id=identity, state='pending', queue_key='shared'))
            with pytest.raises(ServerError) as missing:
                await asyncio.wait_for(asyncio.shield(future), 5)
            assert missing.value.code == 'task_not_found'
        finally:
            await observer.close()
            await server.close()
    asyncio.run(scenario())


def test_shared_dynamic_batch_and_expiry_keep_independent_durable_tasks(backend, tmp_path):
    from test_engine import until
    db, redis_config = backend
    server = Server(name='vectors', handler_factory=VectorBatch, output_model=BatchOutput, config=batch_config(size=3, wait=200),
                    resources=tmp_path / 'batch', data_dir=tmp_path / 'batch', database_url=db.url.render_as_string(hide_password=False), redis_config=redis_config)
    server.add(name='expired', handler_factory=VectorBatch, config=batch_config(size=4, wait=150, timeout=.04))
    async def scenario():
        await server.start()
        try:
            items = [{'inputs': {'values': [index]}, 'idempotency_key': f'batch-{index}'} for index in range(3)]
            result = await server.deploy_batch('vectors', items)
            assert [item['outputs']['total'] for item in result['items']] == [0, 1, 2]
            assert all(item['outputs']['batch_size'] == 3 for item in result['items'])
            worker = server.entries['vectors']['workers'][0]
            assert len(worker.executor.handler.calls) == 1
            replay = await server.deploy_batch('vectors', items)
            assert [item['task_id'] for item in result['items']] == [item['task_id'] for item in replay['items']]
            assert len(worker.executor.handler.calls) == 1
            with pytest.raises(ServerError) as failure:
                await server.deploy('expired', {'values': [9]})
            assert failure.value.code == 'queue_timeout'
            assert db.get_task(failure.value.task_id)['state'] == 'expired'
            assert not server.entries['expired']['workers'][0].executor.handler.calls
            await until(lambda: db.queue_usage(server.entries['expired']['queue'].key)['inflight'] == 0)
        finally:
            await server.close()
    asyncio.run(scenario())


def test_remote_batch_cancel_waits_for_actual_compute_and_keeps_other_items(backend, tmp_path):
    from test_engine import until
    db, redis_config = backend
    server = Server(name='vectors', handler_factory=VectorBatch, config=batch_config(size=2, queue_size=0),
                    resources=tmp_path / 'worker', data_dir=tmp_path / 'worker', database_url=db.url.render_as_string(hide_password=False), redis_config=redis_config)
    observer = Server(resources=tmp_path / 'observer', data_dir=tmp_path / 'observer', database_url=db.url.render_as_string(hide_password=False))
    release, marker = tmp_path / 'release', tmp_path / 'marker'
    async def scenario():
        await server.start()
        await observer.start()
        try:
            accepted = await asyncio.gather(*(server.submit('vectors', {'values': [i], 'release': str(release), 'marker': str(marker)}) for i in (1, 2)))
            await until(marker.exists)
            await observer.cancel_task(accepted[0][0])
            worker = server.entries['vectors']['workers'][0]
            await until(lambda: worker.jobs[accepted[0][0]]['cancelled'])
            assert db.queue_usage(server.entries['vectors']['queue'].key)['inflight'] == 2
            with pytest.raises(ServerError) as full:
                await server.deploy('vectors', {'values': [3]})
            assert full.value.status_code == 429
            release.touch()
            result = await asyncio.gather(*(future for _, future in accepted), return_exceptions=True)
            assert isinstance(result[0], ServerError) and result[0].code == 'task_cancelled'
            assert result[1]['outputs']['total'] == 2
            assert (await server.deploy('vectors', {'values': [3]}))['outputs']['total'] == 3
        finally:
            release.touch()
            await observer.close()
            await server.close()
    asyncio.run(scenario())


def shared_deploy(values):
    marker = values['marker']
    if marker:
        Path(marker).write_text(str(os.getpid()))
    release = values['release']
    if release:
        deadline = time.monotonic() + 15
        while not Path(release).exists():
            if time.monotonic() >= deadline:
                raise RuntimeError('Test release was not signalled')
            time.sleep(.02)
    return {'pid': os.getpid(), 'text': values['text']}


def entry_config():
    return {'inputs': {
        'text': {'type': 'string', 'required': False, 'default': 'echo'},
        'marker': {'type': 'string', 'required': False, 'default': ''},
        'release': {'type': 'string', 'required': False, 'default': ''}},
        'execution': {'isolation': 'process', 'max_queue_size': 3}}


def test_shared_identity_and_idempotency_survive_cross_instance_queries(backend, tmp_path, credentials):
    db, redis_config = backend
    auth_config, headers = credentials
    servers = [Server(name='echo', deploy=shared_deploy, config=entry_config(), resources=tmp_path / 'shared', data_dir=tmp_path / 'shared',
                      database_url=db.url.render_as_string(hide_password=False), redis_config=redis_config,
                      auth_config=auth_config) for _ in range(2)]
    async def scenario():
        try:
            for server in servers:
                await server.start()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=servers[0].app), base_url='http://test') as first, \
                       httpx.AsyncClient(transport=httpx.ASGITransport(app=servers[1].app), base_url='http://test') as second:
                ids = {}
                for person in ('alice', 'bob'):
                    request_headers = dict(headers[person], **{'Idempotency-Key': 'shared-auth'})
                    result = await first.post('/v1/server/echo/deploy', json={'inputs': {'text': person}}, headers=request_headers)
                    assert result.status_code == 200, result.text
                    ids[person] = result.json()['task_id']
                    repeated = await second.post('/v1/server/echo/deploy', json={'inputs': {'text': person}}, headers=request_headers)
                    assert repeated.json()['task_id'] == ids[person]
                    assert db.get_task(ids[person])['owner'] == person
                assert ids['alice'] != ids['bob']
                assert (await second.get('/v1/tasks/' + ids['alice'], headers=headers['alice'])).status_code == 200
                assert (await second.get('/v1/tasks/' + ids['alice'], headers=headers['bob'])).status_code == 404
                assert (await second.post('/v1/tasks/' + ids['alice'] + '/cancel', headers=headers['bob'])).status_code == 404
        finally:
            await asyncio.gather(*(server.close() for server in servers))
    asyncio.run(scenario())


async def until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(.02)
    raise AssertionError('Condition did not become true')


def test_shared_engine_capacity_and_remote_request_scoped_cancellation(backend, tmp_path):
    db, redis_config = backend
    servers = [Server(name='engine', handler_factory=build_engine, config=engine_config(cooperative=True),
                      resources=tmp_path / 'shared', data_dir=tmp_path / 'shared', database_url=db.url.render_as_string(hide_password=False),
                      redis_config=redis_config) for _ in range(2)]
    async def scenario():
        release = tmp_path / 'release'
        marker = tmp_path / 'marker'
        try:
            for server in servers:
                await server.start()
            identity, future = await servers[0].submit('engine', {'release': str(release), 'marker': str(marker)})
            await until(marker.exists)
            owner = next(index for index, server in enumerate(servers) if identity in server.running_workers)
            other = servers[1 - owner]
            assert (await other.deploy('engine', {'text': 'parallel'}))['outputs']['text'] == 'parallel'
            assert db.get_task(identity)['state'] == 'running'
            await other.cancel_task(identity)
            with pytest.raises(ServerError) as stopped:
                await asyncio.wait_for(asyncio.shield(future), 5)
            assert stopped.value.code == 'task_cancelled'
            assert (await servers[owner].deploy('engine', {'text': 'after-cancel'}))['outputs']['text'] == 'after-cancel'
            usage = db.queue_usage(servers[0].entries['engine']['queue'].key)
            assert usage['capacity'] == 2 and usage['inflight'] == 0
        finally:
            release.touch()
            await asyncio.gather(*(server.close() for server in servers))
    asyncio.run(scenario())


def test_shared_batch_partial_admission_and_per_item_retries(backend, tmp_path):
    db, redis_config = backend
    servers = [Server(name='engine', handler_factory=build_engine, config=engine_config(),
                      resources=tmp_path / 'shared', data_dir=tmp_path / 'shared', database_url=db.url.render_as_string(hide_password=False),
                      redis_config=redis_config) for _ in range(2)]
    async def scenario():
        release = tmp_path / 'release'
        items = [{'inputs': {'release': str(release), 'text': str(index)}, 'idempotency_key': str(index)} for index in range(4)]
        try:
            for server in servers:
                await server.start()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=servers[0].app), base_url='http://test') as first:
                accepted = await first.post('/v1/server/engine/deploy/batch', json={'items': items, 'mode': 'async'})
                assert accepted.status_code == 202, accepted.text
                original = accepted.json()['items']
                assert [item['status_code'] for item in original] == [202, 202, 429, 429]
            release.touch()
            await until(lambda: all(db.get_task(item['task_id'])['state'] == 'succeeded' for item in original[:2]))
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=servers[1].app), base_url='http://test') as second:
                response = await second.post('/v1/server/engine/deploy/batch', json={'items': items})
                assert response.status_code == 200, response.text
                retried = response.json()['items']
                assert [item['status_code'] for item in retried] == [200] * 4
                assert [item['task_id'] for item in retried[:2]] == [item['task_id'] for item in original[:2]]
                assert [item['outputs']['text'] for item in retried] == ['0', '1', '2', '3']
        finally:
            release.touch()
            await asyncio.gather(*(server.close() for server in servers))
    asyncio.run(scenario())


def test_shared_stream_can_replay_from_another_instance_after_writer_closes(backend, tmp_path):
    db, redis_config = backend
    servers = [Server(name='echo', deploy=output_stream, config=stream_config('process'), resources=tmp_path / 'shared', data_dir=tmp_path / 'shared',
                      database_url=db.url.render_as_string(hide_password=False), redis_config=redis_config) for _ in range(2)]
    async def scenario():
        try:
            for server in servers:
                await server.start()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=servers[0].app), base_url='http://test') as first:
                response = await first.post('/v1/server/echo/deploy', json={'mode': 'stream'}, headers={'Idempotency-Key': 'stream'})
                assert response.status_code == 200, response.text
                identity = response.headers['x-task-id']
                assert [item['sequence'] for item in parse_events(response.text)] == [1, 2, 3, 4, 5]
            await servers[0].close()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=servers[1].app), base_url='http://test') as second:
                replay = await second.get(f'/v1/tasks/{identity}/events', headers={'Last-Event-ID': '2'})
                assert [item['sequence'] for item in parse_events(replay.text)] == [3, 4, 5]
                repeated = await second.post('/v1/server/echo/deploy', json={'mode': 'stream'}, headers={'Idempotency-Key': 'stream'})
                assert repeated.headers['x-task-id'] == identity
                assert [item['sequence'] for item in parse_events(repeated.text)] == [1, 2, 3, 4, 5]
        finally:
            await asyncio.gather(*(server.close() for server in servers))
    asyncio.run(scenario())


def test_http_request_crosses_to_another_server_worker(backend, tmp_path):
    db, redis_config = backend
    location = db.url.render_as_string(hide_password=False)
    servers = [Server(name='shared', deploy=shared_deploy, config=entry_config(), resources=tmp_path / 'shared', data_dir=tmp_path / 'shared',
                      database_url=location, redis_config=redis_config) for _ in range(2)]
    async def scenario():
        release = tmp_path / 'release'
        try:
            for server in servers:
                await server.start()
            pids = [server.entries['shared']['workers'][0].pid for server in servers]
            assert len(set(pids)) == 2 and os.getpid() not in pids
            marker = tmp_path / 'entered'
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=servers[0].app), base_url='http://test') as client:
                accepted = await client.post('/v1/server/shared/deploy', json={'mode': 'async',
                    'inputs': {'marker': str(marker), 'release': str(release)}})
                assert accepted.status_code == 202, accepted.text
                await until(marker.exists)
                busy = pids.index(int(marker.read_text()))
                # Submit to the instance whose model process is currently busy.
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=servers[busy].app), base_url='http://test') as other:
                    result = await other.post('/v1/server/shared/deploy', json={'inputs': {'text': 'remote'}})
                    assert result.status_code == 200, result.text
                    assert result.json()['outputs'] == {'pid': pids[1 - busy], 'text': 'remote'}
                release.touch()
                await until(lambda: db.get_task(accepted.json()['task_id'])['state'] == 'succeeded')
        finally:
            release.touch()
            await asyncio.gather(*(server.close() for server in servers))
    asyncio.run(scenario())


def test_shared_stop_start_and_instance_shutdown_have_explicit_scope(backend, tmp_path):
    db, redis_config = backend
    location = db.url.render_as_string(hide_password=False)
    servers = [Server(name='shared', deploy=shared_deploy, config=entry_config(), resources=tmp_path / 'shared', data_dir=tmp_path / 'shared',
                      database_url=location, redis_config=redis_config) for _ in range(2)]
    async def scenario():
        try:
            for server in servers:
                await server.start()
            await servers[0].stop_entry('shared')
            await until(lambda: servers[1].entries['shared']['status'] == 'stopped')
            with pytest.raises(ServerError) as stopped:
                await servers[1].deploy('shared', {})
            assert stopped.value.status_code == 503
            await servers[0].start_entry('shared')
            await until(lambda: servers[1].entries['shared']['status'] == 'ready')
            assert (await servers[1].deploy('shared', {'text': 'restarted'}))['outputs']['text'] == 'restarted'
            await servers[0].close()
            assert (await servers[1].deploy('shared', {'text': 'remaining'}))['outputs']['text'] == 'remaining'
        finally:
            await asyncio.gather(*(server.close() for server in servers))
    asyncio.run(scenario())


def test_http_accepts_durable_pending_task_during_redis_outage(backend, redis_service, tmp_path):
    db, redis_config = backend
    server = Server(name='shared', deploy=shared_deploy, config=entry_config(), resources=tmp_path / 'shared', data_dir=tmp_path / 'shared',
                    database_url=db.url.render_as_string(hide_password=False), redis_config=redis_config)
    async def scenario():
        try:
            await server.start()
            await asyncio.to_thread(redis_service.stop)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                assert (await client.get('/health/ready')).status_code == 503
                accepted = await client.post('/v1/server/shared/deploy', json={'mode': 'async', 'inputs': {'text': 'durable'}})
                assert accepted.status_code == 202 and accepted.json()['dispatch_pending'], accepted.text
                await asyncio.to_thread(redis_service.start)
                await until(lambda: db.get_task(accepted.json()['task_id'])['state'] == 'succeeded')
                result = await client.get(accepted.json()['status_url'])
                assert result.json()['outputs']['text'] == 'durable'
                assert not {'worker_token', 'queue_key', 'group_key', 'fingerprint', 'attempt'}.intersection(result.json())
        finally:
            if redis_service.process.poll() is not None:
                await asyncio.to_thread(redis_service.start)
            await server.close()
    asyncio.run(scenario())


def test_completed_result_recovers_after_db_failure_and_server_restart(backend, tmp_path, monkeypatch):
    from sqlalchemy.exc import OperationalError
    db, redis_config = backend
    directory = tmp_path / 'shared'
    location = db.url.render_as_string(hide_password=False)
    first = Server(name='shared', deploy=shared_deploy, config=entry_config(), resources=directory, data_dir=directory,
                   database_url=location, redis_config=redis_config)
    second = Server(name='shared', deploy=shared_deploy, config=entry_config(), resources=directory, data_dir=directory,
                    database_url=location, redis_config=redis_config)
    def unavailable(*args, **kwargs):
        raise OperationalError('UPDATE task', {}, RuntimeError('injected database outage'))
    async def scenario():
        try:
            await first.start()
            monkeypatch.setattr(first.db, 'finish_task', unavailable)
            marker = tmp_path / 'completed-once'
            identity, future = await first.submit('shared', {'marker': str(marker), 'text': 'once'})
            await until(lambda: bool(list((directory / 'queue-results').rglob('*.json'))))
            pid = int(marker.read_text())
            assert not future.done()
            assert db.get_task(identity)['state'] == 'running'
            key = db.get_task(identity)['queue_key']
            assert db.queue_usage(key)['inflight'] == 1
            await first.close()
            assert db.get_task(identity)['state'] == 'running'
            await second.start()
            await until(lambda: db.get_task(identity)['state'] == 'succeeded')
            result = db.get_task(identity)
            assert result['outputs'] == {'pid': pid, 'text': 'once'}
            assert second.entries['shared']['workers'][0].pid != pid
            assert int(marker.read_text()) == pid
            assert db.queue_usage(key)['inflight'] == 0
            await until(lambda: not list((directory / 'queue-results').rglob('*.json')))
        finally:
            await first.close()
            await second.close()
    asyncio.run(scenario())


def test_shared_workers_reject_inaccessible_local_file_storage(backend, tmp_path):
    db, redis_config = backend
    location = db.url.render_as_string(hide_password=False)
    first = Server(name='shared', deploy=shared_deploy, config=entry_config(), resources=tmp_path / 'one', data_dir=tmp_path / 'one',
                   database_url=location, redis_config=redis_config)
    other = Server(name='shared', deploy=shared_deploy, config=entry_config(), resources=tmp_path / 'two', data_dir=tmp_path / 'two',
                   database_url=location, redis_config=redis_config)
    async def scenario():
        try:
            await first.start()
            with pytest.raises(ServerError) as rejected:
                await other.start()
            assert rejected.value.code == 'shared_storage_mismatch'
            assert (await first.deploy('shared', {'text': 'still-ready'}))['outputs']['text'] == 'still-ready'
        finally:
            await first.close()
            await other.close()
    asyncio.run(scenario())
