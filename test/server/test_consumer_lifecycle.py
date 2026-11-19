"""Internal consumer cancellation must not abandon computation or teardown."""
import asyncio
from pathlib import Path
import time
import threading
import uuid

import pytest

from nndeploy.server import Server
from nndeploy.server.errors import ServerError
from test_engine import until
from test_db import db
from test_redis_queue import backend, redis_service


def held(inputs):
    root = Path(inputs['directory'])
    with (root / 'entered').open('a') as output:
        output.write('run\n')
    deadline = time.monotonic() + 10
    while not (root / 'release').exists():
        if time.monotonic() >= deadline:
            raise RuntimeError('Test execution was not released')
        time.sleep(.01)
    (root / 'exited').touch()
    return {'value': 'completed'}


async def held_async(inputs):
    root = Path(inputs['directory'])
    with (root / 'entered').open('a') as output:
        output.write('run\n')
    while not (root / 'release').exists():
        await asyncio.sleep(.01)
    (root / 'exited').touch()
    return {'value': 'completed'}


def settings(mode):
    execution = {'isolation': mode} if mode in ('thread', 'process') else {
        'scheduling': mode, 'max_concurrency_per_worker': 2}
    return {'inputs': {'directory': {'type': 'string'}}, 'execution': execution}


@pytest.mark.parametrize('mode', ['thread', 'process', 'concurrent', 'engine'])
def test_consumer_cancel_keeps_execution_owned_until_real_completion(db, tmp_path, mode):
    root = tmp_path / 'execution'
    root.mkdir()
    server = Server(name='echo', deploy=held_async if mode == 'engine' else held, config=settings(mode),
                    resources=tmp_path / 'server', data_dir=tmp_path / 'server', database_url=db.url.render_as_string(hide_password=False))

    async def scenario():
        closing = None
        await server.start()
        workers = server.entries['echo']['workers'][:]
        try:
            identity, result = await server.submit('echo', {'directory': str(root)})
            await until(lambda: (root / 'entered').exists())
            for consumer in server.entries['echo']['consumers']:
                consumer.cancel()
            await asyncio.sleep(.05)
            assert not result.done(), 'Cancelling a consumer must not invent a successful empty result'
            assert server.inflight == 1
            assert identity in server.running_workers
            assert server.db.get_task(identity)['state'] == 'running'
            closing = asyncio.create_task(server.close())
            await asyncio.sleep(.05)
            assert not closing.done(), 'Closing must join the actual calculation'
            assert not (root / 'exited').exists()
            (root / 'release').touch()
            assert (await asyncio.wait_for(result, 5))['outputs'] == {'value': 'completed'}
            await asyncio.wait_for(closing, 5)
            assert server.entries['echo']['workers'] == []
            assert all(worker.closed for worker in workers)
            assert server.inflight == 0
            assert db.get_task(identity)['outputs'] == {'value': 'completed'}
            assert (root / 'entered').read_text() == 'run\n'
        finally:
            (root / 'release').touch()
            await asyncio.gather(server.close(), return_exceptions=True)
            # Keep the failing baseline test from leaking its model thread/process.
            await asyncio.gather(*(worker.close() for worker in workers), return_exceptions=True)
            if closing is not None:
                await asyncio.gather(closing, return_exceptions=True)
    asyncio.run(scenario())


@pytest.mark.parametrize('mode', ['thread', 'process', 'engine'])
@pytest.mark.parametrize('stage', ['claim', 'execute', 'result'])
def test_shared_consumer_cancel_preserves_claim_execution_and_result(backend, tmp_path, monkeypatch, mode, stage):
    database, redis_config = backend
    root = tmp_path / 'execution'
    root.mkdir()
    server = Server(name='echo', deploy=held_async if mode == 'engine' else held, config=settings(mode),
                    resources=tmp_path / 'server', data_dir=tmp_path / 'server', database_url=database.url.render_as_string(hide_password=False),
                    redis_config=redis_config)
    blocked, release = threading.Event(), threading.Event()
    method = 'claim_task' if stage == 'claim' else 'finish_task'
    original = getattr(server.db, method)

    def held_receipt(*args, **kwargs):
        value = original(*args, **kwargs)
        blocked.set()
        assert release.wait(10), 'Test transaction receipt was not released'
        return value

    if stage != 'execute':
        monkeypatch.setattr(server.db, method, held_receipt)
        (root / 'release').touch()

    async def scenario():
        closing = None
        await server.start()
        workers = server.entries['echo']['workers'][:]
        try:
            identity, observer = await server.submit('echo', {'directory': str(root)})
            observer.cancel()  # Query the durable record independently below.
            if stage == 'execute':
                await until(lambda: (root / 'entered').exists())
            else:
                assert await asyncio.to_thread(blocked.wait, 5)
            for consumer in server.entries['echo']['consumers']:
                consumer.cancel()
            closing = asyncio.create_task(server.close())
            await asyncio.sleep(.05)
            assert not closing.done(), 'Shutdown must join an in-flight claim, compute or result write'
            if stage != 'result':
                assert database.get_task(identity)['state'] == 'running'
                usage = database.queue_usage(server.entries['echo']['queue'].key)
                assert usage['inflight'] == 1
            release.set()
            (root / 'release').touch()
            await asyncio.wait_for(closing, 8)
            record = database.get_task(identity)
            assert record['state'] == 'succeeded', record
            assert record['outputs'] == {'value': 'completed'}
            assert (root / 'entered').read_text() == 'run\n'
            assert all(worker.closed for worker in workers)
            assert database.queue_usage(server.entries['echo']['queue'].key)['inflight'] == 0
        finally:
            release.set()
            (root / 'release').touch()
            await asyncio.gather(server.close(), return_exceptions=True)
            await asyncio.gather(*(worker.close() for worker in workers), return_exceptions=True)
            if closing is not None:
                await asyncio.gather(closing, return_exceptions=True)
    asyncio.run(scenario())


def test_consumer_exception_still_closes_every_worker_and_dependency(tmp_path, monkeypatch):
    released = []
    class Handler:
        def deploy(self, values):
            return values
        def deinit(self):
            released.append('worker')
    server = Server(name='echo', handler_factory=Handler, resources=tmp_path, data_dir=tmp_path)
    server.add(name='other', handler_factory=Handler)
    original = server._consume
    async def faulty(entry, worker):
        if entry['config'].name == 'echo':
            raise RuntimeError('Controlled consumer failure')
        return await original(entry, worker)
    monkeypatch.setattr(server, '_consume', faulty)
    close = server.db.close
    def close_db():
        released.append('database')
        close()
    monkeypatch.setattr(server.db, 'close', close_db)

    async def scenario():
        await server.start()
        with pytest.raises(ServerError) as failure:
            await server.close()
        assert failure.value.code == 'consumer_failed'
        assert released.count('worker') == 2
        assert released[-1] == 'database'
        assert all(not entry['workers'] for entry in server.entries.values())
    asyncio.run(scenario())


@pytest.mark.parametrize('failure', ['raise', 'return'])
def test_abandoned_local_requests_fail_after_worker_release_and_other_entry_stays_available(db, tmp_path, monkeypatch, failure):
    gate = asyncio.Event()
    released = []
    class Handler:
        def deploy(self, inputs):
            return inputs
        def deinit(self):
            released.append(True)
    server = Server(name='broken', handler_factory=Handler, resources=tmp_path, data_dir=tmp_path,
                    database_url=db.url.render_as_string(hide_password=False))
    server.add(name='healthy', deploy=lambda values: values)
    original = server._consume
    async def consumer(entry, worker):
        if entry['config'].name == 'broken':
            await gate.wait()
            if failure == 'raise':
                raise RuntimeError('Controlled consumer exit')
            return
        return await original(entry, worker)
    monkeypatch.setattr(server, '_consume', consumer)

    async def scenario():
        await server.start()
        identity, observer = await server.submit('broken', {'value': 'must not run'})
        gate.set()
        await until(lambda: not server.accepting('broken'))
        assert server.describe('broken')['status'] == 'failed'
        with pytest.raises(ServerError):
            await server.submit('broken', {})
        assert (await server.deploy('healthy', {'value': 'available'}))['outputs'] == {'value': 'available'}
        assert not observer.done()
        assert db.get_task(identity)['state'] == 'pending'
        with pytest.raises(ServerError, match='consumer exited'):
            await server.close()
        assert released == [True]
        assert db.get_task(identity)['state'] == 'failed'
        with pytest.raises(ServerError) as result:
            await observer
        assert result.value.code == 'consumer_failed'
        assert server.inflight == 0
    asyncio.run(scenario())


def test_failed_shared_consumer_only_finalizes_its_own_runtime_after_exit(backend, tmp_path, monkeypatch):
    database, redis_config = backend
    servers = [Server(name='echo', deploy=held, config=settings('process'), resources=tmp_path / 'shared', data_dir=tmp_path / 'shared',
                      database_url=database.url.render_as_string(hide_password=False), redis_config=redis_config) for _ in range(2)]
    failed_root, healthy_root = tmp_path / 'failed', tmp_path / 'healthy'
    failed_root.mkdir()
    healthy_root.mkdir()

    async def faulty(entry, worker):
        while True:
            record = await entry['queue'].get(uuid.uuid4().hex, holder=worker.shared_holder)
            if record is not None:
                raise RuntimeError('Consumer exited after durable claim')
    monkeypatch.setattr(servers[0], '_consume_shared', faulty)

    async def scenario():
        try:
            await servers[0].start()
            unknown, observer = await servers[0].submit('echo', {'directory': str(failed_root)})
            observer.cancel()
            await until(lambda: not servers[0].accepting('echo'))
            assert database.get_task(unknown)['state'] == 'running'
            await servers[1].start()
            healthy, result = await servers[1].submit('echo', {'directory': str(healthy_root)})
            await until(lambda: (healthy_root / 'entered').exists())
            with pytest.raises(ServerError) as failure:
                await servers[0].close()
            assert failure.value.code == 'consumer_failed'
            assert database.get_task(unknown)['state'] == 'failed'
            assert not (failed_root / 'entered').exists()
            assert database.get_task(healthy)['state'] == 'running'
            assert not result.done()
            assert database.queue_usage(servers[1].entries['echo']['queue'].key)['inflight'] == 1
            (healthy_root / 'release').touch()
            assert (await asyncio.wait_for(result, 5))['outputs'] == {'value': 'completed'}
        finally:
            (healthy_root / 'release').touch()
            (failed_root / 'release').touch()
            await asyncio.gather(*(server.close() for server in servers), return_exceptions=True)
    asyncio.run(scenario())


def test_idle_consumer_cancel_does_not_skip_worker_deinit(tmp_path):
    released = []
    class Handler:
        def deploy(self, inputs):
            return inputs
        def deinit(self):
            released.append(True)
    server = Server(handler_factory=Handler, resources=tmp_path, data_dir=tmp_path)
    async def scenario():
        await server.start()
        workers = server.entries['default']['workers'][:]
        try:
            server.entries['default']['consumers'][0].cancel()
            await server.close()
            assert released == [True]
        finally:
            await asyncio.gather(*(worker.close() for worker in workers), return_exceptions=True)
    asyncio.run(scenario())


def test_explicit_cancel_racing_failed_consumer_stop_releases_capacity_once(tmp_path, monkeypatch):
    committed, release = threading.Event(), threading.Event()
    gate = asyncio.Event()
    server = Server(name='echo', deploy=lambda values: values, resources=tmp_path, data_dir=tmp_path)
    async def consumer(entry, worker):
        await gate.wait()
        raise RuntimeError('Controlled consumer exit')
    monkeypatch.setattr(server, '_consume', consumer)
    original = server.db.finish_task
    def finish(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[1] == 'cancelled':
            committed.set()
            assert release.wait(5)
        return result
    monkeypatch.setattr(server.db, 'finish_task', finish)

    async def scenario():
        cancelling = None
        await server.start()
        try:
            identity, observer = await server.submit('echo', {})
            gate.set()
            await until(lambda: not server.accepting('echo'))
            cancelling = asyncio.create_task(server.cancel_task(identity))
            assert await asyncio.to_thread(committed.wait, 3)
            with pytest.raises(ServerError) as failure:
                await server.stop_entry('echo')
            assert failure.value.code == 'consumer_failed'
            assert server.inflight == 0
            release.set()
            assert (await cancelling)['state'] == 'cancelled'
            assert server.inflight == 0
            with pytest.raises(ServerError) as result:
                await observer
            assert result.value.code == 'task_cancelled'
        finally:
            release.set()
            if cancelling is not None:
                await asyncio.gather(cancelling, return_exceptions=True)
            await asyncio.gather(server.close(), return_exceptions=True)
    asyncio.run(scenario())


def abort_or_return_sync(inputs):
    if inputs.get('abort'):
        raise asyncio.CancelledError('Algorithm ended its own request')
    return inputs


async def abort_or_return(inputs):
    return abort_or_return_sync(inputs)


def abort_config(mode):
    config = {'inputs': {'abort': {'type': 'boolean'}}}
    config['execution'] = settings(mode)['execution']
    return config


def check_algorithm_cancel(server):
    async def scenario():
        await server.start()
        try:
            identity, observer = await server.submit('echo', {'abort': True})
            with pytest.raises(ServerError) as failure:
                await asyncio.wait_for(observer, 3)
            assert failure.value.code == 'task_cancelled'
            assert server.db.get_task(identity)['state'] == 'cancelled'
            assert server.accepting('echo')
            assert (await server.deploy('echo', {'abort': False}))['outputs'] == {'abort': False}
        finally:
            await asyncio.gather(server.close(), return_exceptions=True)
    asyncio.run(scenario())


@pytest.mark.parametrize('mode', ['thread', 'process', 'concurrent', 'engine'])
def test_algorithm_cancelled_error_is_not_reported_as_success(db, tmp_path, mode):
    server = Server(name='echo', deploy=abort_or_return_sync if mode == 'concurrent' else abort_or_return,
                    config=abort_config(mode), resources=tmp_path, data_dir=tmp_path,
                    database_url=db.url.render_as_string(hide_password=False))
    check_algorithm_cancel(server)


@pytest.mark.parametrize('mode', ['thread', 'process', 'concurrent', 'engine'])
def test_shared_algorithm_cancelled_error_keeps_worker_available(backend, tmp_path, mode):
    database, redis_config = backend
    server = Server(name='echo', deploy=abort_or_return_sync if mode == 'concurrent' else abort_or_return,
                    config=abort_config(mode), resources=tmp_path, data_dir=tmp_path,
                    database_url=database.url.render_as_string(hide_password=False), redis_config=redis_config)
    check_algorithm_cancel(server)


def test_unexpected_consumer_abort_never_settles_an_empty_success(db, tmp_path, monkeypatch):
    class ConsumerAbort(BaseException):
        pass
    server = Server(name='echo', deploy=lambda values: values, resources=tmp_path, data_dir=tmp_path,
                    database_url=db.url.render_as_string(hide_password=False))
    async def abort(*args, **kwargs):
        raise ConsumerAbort('Unexpected execution boundary failure')
    async def scenario():
        await server.start()
        worker = server.entries['echo']['workers'][0]
        monkeypatch.setattr(worker, 'deploy', abort)
        try:
            identity, observer = await server.submit('echo', {})
            await until(lambda: not server.accepting('echo'))
            assert not observer.done(), 'An unknown execution outcome must not become an empty success'
            assert server.inflight == 1
            assert db.get_task(identity)['state'] == 'running'
            with pytest.raises(ServerError) as failure:
                await server.stop_entry('echo')
            assert failure.value.code == 'consumer_failed'
            assert worker.closed
            assert server.inflight == 0
            assert db.get_task(identity)['state'] == 'failed'
            with pytest.raises(ServerError) as failure:
                await observer
            assert failure.value.code == 'consumer_failed'
        finally:
            await asyncio.gather(server.close(), return_exceptions=True)
    asyncio.run(scenario())


@pytest.mark.parametrize('stage', ['key', 'batch'])
def test_batch_algorithm_cancel_keeps_scheduler_and_worker_available(db, tmp_path, stage):
    from nndeploy.server.worker import Worker
    class Handler:
        def batch_key(self, inputs):
            if stage == 'key':
                abort_or_return_sync(inputs)
            return 'same'
        def deploy_batch(self, items):
            return [abort_or_return_sync(item['inputs']) for item in items]
    config = {'inputs': {'abort': {'type': 'boolean'}},
              'execution': {'batching': {'enabled': True, 'max_size': 2, 'max_wait_ms': 10}}}
    server = Server(name='echo', handler_factory=Handler, config=config, resources=tmp_path, data_dir=tmp_path,
                    database_url=db.url.render_as_string(hide_password=False))
    async def scenario():
        await server.start()
        worker = server.entries['echo']['workers'][0]
        try:
            identity, observer = await server.submit('echo', {'abort': True})
            with pytest.raises(ServerError) as failure:
                await asyncio.wait_for(observer, 2)
            assert failure.value.code == 'task_cancelled'
            assert db.get_task(identity)['state'] == 'cancelled'
            assert server.accepting('echo') and not worker.scheduler.done()
            assert (await server.deploy('echo', {'abort': False}))['outputs'] == {'abort': False}
        finally:
            # A failing baseline can strand a batch job. Clean it only after
            # the assertions, so pytest itself does not leak the compute pool.
            for job in list(worker.jobs.values()):
                worker._finish(job, failure=ServerError(500, 'test_cleanup', 'Test cleanup'))
            await asyncio.gather(server.close(), return_exceptions=True)
            await Worker.close(worker)
    asyncio.run(scenario())


def test_shared_stop_with_failed_consumer_does_not_wait_on_its_abandoned_task(backend, tmp_path, monkeypatch):
    database, redis_config = backend
    server = Server(name='echo', deploy=held_async, config=settings('engine'), resources=tmp_path, data_dir=tmp_path,
                    database_url=database.url.render_as_string(hide_password=False), redis_config=redis_config)
    consume = server._consume_shared
    selected = False
    claimed = asyncio.Event()
    async def consume_or_fail(entry, worker):
        nonlocal selected
        if selected:
            await claimed.wait()
            return await consume(entry, worker)
        selected = True
        while True:
            record = await entry['queue'].get(uuid.uuid4().hex, holder=worker.shared_holder)
            if record is not None:
                claimed.set()
                raise RuntimeError('Consumer exited after durable claim')
    monkeypatch.setattr(server, '_consume_shared', consume_or_fail)
    async def scenario():
        await server.start()
        entry = server.entries['echo']
        try:
            # Keep the healthy consumer out of the claim until the failing
            # consumer has accepted this request; it still joins the real Stop.
            healthy = entry['consumer_runs'][1]
            identity, observer = await server.submit('echo', {'directory': str(tmp_path)})
            observer.cancel()
            await until(lambda: not server.accepting('echo'))
            assert database.get_task(identity)['state'] == 'running'
            assert not healthy.done()
            with pytest.raises(ServerError) as failure:
                await asyncio.wait_for(server.stop_entry('echo'), 2)
            assert failure.value.code == 'consumer_failed'
            assert database.get_task(identity)['state'] == 'failed'
            assert database.queue_usage(entry['queue'].key)['inflight'] == 0
        finally:
            entry['local_stop'] = True
            await asyncio.gather(server.close(), return_exceptions=True)
    asyncio.run(scenario())
