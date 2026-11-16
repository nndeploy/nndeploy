"""Initialization and warmup share physical device concurrency with Deploy."""
import asyncio
import os
from pathlib import Path
import threading
import uuid

import pytest
from sqlalchemy import select, func

from nndeploy.server.db import tasks
from nndeploy.server import Server
from nndeploy.server.errors import ServerError
from test_db import db
from test_devices import make
from test_device_execution import limits, config, activity
from test_shared_server import shared_deploy, until
from test_dynamic_batching import VectorBatch, batch_config
from test_engine import ControlledEngine, engine_config
from test_concurrent import ThreadSafeHandler, concurrent_config
from test_redis_queue import backend, redis_service


def record_factory():
    Path(os.environ['NNDEPLOY_TEST_INITIALIZER']).write_text(str(os.getpid()))


def synchronous_factory():
    record_factory()
    return shared_deploy


def engine_factory():
    record_factory()
    return ControlledEngine()


def concurrent_factory():
    record_factory()
    return ThreadSafeHandler()


def batch_factory():
    record_factory()
    return VectorBatch()


@pytest.mark.parametrize('mode', ['thread', 'process', 'concurrent', 'engine', 'batch'])
def test_start_waits_before_factory_and_holds_slot_through_warmup(db, tmp_path, monkeypatch, mode):
    server = make(db, tmp_path, limits(devices=('cpu', 'other')))
    server.add(name='busy', deploy=shared_deploy, config=config())
    server.add(name='independent', deploy=shared_deploy, config=config(('other',)))
    factory_marker, busy_marker, warm_marker, next_marker = [tmp_path / name for name in ('factory', 'busy', 'warm', 'next')]
    busy_release, warm_release = tmp_path / 'release-busy', tmp_path / 'release-warm'
    monkeypatch.setenv('NNDEPLOY_TEST_INITIALIZER', str(factory_marker))
    factories = {'thread': synchronous_factory, 'process': synchronous_factory,
                 'engine': engine_factory, 'concurrent': concurrent_factory, 'batch': batch_factory}
    cfg = (engine_config() if mode == 'engine' else concurrent_config() if mode == 'concurrent'
           else batch_config() if mode == 'batch' else config(isolation=mode))
    warm = {'marker': str(warm_marker), 'release': str(warm_release)}
    if mode == 'batch':
        warm['values'] = [1]
    cfg['execution'].update(devices=['cpu'], startup_timeout_seconds=10.0, warmup=[{'inputs': warm}])
    async def scenario():
        starting = None
        try:
            await server.start()
            _, busy = await server.submit('busy', {'marker': str(busy_marker), 'release': str(busy_release)})
            await until(busy_marker.exists)
            server.add(name='new', handler_factory=factories[mode], config=cfg)
            starting = asyncio.create_task(server.start_entry('new'))
            await until(lambda: activity(server)['waiting'] == 1)
            assert not factory_marker.exists() and not warm_marker.exists()
            assert (await server.deploy('independent', {}))['outputs']['text'] == 'old'
            busy_release.touch()
            await busy
            await until(warm_marker.exists)
            assert factory_marker.exists() and server.describe('new')['status'] == 'starting'
            assert activity(server)['running'] == 1
            _, next_call = await server.submit('busy', {'marker': str(next_marker)})
            await until(lambda: activity(server)['waiting'] == 1)
            assert not next_marker.exists()
            warm_release.touch()
            await starting
            await next_call
            assert server.describe('new')['status'] == 'ready'
            with db.engine.connect() as connection:
                assert connection.execute(select(func.count()).select_from(tasks).where(tasks.c.name == 'new')).scalar_one() == 0
            await until(lambda: activity(server) == {'running': 0, 'waiting': 0})
        finally:
            busy_release.touch()
            warm_release.touch()
            if starting:
                await asyncio.gather(starting, return_exceptions=True)
            await server.close()
    asyncio.run(scenario())


def test_device_wait_exhausts_startup_deadline_without_calling_factory(db, tmp_path, monkeypatch):
    server = make(db, tmp_path, limits())
    server.add(name='busy', deploy=shared_deploy, config=config())
    marker, release, initialized = [tmp_path / name for name in ('busy', 'release', 'factory')]
    monkeypatch.setenv('NNDEPLOY_TEST_INITIALIZER', str(initialized))
    async def scenario():
        try:
            await server.start()
            _, busy = await server.submit('busy', {'marker': str(marker), 'release': str(release)})
            await until(marker.exists)
            server.add(name='late', handler_factory=synchronous_factory, config=config(startup_timeout_seconds=.2))
            with pytest.raises(ServerError) as failure:
                await asyncio.wait_for(server.start_entry('late'), 3)
            assert failure.value.code == 'startup_timeout' and not initialized.exists()
            assert not busy.done() and activity(server) == {'running': 1, 'waiting': 0}
            assert server.devices.snapshot()['cpu']['used']['workers'] == 1
            release.touch()
            await busy
            await server.start_entry('late')
            assert initialized.exists() and server.describe('late')['status'] == 'ready'
        finally:
            release.touch()
            await server.close()
    asyncio.run(scenario())


def test_reload_timeout_keeps_old_entry_ready_and_does_not_initialize_candidate(db, tmp_path, monkeypatch):
    server = make(db, tmp_path, limits())
    initialized, marker, release = [tmp_path / name for name in ('factory', 'busy', 'release')]
    monkeypatch.setenv('NNDEPLOY_TEST_INITIALIZER', str(initialized))
    server.add(name='model', handler_factory=synchronous_factory, config=config(startup_timeout_seconds=.2))
    async def scenario():
        try:
            await server.start()
            worker = server.entries['model']['workers'][0]
            initialized.unlink()
            _, busy = await server.submit('model', {'marker': str(marker), 'release': str(release)})
            await until(marker.exists)
            with pytest.raises(ServerError) as failure:
                await server.reload_entry('model')
            assert failure.value.code == 'startup_timeout' and not initialized.exists()
            assert server.describe('model')['status'] == 'ready' and server.entries['model']['workers'][0] is worker
            assert server.devices.snapshot()['cpu']['used']['workers'] == 1
            release.touch()
            await busy
            assert (await server.deploy('model', {}))['outputs']['text'] == 'old'
        finally:
            release.touch()
            await server.close()
    asyncio.run(scenario())


def test_startup_grant_ack_loss_does_not_duplicate_factory_and_database_failure_never_loads(db, tmp_path, monkeypatch):
    server = make(db, tmp_path, limits())
    created, attempts = [], []
    def factory():
        created.append(1)
        return shared_deploy
    server.add(name='model', handler_factory=factory, config=config(startup_timeout_seconds=.3))
    original = server.devices._begin
    def lost(*args, **kwargs):
        granted = original(*args, **kwargs)
        attempts.append(granted)
        if len(attempts) == 1:
            raise OSError('Grant committed, acknowledgement lost')
        return granted
    async def scenario():
        try:
            monkeypatch.setattr(server.devices, '_begin', lost)
            await server.start_entry('model')
            assert created == [1] and attempts == [True, True]
            assert activity(server) == {'running': 0, 'waiting': 0}
            await server.stop_entry('model')
            def offline(*args, **kwargs):
                raise OSError('Database offline')
            monkeypatch.setattr(server.devices, '_begin', offline)
            with pytest.raises(ServerError) as failure:
                await server.start_entry('model')
            assert failure.value.code == 'device_coordination_unavailable' and created == [1]
            assert activity(server) == {'running': 0, 'waiting': 0}
        finally:
            await server.close()
    asyncio.run(scenario())


def test_failed_initializer_keeps_execution_slot_until_cleanup_is_confirmed(db, tmp_path, monkeypatch):
    from nndeploy.server.worker import Worker
    entered, release = threading.Event(), threading.Event()
    class Handler:
        def init(self):
            entered.set()
            assert release.wait(5)
            raise RuntimeError('Initialization failed')
        def deinit(self):
            pass
        def deploy(self, inputs):
            return inputs
    server = make(db, tmp_path, limits())
    server.add(name='bad', handler_factory=Handler, config=config())
    original_close = Worker.close
    async def unconfirmed(self):
        if self.config.name == 'bad':
            raise RuntimeError('Could not confirm physical cleanup')
        await original_close(self)
    monkeypatch.setattr(Worker, 'close', unconfirmed)
    async def scenario():
        starting = None
        try:
            starting = asyncio.create_task(server.start_entry('bad'))
            await until(entered.is_set)
            assert activity(server)['running'] == 1
            release.set()
            with pytest.raises(RuntimeError):
                await starting
            assert activity(server)['running'] == 1
            server.devices.recover()
            assert activity(server)['running'] == 1
            monkeypatch.setattr(Worker, 'close', original_close)
            for worker in server.entries['bad']['workers']:
                await worker.close()
            await server._release_entry_files(server.entries['bad'])
            server.entries['bad']['workers'] = []
            assert activity(server) == {'running': 0, 'waiting': 0}
        finally:
            release.set()
            if starting:
                await asyncio.gather(starting, return_exceptions=True)
            monkeypatch.setattr(Worker, 'close', original_close)
            await server.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('isolation', ['thread', 'process'])
def test_device_wait_and_warmup_share_one_startup_deadline(db, tmp_path, isolation):
    server = make(db, tmp_path, limits())
    server.add(name='busy', deploy=shared_deploy, config=config())
    busy_marker, warm_marker, busy_release, warm_release = [tmp_path / name for name in ('busy', 'warm', 'busy-release', 'warm-release')]
    async def scenario():
        starting = None
        try:
            await server.start()
            _, busy = await server.submit('busy', {'marker': str(busy_marker), 'release': str(busy_release)})
            await until(busy_marker.exists)
            server.add(name='new', deploy=shared_deploy, config=config(isolation=isolation, startup_timeout_seconds=2.0,
                warmup=[{'inputs': {'marker': str(warm_marker), 'release': str(warm_release)}}]))
            starting = asyncio.create_task(server.start_entry('new'))
            await until(lambda: activity(server)['waiting'] == 1)
            await asyncio.sleep(.6)
            busy_release.touch()
            await busy
            await until(warm_marker.exists)
            # If the startup deadline were reset after acquiring the device,
            # this release would let initialization succeed. The original
            # deadline has already elapsed, even though native work may live on.
            await asyncio.sleep(1.5)
            if isolation == 'thread':
                assert not starting.done() and activity(server)['running'] == 1
            warm_release.touch()
            with pytest.raises(ServerError) as failure:
                await starting
            assert failure.value.code == 'startup_timeout'
            assert server.describe('new')['status'] == 'failed'
            assert activity(server) == {'running': 0, 'waiting': 0}
            assert (await server.deploy('busy', {}))['outputs']['text'] == 'old'
        finally:
            busy_release.touch()
            warm_release.touch()
            if starting:
                await asyncio.gather(starting, return_exceptions=True)
            await server.close()
    asyncio.run(scenario())


def test_host_startup_wait_is_owned_after_caller_cancel_and_preserves_reserved_capacity(db, tmp_path):
    settings = limits(2, {'reserved': 1}, host=uuid.uuid4().hex)
    first, second = [make(db, tmp_path / str(i), settings) for i in range(2)]
    first.add(name='general', deploy=shared_deploy, config=config())
    warm_marker, release = tmp_path / 'warm', tmp_path / 'release'
    warm_cfg = config(warmup=[{'inputs': {'marker': str(warm_marker), 'release': str(release)}}])
    second.add(name='waiting', deploy=shared_deploy, config=warm_cfg)
    marker, busy_release = tmp_path / 'busy', tmp_path / 'busy-release'
    async def scenario():
        starting = None
        try:
            await first.start()
            _, busy = await first.submit('general', {'marker': str(marker), 'release': str(busy_release)})
            await until(marker.exists)
            starting = asyncio.create_task(second.start_entry('waiting'))
            await until(lambda: activity(first)['waiting'] == 1)
            starting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await starting
            first.add(name='reserved', deploy=shared_deploy, config=config())
            await asyncio.wait_for(first.start_entry('reserved'), 3)
            assert (await first.deploy('reserved', {}))['outputs']['text'] == 'old'
            assert not warm_marker.exists()
            busy_release.touch()
            await busy
            await until(warm_marker.exists)
            assert second.describe('waiting')['status'] == 'starting'
            release.touch()
            await until(lambda: second.describe('waiting')['status'] == 'ready')
            assert activity(first) == {'running': 0, 'waiting': 0}
        finally:
            release.touch()
            busy_release.touch()
            if starting:
                await asyncio.gather(starting, return_exceptions=True)
            await asyncio.gather(first.close(), second.close())
    asyncio.run(scenario())


def test_shared_worker_warmup_uses_same_device_capacity_as_existing_redis_tasks(backend, tmp_path):
    db, redis = backend
    settings = limits()
    options = dict(data_dir=tmp_path / 'shared', device_limits=settings, redis_config=redis,
                   database_url=db.url.render_as_string(hide_password=False))
    first, second = Server(**options), Server(**options)
    first.add(name='busy', deploy=shared_deploy, config=config())
    busy_marker, warm_marker, busy_release, warm_release = [tmp_path / name for name in ('busy', 'warm', 'busy-release', 'warm-release')]
    second.add(name='warming', deploy=shared_deploy, config=config(warmup=[{'inputs': {'marker': str(warm_marker), 'release': str(warm_release)}}]))
    async def scenario():
        starting = None
        try:
            await first.start()
            _, busy = await first.submit('busy', {'marker': str(busy_marker), 'release': str(busy_release)})
            await until(busy_marker.exists)
            starting = asyncio.create_task(second.start())
            await until(lambda: activity(first)['waiting'] == 1)
            assert not warm_marker.exists()
            busy_release.touch()
            await busy
            await until(warm_marker.exists)
            assert activity(first)['running'] == 1 and second.describe('warming')['status'] == 'starting'
            warm_release.touch()
            await starting
            result = await second.deploy('warming', {'text': 'actual'})
            assert result['outputs']['text'] == 'actual'
            with db.engine.connect() as connection:
                assert connection.execute(select(func.count()).select_from(tasks).where(tasks.c.name == 'warming',
                    tasks.c.queue_key == db.get_task(result['task_id'])['queue_key'])).scalar_one() == 1
        finally:
            busy_release.touch()
            warm_release.touch()
            if starting:
                await asyncio.gather(starting, return_exceptions=True)
            await asyncio.gather(first.close(), second.close())
    asyncio.run(scenario())
