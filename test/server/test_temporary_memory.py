"""Declared temporary memory shares one budget with live model allocations."""
import asyncio
import copy
import threading
import uuid

import pytest
from pydantic import ValidationError
from sqlalchemy import select, text

from nndeploy.server import Server
from nndeploy.server.config import ExecutionConfig
from nndeploy.server.db import DB
from nndeploy.server.devices import claims, runs, run_devices
from nndeploy.server.errors import ServerError
from test_db import db
from test_devices import make
from test_device_execution import limits, config, activity
from test_shared_server import shared_deploy, until
from test_dynamic_batching import VectorBatch, batch_config
from test_engine import build_engine, engine_config
from test_concurrent import build_thread_safe_handler, concurrent_config
from test_redis_queue import backend, redis_service


def memory_limits(amount=100, devices=('cpu',), *, host=None, concurrency=None, reserved=None):
    settings = limits(concurrency or 1, reserved, devices, host)
    for value in settings['devices'].values():
        value['memory_bytes'] = amount
        if concurrency is None:
            value.pop('max_concurrency')
    return settings


def memory_config(resident=10, temporary=50, devices=('cpu',), **execution):
    cfg = config(devices, **execution)
    cfg['execution'].update(memory_bytes={device: resident for device in devices},
                            temporary_memory_bytes={device: temporary for device in devices} if temporary else {})
    return cfg


def memory(server, device='cpu'):
    return server.devices.snapshot()[device]['memory']


def test_temporary_configuration_is_bounded_and_default_fingerprints_unchanged():
    assert 'temporary_memory_bytes' not in ExecutionConfig().model_dump()
    for value in [{'temporary_memory_bytes': {'cpu': 1}},
                  {'devices': ['cpu'], 'temporary_memory_bytes': {'cpu': 0}},
                  {'devices': ['cpu'], 'temporary_memory_bytes': {'cpu': True}},
                  {'devices': ['cpu'], 'temporary_memory_bytes': {'cpu': 2**63}},
                  {'devices': ['cpu'], 'temporary_memory_bytes': {'cpu': 2**62}, 'batching': {'enabled': True, 'max_size': 2}}]:
        with pytest.raises(ValidationError):
            ExecutionConfig.model_validate(value)


@pytest.mark.parametrize('mode', ['thread', 'process', 'engine', 'concurrent'])
def test_memory_alone_limits_actual_calls_without_serializing_other_devices(db, tmp_path, mode):
    server = make(db, tmp_path, memory_limits(90, ('cpu', 'other')))
    if mode in ('thread', 'process'):
        cfg, source = memory_config(workers=2, isolation=mode), {'deploy': shared_deploy}
    else:
        cfg = engine_config() if mode == 'engine' else concurrent_config()
        cfg['execution'].update(devices=['cpu'], memory_bytes={'cpu': 10}, temporary_memory_bytes={'cpu': 50})
        source = {'handler_factory': build_engine if mode == 'engine' else build_thread_safe_handler}
    server.add(name='model', config=cfg, **source)
    server.add(name='other', deploy=shared_deploy, config=memory_config(devices=('other',)))
    marker, blocked, release = [tmp_path / name for name in ('first', 'second', 'release')]
    async def scenario():
        try:
            await server.start()
            _, first = await server.submit('model', {'marker': str(marker), 'release': str(release)})
            await until(marker.exists)
            _, second = await server.submit('model', {'marker': str(blocked)})
            await until(lambda: activity(server)['waiting'] == 1)
            assert memory(server)['temporary_bytes'] == 50
            assert memory(server)['used_bytes'] == (70 if mode in ('thread', 'process') else 60)
            assert (await server.deploy('other', {'text': 'independent'}))['outputs']['text'] == 'independent'
            assert not blocked.exists()
            release.touch()
            await asyncio.gather(first, second)
            await until(lambda: memory(server)['temporary_bytes'] == 0)
        finally:
            release.touch()
            await server.close()
    asyncio.run(scenario())


def test_new_model_preserves_existing_runtime_headroom_and_counts_live_temporary_bytes(db, tmp_path):
    server = make(db, tmp_path, memory_limits())
    server.add(name='active', deploy=shared_deploy, config=memory_config(10, 40, scheduling='concurrent', max_concurrency_per_worker=2))
    initialized = []
    def factory():
        initialized.append(1)
        return shared_deploy
    release = tmp_path / 'release'
    async def scenario():
        try:
            await server.start()
            submitted = []
            for i in range(2):
                marker = tmp_path / str(i)
                submitted.append(await server.submit('active', {'marker': str(marker), 'release': str(release)}))
                await until(marker.exists)
            assert memory(server)['used_bytes'] == 90
            server.add(name='new', handler_factory=factory, config=memory_config(20, 5))
            with pytest.raises(ServerError) as failure:
                await server.start_entry('new')
            assert failure.value.code == 'device_capacity_exceeded' and not initialized
            release.touch()
            await asyncio.gather(*(future for _, future in submitted))
            await until(lambda: memory(server)['temporary_bytes'] == 0)
            await server.start_entry('new')
            assert initialized == [1] and memory(server)['resident_bytes'] == 30
            server.add(name='too-large', handler_factory=factory, config=memory_config(40, 1))
            with pytest.raises(ServerError) as failure:
                await server.start_entry('too-large')
            assert failure.value.code == 'device_capacity_exceeded' and initialized == [1]
            assert memory(server)['required_headroom_bytes'] == 40
            assert (await server.deploy('active', {}))['outputs']['text'] == 'old'
        finally:
            release.touch()
            await server.close()
    asyncio.run(scenario())


def test_new_model_cannot_consume_unused_reserved_memory_while_general_calls_run(db, tmp_path):
    server = make(db, tmp_path, memory_limits(concurrency=4, reserved={'reserved': 1}))
    server.add(name='general', deploy=shared_deploy,
               config=memory_config(10, 30, scheduling='concurrent', max_concurrency_per_worker=2))
    server.add(name='reserved', deploy=shared_deploy, config=memory_config(10, 20))
    initialized = []
    def factory():
        initialized.append(1)
        return shared_deploy
    release = tmp_path / 'release'
    async def scenario():
        try:
            await server.start()
            pending = []
            for index in range(2):
                marker = tmp_path / str(index)
                _, future = await server.submit('general', {'marker': str(marker), 'release': str(release)})
                pending.append(future)
                await until(marker.exists)
            assert memory(server)['used_bytes'] == 80
            server.add(name='new', handler_factory=factory, config=memory_config(10, 0))
            with pytest.raises(ServerError) as failure:
                await server.start_entry('new')
            assert failure.value.code == 'device_capacity_exceeded' and not initialized
            assert (await asyncio.wait_for(server.deploy('reserved', {}), 3))['outputs']['text'] == 'old'
            assert all(not future.done() for future in pending)
            release.touch()
            await asyncio.gather(*pending)
            await until(lambda: memory(server)['temporary_bytes'] == 0)
            await server.start_entry('new')
            assert initialized == [1]
        finally:
            release.touch()
            await server.close()
    asyncio.run(scenario())


def test_reserved_slots_protect_memory_as_well_as_call_count(db, tmp_path):
    server = make(db, tmp_path, memory_limits(concurrency=4, reserved={'reserved': 1}))
    server.add(name='general', deploy=shared_deploy, config=memory_config(10, 30, scheduling='concurrent', max_concurrency_per_worker=2))
    server.add(name='reserved', deploy=shared_deploy, config=memory_config(10, 50))
    marker, blocked, release = [tmp_path / name for name in ('first', 'second', 'release')]
    async def scenario():
        try:
            await server.start()
            assert memory(server)['required_headroom_bytes'] == 80
            _, first = await server.submit('general', {'marker': str(marker), 'release': str(release)})
            await until(marker.exists)
            _, second = await server.submit('general', {'marker': str(blocked)})
            await until(lambda: activity(server)['waiting'] == 1)
            assert memory(server)['available_bytes'] == 50
            assert (await asyncio.wait_for(server.deploy('reserved', {}), 3))['outputs']['text'] == 'old'
            assert not blocked.exists() and not first.done()
            release.touch()
            await asyncio.gather(first, second)
        finally:
            release.touch()
            await server.close()
    asyncio.run(scenario())


def test_multidevice_temporary_memory_is_acquired_atomically(db, tmp_path):
    server = make(db, tmp_path, memory_limits(devices=('cpu', 'other')))
    server.add(name='busy', deploy=shared_deploy, config=memory_config(5, 60))
    server.add(name='both', deploy=shared_deploy, config=memory_config(5, 40, ('cpu', 'other')))
    marker, blocked, release = [tmp_path / name for name in ('first', 'second', 'release')]
    async def scenario():
        try:
            await server.start()
            _, first = await server.submit('busy', {'marker': str(marker), 'release': str(release)})
            await until(marker.exists)
            _, second = await server.submit('both', {'marker': str(blocked)})
            await until(lambda: activity(server)['waiting'] == 1)
            assert memory(server)['temporary_bytes'] == 60 and memory(server, 'other')['temporary_bytes'] == 0
            assert not blocked.exists()
            release.touch()
            await asyncio.gather(first, second)
            assert blocked.exists()
        finally:
            release.touch()
            await server.close()
    asyncio.run(scenario())


def test_dynamic_batch_preparation_and_computation_both_reserve_memory(db, tmp_path):
    server = make(db, tmp_path, memory_limits())
    cfg = batch_config(size=3, wait=1000)
    cfg['execution'].update(devices=['cpu'], memory_bytes={'cpu': 10}, temporary_memory_bytes={'cpu': 20})
    observed = []
    class InspectingBatch(VectorBatch):
        def batch_key(self, *args, **kwargs):
            observed.append(memory(server)['temporary_bytes'])
            return super().batch_key(*args, **kwargs)
    server.add(name='vectors', handler_factory=InspectingBatch, config=cfg)
    server.add(name='busy', deploy=shared_deploy, config=memory_config(10, 60))
    marker, busy_marker, release, busy_release = [tmp_path / name for name in ('batch', 'busy', 'release', 'busy-release')]
    async def scenario():
        try:
            await server.start()
            _, busy = await server.submit('busy', {'marker': str(busy_marker), 'release': str(busy_release)})
            await until(busy_marker.exists)
            submitted = await asyncio.gather(*(server.submit('vectors', {'values': [i], 'marker': str(marker), 'release': str(release)}) for i in range(3)))
            # Individual key computations fit (20 bytes), the full batch (60)
            # must wait for the other model's 60-byte call to finish.
            await until(lambda: len(server.entries['vectors']['workers'][0].jobs) == 3 and
                        all(job['ready'] for job in server.entries['vectors']['workers'][0].jobs.values()))
            await until(lambda: activity(server)['waiting'] == 1)
            assert not marker.exists() and memory(server)['temporary_bytes'] == 60
            assert observed == [80, 80, 80]
            busy_release.touch()
            await busy
            await until(marker.exists)
            assert marker.read_text() == '3' and memory(server)['temporary_bytes'] == 60
            release.touch()
            results = await asyncio.gather(*(future for _, future in submitted))
            assert [result['outputs']['total'] for result in results] == [0, 1, 2]
            assert all(result['outputs']['batch_size'] == 3 for result in results)
        finally:
            release.touch()
            busy_release.touch()
            await server.close()
    asyncio.run(scenario())


def test_cancelled_batch_member_reduces_waiting_memory_without_running_it(db, tmp_path):
    server = make(db, tmp_path, memory_limits())
    cfg = batch_config(size=2, wait=1000)
    cfg['execution'].update(devices=['cpu'], memory_bytes={'cpu': 10}, temporary_memory_bytes={'cpu': 30})
    server.add(name='vectors', handler_factory=VectorBatch, config=cfg)
    server.add(name='busy', deploy=shared_deploy, config=memory_config(10, 50))
    marker, busy_marker, release = [tmp_path / name for name in ('batch', 'busy', 'release')]
    async def scenario():
        try:
            await server.start()
            _, busy = await server.submit('busy', {'marker': str(busy_marker), 'release': str(release)})
            await until(busy_marker.exists)
            submitted = await asyncio.gather(*(server.submit('vectors', {'values': [i], 'marker': str(marker)}) for i in range(2)))
            await until(lambda: len(server.entries['vectors']['workers'][0].jobs) == 2 and
                        all(job['ready'] for job in server.entries['vectors']['workers'][0].jobs.values()))
            await until(lambda: activity(server)['waiting'] == 1)
            await server.cancel_task(submitted[0][0])
            results = await asyncio.gather(*(future for _, future in submitted), return_exceptions=True)
            assert isinstance(results[0], ServerError) and results[0].code == 'task_cancelled'
            assert results[1]['outputs']['total'] == 1 and results[1]['outputs']['batch_size'] == 1
            assert not busy.done() and marker.read_text() == '1'
            release.touch()
            await busy
        finally:
            release.touch()
            await server.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('cancel', [False, True])
def test_memory_wait_obeys_cancellation_and_queue_deadline(db, tmp_path, cancel):
    server = make(db, tmp_path, memory_limits(90))
    server.add(name='busy', deploy=shared_deploy, config=memory_config())
    server.add(name='waiting', deploy=shared_deploy, config=memory_config(queue_timeout_seconds=5.0 if cancel else .3))
    marker, blocked, release = [tmp_path / name for name in ('first', 'second', 'release')]
    async def scenario():
        try:
            await server.start()
            _, first = await server.submit('busy', {'marker': str(marker), 'release': str(release)})
            await until(marker.exists)
            identity, second = await server.submit('waiting', {'marker': str(blocked)})
            await until(lambda: activity(server)['waiting'] == 1)
            if cancel:
                await server.cancel_task(identity)
            with pytest.raises(ServerError) as failure:
                await asyncio.wait_for(asyncio.shield(second), 3)
            assert failure.value.code == ('task_cancelled' if cancel else 'queue_timeout')
            assert not blocked.exists() and memory(server)['temporary_bytes'] == 50 and not first.done()
            release.touch()
            await first
        finally:
            release.touch()
            await server.close()
    asyncio.run(scenario())


def test_warmup_waits_for_temporary_memory_and_cancellation_does_not_free_live_computation(db, tmp_path):
    server = make(db, tmp_path, memory_limits())
    server.add(name='busy', deploy=shared_deploy, config=memory_config(10, 80))
    marker, warmed, release, warm_release = [tmp_path / name for name in ('busy', 'warm', 'release', 'warm-release')]
    async def scenario():
        starting = None
        try:
            await server.start()
            identity, first = await server.submit('busy', {'marker': str(marker), 'release': str(release)})
            await until(marker.exists)
            server.add(name='new', deploy=shared_deploy, config=memory_config(10, 20, warmup=[{'inputs': {'marker': str(warmed), 'release': str(warm_release)}}]))
            starting = asyncio.create_task(server.start_entry('new'))
            await until(lambda: activity(server)['waiting'] == 1)
            assert not warmed.exists() and memory(server)['used_bytes'] == 100
            await server.cancel_task(identity)
            assert memory(server)['temporary_bytes'] == 80 and not warmed.exists()
            release.touch()
            await asyncio.gather(first, return_exceptions=True)
            await until(warmed.exists)
            assert memory(server)['temporary_bytes'] == 20 and server.describe('new')['status'] == 'starting'
            warm_release.touch()
            await starting
            assert memory(server)['temporary_bytes'] == 0
        finally:
            release.touch()
            warm_release.touch()
            if starting:
                await asyncio.gather(starting, return_exceptions=True)
            await server.close()
    asyncio.run(scenario())


def test_physical_aliases_share_temporary_memory_across_host_servers(db, tmp_path):
    settings = memory_limits(host=uuid.uuid4().hex)
    alias_settings = copy.deepcopy(settings)
    alias_settings['devices']['alias'] = alias_settings['devices'].pop('cpu')
    first, second = make(db, tmp_path / 'first', settings), make(db, tmp_path / 'second', alias_settings)
    first.add(name='first', deploy=shared_deploy, config=memory_config(10, 50))
    second.add(name='second', deploy=shared_deploy, config=memory_config(20, 60, ('alias',)))
    marker, blocked, release = [tmp_path / name for name in ('busy', 'blocked', 'release')]
    async def scenario():
        try:
            await first.start()
            await second.start()
            _, busy = await first.submit('first', {'marker': str(marker), 'release': str(release)})
            await until(marker.exists)
            _, waiting = await second.submit('second', {'marker': str(blocked)})
            await until(lambda: activity(second, 'alias')['waiting'] == 1)
            assert memory(first)['used_bytes'] == memory(second, 'alias')['used_bytes'] == 80
            assert not blocked.exists()
            release.touch()
            await asyncio.gather(busy, waiting)
        finally:
            release.touch()
            await asyncio.gather(first.close(), second.close())
    asyncio.run(scenario())


def test_release_failure_retains_memory_without_changing_successful_result(db, tmp_path, monkeypatch):
    server = make(db, tmp_path, memory_limits(90))
    server.add(name='model', deploy=shared_deploy, config=memory_config(workers=2))
    unavailable = threading.Event()
    original = server.devices._end
    def end(token):
        if unavailable.is_set():
            raise OSError('Temporary database outage during release')
        return original(token)
    async def scenario():
        try:
            await server.start()
            monkeypatch.setattr(server.devices, '_end', end)
            unavailable.set()
            result = await server.deploy('model', {'text': 'computed'})
            assert result['outputs']['text'] == 'computed' and db.get_task(result['task_id'])['state'] == 'succeeded'
            assert memory(server)['temporary_bytes'] == 50
            marker = tmp_path / 'next'
            _, waiting = await server.submit('model', {'marker': str(marker)})
            await until(lambda: activity(server)['waiting'] == 1)
            assert not marker.exists()
            unavailable.clear()
            assert (await asyncio.wait_for(asyncio.shield(waiting), 3))['outputs']['text'] == 'old'
            await until(lambda: memory(server)['temporary_bytes'] == 0)
        finally:
            unavailable.clear()
            await server.close()
    asyncio.run(scenario())


def test_memory_declarations_require_limits_and_impossible_peak_is_rejected_before_factory(db, tmp_path):
    first = make(db, tmp_path / 'no-limit', limits())
    second = make(db, tmp_path / 'too-small', memory_limits())
    called = []
    def factory():
        called.append(1)
        return VectorBatch()
    cfg = batch_config(size=3)
    cfg['execution'].update(devices=['cpu'], memory_bytes={'cpu': 10}, temporary_memory_bytes={'cpu': 40})
    first.add(name='model', handler_factory=factory, config=cfg)
    second.add(name='model', handler_factory=factory, config=cfg)
    async def scenario():
        try:
            for server, code in ((first, 'memory_limit_required'), (second, 'device_capacity_exceeded')):
                with pytest.raises(ServerError) as failure:
                    await server.start_entry('model')
                assert failure.value.code == code
            assert not called
        finally:
            await asyncio.gather(first.close(), second.close())
    asyncio.run(scenario())


def test_memory_integer_boundary_has_no_sql_sum_overflow(db, tmp_path):
    maximum = 2**63 - 1
    server = make(db, tmp_path, memory_limits(maximum))
    server.add(name='model', deploy=shared_deploy, config=memory_config(1, maximum - 1))
    marker, release = tmp_path / 'busy', tmp_path / 'release'
    async def scenario():
        try:
            await server.start()
            _, busy = await server.submit('model', {'marker': str(marker), 'release': str(release)})
            await until(marker.exists)
            assert memory(server)['used_bytes'] == maximum
            server.add(name='too-large', deploy=shared_deploy, config=memory_config(1, 0))
            with pytest.raises(ServerError) as failure:
                await server.start_entry('too-large')
            assert failure.value.code == 'device_capacity_exceeded'
            release.touch()
            await busy
            await until(lambda: memory(server)['temporary_bytes'] == 0)
            assert memory(server)['required_headroom_bytes'] == maximum - 1
        finally:
            release.touch()
            await server.close()
    asyncio.run(scenario())


def test_migration_preserves_live_legacy_ownership_without_fabricating_memory_usage(db, tmp_path):
    server = make(db, tmp_path, memory_limits(concurrency=1))
    server.add(name='model', deploy=shared_deploy, config=memory_config(10, 0))
    marker, release = tmp_path / 'busy', tmp_path / 'release'
    async def scenario():
        upgraded = None
        try:
            await server.start_entry('model')
            identity, future = await server.submit('model', {'marker': str(marker), 'release': str(release)})
            await until(marker.exists)
            with db.engine.begin() as connection:
                token = connection.execute(select(runs.c.token).where(runs.c.scope == server.devices.scope)).scalar_one()
                connection.execute(text('ALTER TABLE server_device_claims DROP COLUMN peak_memory_bytes'))
                connection.execute(text('ALTER TABLE server_device_run_devices DROP COLUMN memory_bytes'))
                connection.execute(text("UPDATE alembic_version SET version_num = '0022_device_execution'"))
            upgraded = DB(db.url).init_schema()
            with upgraded.engine.connect() as connection:
                assert connection.execute(select(runs.c.running).where(runs.c.token == token)).scalar_one()
                assert connection.execute(select(run_devices.c.memory_bytes).where(run_devices.c.token == token)).scalar_one() == 0
                assert connection.execute(select(claims.c.peak_memory_bytes).where(claims.c.scope == server.devices.scope)).scalar_one() == 0
            assert db.get_task(identity)['state'] == 'running' and memory(server)['resident_bytes'] == 10
            release.touch()
            await future
        finally:
            release.touch()
            await server.close()
            if upgraded:
                upgraded.close()
    asyncio.run(scenario())


def test_redis_replicas_share_temporary_memory_and_keep_completed_results(backend, tmp_path):
    database, redis = backend
    options = dict(data_dir=tmp_path / 'shared', database_url=database.url.render_as_string(hide_password=False),
                   redis_config=redis, device_limits=memory_limits())
    servers = [Server(name='model', deploy=shared_deploy, config=memory_config(10, 50), **options) for _ in range(2)]
    marker, blocked, release = [tmp_path / name for name in ('busy', 'blocked', 'release')]
    async def scenario():
        try:
            for server in servers:
                await server.start()
            _, first = await servers[0].submit('model', {'marker': str(marker), 'release': str(release)})
            await until(marker.exists)
            identity, second = await servers[1].submit('model', {'marker': str(blocked)})
            await until(lambda: activity(servers[0])['waiting'] == 1)
            assert memory(servers[0])['used_bytes'] == memory(servers[1])['used_bytes'] == 70
            assert not blocked.exists()
            release.touch()
            await asyncio.gather(first, second)
            assert database.get_task(identity)['state'] == 'succeeded'
            await until(lambda: memory(servers[0])['temporary_bytes'] == 0)
        finally:
            release.touch()
            await asyncio.gather(*(server.close() for server in servers))
    asyncio.run(scenario())


def test_model_and_execution_ack_loss_never_double_count_memory(db, tmp_path, monkeypatch):
    server = make(db, tmp_path, memory_limits())
    factories, executions = [], []
    def factory():
        factories.append(1)
        def deploy(values):
            executions.append(1)
            return shared_deploy(values)
        return deploy
    server.add(name='model', handler_factory=factory, config=memory_config())
    original_reserve, original_begin = server.devices._reserve, server.devices._begin
    attempts = {'reserve': 0, 'begin': 0}
    def uncertain_reserve(*args):
        original_reserve(*args)
        attempts['reserve'] += 1
        if attempts['reserve'] == 1:
            raise OSError('Model reservation committed but acknowledgement lost')
    def uncertain_begin(*args, **kwargs):
        result = original_begin(*args, **kwargs)
        if result:
            attempts['begin'] += 1
            if attempts['begin'] == 1:
                raise OSError('Execution memory committed but acknowledgement lost')
        return result
    monkeypatch.setattr(server.devices, '_reserve', uncertain_reserve)
    marker, release = tmp_path / 'busy', tmp_path / 'release'
    async def scenario():
        try:
            await server.start()
            assert factories == [1] and attempts['reserve'] == 2
            assert memory(server)['resident_bytes'] == 10 and memory(server)['required_headroom_bytes'] == 50
            monkeypatch.setattr(server.devices, '_begin', uncertain_begin)
            _, future = await server.submit('model', {'marker': str(marker), 'release': str(release)})
            await until(marker.exists)
            assert attempts['begin'] == 2 and executions == [1]
            assert memory(server)['used_bytes'] == 60 and memory(server)['temporary_bytes'] == 50
            release.touch()
            await future
            await until(lambda: memory(server)['temporary_bytes'] == 0)
        finally:
            release.touch()
            await server.close()
    asyncio.run(scenario())


def test_device_identifiers_remain_distinct_after_legacy_case_insensitive_mysql_upgrade(db, tmp_path):
    columns = [('server_device_claims', 'device'), ('server_device_claims', 'name'),
               ('server_device_run_devices', 'device'), ('server_device_runs', 'name')]
    mysql = db.engine.dialect.name == 'mysql'
    if mysql:
        with db.engine.begin() as connection:
            for table, column in columns:
                connection.execute(text(f'ALTER TABLE {table} MODIFY {column} VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci NOT NULL'))
            connection.execute(text("UPDATE alembic_version SET version_num = '0022_device_execution'"))
    server = None
    async def scenario():
        nonlocal server
        try:
            server = make(db, tmp_path, memory_limits(devices=('cpu', 'CPU')))
            server.add(name='model', deploy=shared_deploy, config=memory_config(10, 20, ('cpu', 'CPU')))
            await server.start()
            assert memory(server, 'cpu')['resident_bytes'] == memory(server, 'CPU')['resident_bytes'] == 10
            assert (await server.deploy('model', {}))['outputs']['text'] == 'old'
            with db.engine.connect() as connection:
                assert set(connection.execute(select(claims.c.device).where(claims.c.scope == server.devices.scope)).scalars()) == {'cpu', 'CPU'}
            await server.close()
            server = make(db, tmp_path / 'names', memory_limits(concurrency=2, reserved={'Model': 1}))
            server.add(name='Model', deploy=shared_deploy, config=memory_config(10, 50))
            server.add(name='model', deploy=shared_deploy, config=memory_config(10, 30))
            await server.start()
            assert memory(server)['required_headroom_bytes'] == 80
        finally:
            if server:
                await server.close()
            if mysql:
                # Keep the shared test database usable even when the old
                # migration reproduces the duplicate-key failure.
                with db.engine.begin() as connection:
                    for table, column in columns:
                        connection.execute(text(f'ALTER TABLE {table} MODIFY {column} VARCHAR(128) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL'))
    asyncio.run(scenario())
