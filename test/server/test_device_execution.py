"""Device execution coordination exercised through existing Server tasks."""
import asyncio
import uuid

import pytest
from pydantic import ValidationError
from sqlalchemy import select, update, text

from nndeploy.server.config import ExecutionConfig
from nndeploy.server import Server
from nndeploy.server.db import DB
from nndeploy.server.devices import DeviceLimits, runs, run_devices, claims
from nndeploy.server.errors import ServerError
from test_db import db
from test_devices import make, configuration
from test_shared_server import shared_deploy, until
from test_dynamic_batching import VectorBatch, BatchOutput, batch_config
from test_engine import build_engine, engine_config
from test_concurrent import build_thread_safe_handler, concurrent_config
from test_redis_queue import backend, redis_service


def limits(capacity=1, reserved=None, devices=('cpu',), host=None):
    result = {'devices': {device: {'max_workers': 16, 'max_concurrency': capacity,
                                  'reserved_slots': reserved or {}} for device in devices}}
    if host:
        result.update(scope='host', host_id=host)
        for name, value in result['devices'].items():
            value['identity'] = name
    return result


def config(devices=('cpu',), **execution):
    result = configuration()
    result['execution'].update(devices=list(devices), memory_bytes={}, **execution)
    return result


def activity(server, device='cpu'):
    return server.devices.snapshot()[device]['execution']


def test_execution_policy_validation_and_old_defaults():
    assert 'priority' not in ExecutionConfig().model_dump()
    for policy in [limits(0), limits(1, {'a': 2}), limits(1, {'a': 0}), limits(1, {'a': True}),
                   limits(1, {'bad name': 1}), {'devices': {'cpu': {'max_workers': 1, 'reserved_slots': {'a': 1}}}}]:
        with pytest.raises(ValidationError):
            DeviceLimits.model_validate(policy)
    with pytest.raises(ValidationError):
        ExecutionConfig(priority=11)


@pytest.mark.parametrize('mode', ['thread', 'process', 'concurrent', 'engine'])
def test_actual_computation_occupies_slot_and_disjoint_device_progresses(db, tmp_path, mode):
    server = make(db, tmp_path, limits(devices=('cpu', 'other')))
    if mode == 'engine':
        cfg, factory = engine_config(), build_engine
    elif mode == 'concurrent':
        cfg, factory = concurrent_config(), build_thread_safe_handler
    else:
        cfg, factory = config(isolation=mode, workers=2), None
    cfg['execution'].update(devices=['cpu'])
    server.add(name='model', config=cfg, **({'handler_factory': factory} if factory else {'deploy': shared_deploy}))
    server.add(name='independent', deploy=shared_deploy, config=config(('other',)))
    release, marker, second_marker = (tmp_path / name for name in ('release', 'first', 'second'))
    async def scenario():
        try:
            await server.start()
            _, first = await server.submit('model', {'marker': str(marker), 'release': str(release)})
            await until(marker.exists)
            _, second = await server.submit('model', {'marker': str(second_marker)})
            await until(lambda: activity(server) == {'running': 1, 'waiting': 1})
            result = await asyncio.wait_for(server.deploy('independent', {'text': 'free device'}), 3)
            assert result['outputs']['text'] == 'free device' and not second_marker.exists()
            release.touch()
            await asyncio.gather(first, second)
            await until(lambda: activity(server) == {'running': 0, 'waiting': 0})
        finally:
            release.touch()
            await server.close()
    asyncio.run(scenario())


def test_host_servers_share_execution_budget_and_multi_device_acquisition_is_atomic(db, tmp_path):
    settings = limits(devices=('cpu', 'other'), host=uuid.uuid4().hex)
    first, second = [make(db, tmp_path / str(i), settings) for i in range(2)]
    first.add(name='busy', deploy=shared_deploy, config=config(('other',)))
    second.add(name='both', deploy=shared_deploy, config=config(('cpu', 'other')))
    second.add(name='cpu', deploy=shared_deploy, config=config())
    release, marker, blocked = (tmp_path / name for name in ('release', 'first', 'blocked'))
    async def scenario():
        try:
            await first.start()
            await second.start()
            _, running = await first.submit('busy', {'marker': str(marker), 'release': str(release)})
            await until(marker.exists)
            _, waiting = await second.submit('both', {'marker': str(blocked)})
            await until(lambda: activity(first, 'cpu')['waiting'] == 1)
            assert activity(first, 'cpu')['running'] == 0
            assert activity(first, 'other')['running'] == 1
            assert not blocked.exists()
            release.touch()
            await asyncio.gather(running, waiting)
            assert blocked.exists()
        finally:
            release.touch()
            await asyncio.gather(first.close(), second.close())
    asyncio.run(scenario())


def test_reserved_entry_progresses_past_waiters_for_general_capacity(db, tmp_path):
    server = make(db, tmp_path, limits(2, {'reserved': 1}))
    server.add(name='general', deploy=shared_deploy, config=config(workers=2))
    server.add(name='reserved', deploy=shared_deploy, config=config())
    release, marker, blocked = (tmp_path / name for name in ('release', 'first', 'blocked'))
    async def scenario():
        try:
            await server.start()
            _, first = await server.submit('general', {'marker': str(marker), 'release': str(release)})
            await until(marker.exists)
            _, waiting = await server.submit('general', {'marker': str(blocked)})
            await until(lambda: activity(server)['waiting'] == 1)
            assert (await asyncio.wait_for(server.deploy('reserved', {'text': 'reserved'}), 3))['outputs']['text'] == 'reserved'
            assert not blocked.exists() and not first.done()
            release.touch()
            await asyncio.gather(first, waiting)
        finally:
            release.touch()
            await server.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('cancel', [False, True])
def test_waiting_deadline_and_cancellation_never_invoke_algorithm(db, tmp_path, cancel):
    server = make(db, tmp_path, limits())
    server.add(name='busy', deploy=shared_deploy, config=config())
    server.add(name='waiting', deploy=shared_deploy, config=config(queue_timeout_seconds=.5 if not cancel else 10.0))
    release, marker, blocked = (tmp_path / name for name in ('release', 'first', 'blocked'))
    async def scenario():
        try:
            await server.start()
            _, first = await server.submit('busy', {'marker': str(marker), 'release': str(release)})
            await until(marker.exists)
            identity, waiting = await server.submit('waiting', {'marker': str(blocked)})
            await until(lambda: activity(server)['waiting'] == 1)
            if cancel:
                await server.cancel_task(identity)
            with pytest.raises(ServerError) as failure:
                await asyncio.wait_for(asyncio.shield(waiting), 3)
            assert failure.value.code == ('task_cancelled' if cancel else 'queue_timeout')
            assert not blocked.exists() and not first.done()
            assert db.get_task(identity)['state'] == ('cancelled' if cancel else 'expired')
            await until(lambda: activity(server) == {'running': 1, 'waiting': 0})
            release.touch()
            await first
        finally:
            release.touch()
            await server.close()
    asyncio.run(scenario())


def test_running_thread_cancellation_does_not_release_physical_slot(db, tmp_path):
    server = make(db, tmp_path, limits())
    server.add(name='model', deploy=shared_deploy, config=config(workers=2))
    release, marker, blocked = (tmp_path / name for name in ('release', 'first', 'blocked'))
    async def scenario():
        try:
            await server.start()
            identity, first = await server.submit('model', {'marker': str(marker), 'release': str(release)})
            await until(marker.exists)
            _, second = await server.submit('model', {'marker': str(blocked)})
            await until(lambda: activity(server)['waiting'] == 1)
            await server.cancel_task(identity)
            assert activity(server) == {'running': 1, 'waiting': 1} and not blocked.exists()
            release.touch()
            await asyncio.gather(first, second, return_exceptions=True)
            assert blocked.exists()
        finally:
            release.touch()
            await server.close()
    asyncio.run(scenario())


def test_dynamic_batch_larger_than_device_capacity_uses_one_slot(db, tmp_path):
    server = make(db, tmp_path, limits())
    cfg = batch_config(size=3, wait=200)
    cfg['execution']['devices'] = ['cpu']
    server.add(name='vectors', handler_factory=VectorBatch, config=cfg, output_model=BatchOutput)
    server.add(name='other', deploy=shared_deploy, config=config())
    release, marker, blocked = (tmp_path / name for name in ('release', 'batch', 'blocked'))
    async def scenario():
        try:
            await server.start()
            submitted = await asyncio.gather(*(server.submit('vectors', {'values': [i], 'marker': str(marker), 'release': str(release)}) for i in range(3)))
            await until(marker.exists)
            assert marker.read_text() == '3' and activity(server)['running'] == 1
            _, other = await server.submit('other', {'marker': str(blocked)})
            await until(lambda: activity(server)['waiting'] == 1)
            assert not blocked.exists()
            release.touch()
            results = await asyncio.gather(*(future for _, future in submitted))
            assert [result['outputs']['total'] for result in results] == [0, 1, 2]
            assert all(result['outputs']['batch_size'] == 3 and result['outputs']['task_id'] == identity
                       for (identity, _), result in zip(submitted, results))
            await other
        finally:
            release.touch()
            await server.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('aged', [False, True])
def test_priority_with_waiting_age_prevents_permanent_starvation(db, tmp_path, aged):
    server = make(db, tmp_path, limits())
    for name, priority in [('busy', 0), ('low', -10), ('high', 10)]:
        server.add(name=name, deploy=shared_deploy, config=config(priority=priority, queue_timeout_seconds=60.0))
    busy_release, winner_release = tmp_path / 'busy-release', tmp_path / 'winner-release'
    busy_marker, low_marker, high_marker = (tmp_path / name for name in ('busy', 'low', 'high'))
    async def scenario():
        try:
            await server.start()
            _, busy = await server.submit('busy', {'marker': str(busy_marker), 'release': str(busy_release)})
            await until(busy_marker.exists)
            _, low = await server.submit('low', {'marker': str(low_marker), 'release': str(winner_release)})
            await until(lambda: activity(server)['waiting'] == 1)
            if aged:
                # Advance only the existing waiter's scheduling age. No sleep
                # or fake task ownership; actual algorithms verify run order.
                with db.engine.begin() as connection:
                    connection.execute(update(runs).where(runs.c.scope == server.devices.scope, runs.c.name == 'low')
                                       .values(rank=runs.c.rank - 25))
            _, high = await server.submit('high', {'marker': str(high_marker), 'release': str(winner_release)})
            await until(lambda: activity(server)['waiting'] == 2)
            busy_release.touch()
            winner, loser = (low_marker, high_marker) if aged else (high_marker, low_marker)
            await until(winner.exists)
            assert not loser.exists()
            winner_release.touch()
            await asyncio.gather(busy, low, high)
        finally:
            busy_release.touch()
            winner_release.touch()
            await server.close()
    asyncio.run(scenario())


def test_commit_ack_loss_and_release_failure_do_not_repeat_or_fail_computation(db, tmp_path, monkeypatch):
    server = make(db, tmp_path, limits())
    server.add(name='model', deploy=shared_deploy, config=config())
    original_begin, original_end = server.devices._begin, server.devices._end
    counts = {'grants': 0, 'ends': 0}
    def uncertain_begin(*args):
        granted = original_begin(*args)
        if granted:
            counts['grants'] += 1
            if counts['grants'] == 1:
                raise OSError('Committed grant, lost acknowledgement')
        return granted
    def failed_end(*args):
        counts['ends'] += 1
        if counts['ends'] == 1:
            raise OSError('Release database unavailable')
        return original_end(*args)
    async def scenario():
        try:
            await server.start()
            monkeypatch.setattr(server.devices, '_begin', uncertain_begin)
            monkeypatch.setattr(server.devices, '_end', failed_end)
            executor = server.entries['model']['workers'][0].executor
            original_deploy, calls = executor.deploy, []
            def counted(*args):
                calls.append(1)
                return original_deploy(*args)
            monkeypatch.setattr(executor, 'deploy', counted)
            result = await server.deploy('model', {'text': 'once'})
            assert result['outputs']['text'] == 'once' and db.get_task(result['task_id'])['state'] == 'succeeded'
            await until(lambda: activity(server) == {'running': 0, 'waiting': 0})
            assert counts['grants'] == 2 and counts['ends'] >= 2
            assert calls == [1]
        finally:
            await server.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('batch', [False, True])
def test_redis_replicas_share_device_gate_and_preserve_task_results(backend, tmp_path, batch):
    db, redis = backend
    settings = limits(host=uuid.uuid4().hex)
    servers = [Server(data_dir=tmp_path / 'shared', device_limits=settings,
                      database_url=db.url.render_as_string(hide_password=False), redis_config=redis) for i in range(2)]
    for server in servers:
        server.add(name='busy', deploy=shared_deploy, config=config())
        if batch:
            cfg = batch_config(size=2, wait=100)
            cfg['execution']['devices'] = ['cpu']
            server.add(name='waiting', handler_factory=VectorBatch, config=cfg, output_model=BatchOutput)
        else:
            server.add(name='waiting', deploy=shared_deploy, config=config())
    release, marker, blocked = (tmp_path / name for name in ('release', 'first', 'blocked'))
    async def scenario():
        try:
            for server in servers:
                await server.start()
            _, first = await servers[0].submit('busy', {'marker': str(marker), 'release': str(release)})
            await until(marker.exists)
            identity, second = await servers[1].submit('waiting', {'marker': str(blocked), **({'values': [4]} if batch else {})})
            await until(lambda: activity(servers[0])['waiting'] == 1)
            assert db.get_task(identity)['state'] == 'running' and not blocked.exists()
            assert activity(servers[1]) == {'running': 1, 'waiting': 1}
            release.touch()
            await first
            result = await second
            assert result['task_id'] == identity and db.get_task(identity)['state'] == 'succeeded'
            assert blocked.exists()
            if batch:
                assert result['outputs']['total'] == 4
            await until(lambda: activity(servers[0]) == {'running': 0, 'waiting': 0})
        finally:
            release.touch()
            await asyncio.gather(*(server.close() for server in servers))
    asyncio.run(scenario())


@pytest.mark.parametrize('cancel', [False, True])
def test_redis_device_wait_obeys_deadline_and_remote_cancellation(backend, tmp_path, cancel):
    db, redis = backend
    settings = limits(host=uuid.uuid4().hex)
    servers = [Server(data_dir=tmp_path / 'shared', device_limits=settings,
                      database_url=db.url.render_as_string(hide_password=False), redis_config=redis) for i in range(2)]
    for server in servers:
        server.add(name='busy', deploy=shared_deploy, config=config())
        server.add(name='waiting', deploy=shared_deploy, config=config(queue_timeout_seconds=10.0 if cancel else .5))
    release, marker, blocked = (tmp_path / name for name in ('release', 'first', 'blocked'))
    async def scenario():
        try:
            for server in servers:
                await server.start()
            _, first = await servers[0].submit('busy', {'marker': str(marker), 'release': str(release)})
            await until(marker.exists)
            identity, second = await servers[1].submit('waiting', {'marker': str(blocked)})
            await until(lambda: activity(servers[0])['waiting'] == 1)
            if cancel:
                await servers[0].cancel_task(identity)
            with pytest.raises(ServerError) as error:
                await asyncio.wait_for(asyncio.shield(second), 4)
            assert error.value.code == ('task_cancelled' if cancel else 'queue_timeout')
            assert not blocked.exists() and not first.done()
            await until(lambda: activity(servers[0]) == {'running': 1, 'waiting': 0})
            release.touch()
            await first
        finally:
            release.touch()
            await asyncio.gather(*(server.close() for server in servers))
    asyncio.run(scenario())


def test_cancelled_batch_member_is_filtered_after_device_wait(db, tmp_path):
    server = make(db, tmp_path, limits())
    cfg = batch_config(size=2, wait=100)
    cfg['execution']['devices'] = ['cpu']
    server.add(name='vectors', handler_factory=VectorBatch, config=cfg, output_model=BatchOutput)
    server.add(name='busy', deploy=shared_deploy, config=config())
    release, marker = tmp_path / 'release', tmp_path / 'busy'
    async def scenario():
        try:
            await server.start()
            _, first = await server.submit('busy', {'marker': str(marker), 'release': str(release)})
            await until(marker.exists)
            submitted = await asyncio.gather(*(server.submit('vectors', {'values': [i]}) for i in range(2)))
            await until(lambda: activity(server)['waiting'] == 1)
            await server.cancel_task(submitted[0][0])
            release.touch()
            await first
            results = await asyncio.gather(*(future for _, future in submitted), return_exceptions=True)
            assert isinstance(results[0], ServerError) and results[0].code == 'task_cancelled'
            assert results[1]['outputs']['total'] == 1 and results[1]['outputs']['batch_size'] == 1
        finally:
            release.touch()
            await server.close()
    asyncio.run(scenario())


def test_device_database_unavailable_never_invokes_algorithm(db, tmp_path, monkeypatch):
    server = make(db, tmp_path, limits())
    server.add(name='model', deploy=shared_deploy, config=config(queue_timeout_seconds=.2))
    marker = tmp_path / 'should-not-exist'
    def unavailable(*args):
        raise OSError('No durable admission confirmation')
    async def scenario():
        try:
            await server.start()
            monkeypatch.setattr(server.devices, '_begin', unavailable)
            with pytest.raises(ServerError) as failure:
                await asyncio.wait_for(server.deploy('model', {'marker': str(marker)}), 3)
            assert failure.value.code == 'device_coordination_unavailable' and not marker.exists()
            assert activity(server) == {'running': 0, 'waiting': 0}
        finally:
            await server.close()
    asyncio.run(scenario())


def test_migration_preserves_resident_ownership_without_inventing_running_requests(db, tmp_path):
    server = make(db, tmp_path, limits())
    server.add(name='model', deploy=shared_deploy, config=config())
    async def scenario():
        upgraded = None
        try:
            await server.start()
            with db.engine.begin() as connection:
                token = connection.execute(select(claims.c.token).where(claims.c.scope == server.devices.scope)).scalar_one()
                run_devices.drop(connection)
                runs.drop(connection)
                connection.execute(text("UPDATE alembic_version SET version_num = '0021_device_budgets'"))
            upgraded = DB(db.url).init_schema()
            with upgraded.engine.connect() as connection:
                assert connection.execute(select(claims.c.token).where(claims.c.scope == server.devices.scope)).scalar_one() == token
                assert not connection.execute(select(runs)).first()
            result = await server.deploy('model', {})
            assert result['outputs']['text'] == 'old'
        finally:
            await server.close()
            if upgraded:
                upgraded.close()
    asyncio.run(scenario())


def test_cancelled_caller_does_not_release_device_or_interrupt_accepted_task(db, tmp_path):
    server = make(db, tmp_path, limits())
    server.add(name='model', deploy=shared_deploy, config=config(workers=2))
    release, marker, blocked = (tmp_path / name for name in ('release', 'first', 'blocked'))
    async def scenario():
        caller = None
        try:
            await server.start()
            caller = asyncio.create_task(server.deploy('model', {'marker': str(marker), 'release': str(release)}))
            await until(marker.exists)
            identity = next(iter(server.running_workers))
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller
            _, second = await server.submit('model', {'marker': str(blocked)})
            await until(lambda: activity(server)['waiting'] == 1)
            assert activity(server)['running'] == 1 and not blocked.exists()
            assert db.get_task(identity)['state'] == 'running'
            release.touch()
            await second
            await until(lambda: db.get_task(identity)['state'] == 'succeeded')
            assert db.get_task(identity)['outputs']['text'] == 'old'
        finally:
            release.touch()
            if caller:
                await asyncio.gather(caller, return_exceptions=True)
            await server.close()
    asyncio.run(scenario())
