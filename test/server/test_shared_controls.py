"""Control intent must not be overwritten by delayed shared-worker actions."""
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import uuid

import pytest
from sqlalchemy import inspect, text

from nndeploy.server import Server
from nndeploy.server.errors import ServerError
from nndeploy.server.worker import Worker
from test_redis_queue import redis_service, backend
from test_shared_server import shared_deploy, entry_config, until
from test_db import db
from nndeploy.server.db import DB
from test_server_api import text_graph


def text_factory():
    from nndeploy.dag import Graph
    graph = Graph('echo')
    graph.deserialize(json.dumps(text_graph()))
    return graph


def servers_for(backend, tmp_path):
    db, redis_config = backend
    config = entry_config()
    config['execution']['isolation'] = 'thread'
    return [Server(name='shared', deploy=shared_deploy, config=config, resources=tmp_path / 'shared', data_dir=tmp_path / 'shared',
                   database_url=db.url.render_as_string(hide_password=False), redis_config=redis_config) for _ in range(2)]


@pytest.mark.parametrize('hold_stop_return', [False, True])
def test_late_start_cannot_override_a_newer_stop_even_when_controller_is_already_stopped(backend, tmp_path, monkeypatch, hold_stop_return):
    db, _ = backend
    first, second = servers_for(backend, tmp_path)

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        stop_return = asyncio.Event()
        original_stop = second._stop_entry
        async def held_stop(*args, **kwargs):
            result = await original_stop(*args, **kwargs)
            if hold_stop_return and kwargs.get('persist') is False:
                await stop_return.wait()
            return result
        monkeypatch.setattr(second, '_stop_entry', held_stop)
        starting = None
        try:
            await first.start()
            await second.start()
            await first.stop_entry('shared')
            await until(lambda: second.describe('shared')['status'] == 'stopped')
            # The model can be stopped before the enclosing control operation
            # returns. Finish that operation before testing a later Start.
            stop_return.set()
            previous = second.entry_controls.get('shared')
            if previous is not None:
                assert previous['action'] == 'stop'
                await asyncio.shield(previous['task'])
            original = Worker.start
            async def delayed(worker):
                entered.set()
                await release.wait()
                await original(worker)
            monkeypatch.setattr(Worker, 'start', delayed)
            starting = asyncio.create_task(second.start_entry('shared'))
            await until(lambda: entered.is_set() or starting.done(), timeout=5)
            if starting.done():
                await starting
            assert entered.is_set()
            await first.stop_entry('shared')
            release.set()
            with pytest.raises(ServerError) as obsolete:
                await starting
            assert obsolete.value.code == 'shared_control_changed'
            assert not db.queue_usage(first.entries['shared']['queue'].key)['accepting']
            assert not second.entries['shared']['workers']
            assert second.describe('shared')['status'] == 'stopped'
            assert (await first.start_entry('shared'))['status'] == 'ready'
            await until(lambda: second.describe('shared')['status'] == 'ready')
            assert (await second.deploy('shared', {}))['outputs']['text'] == 'echo'
        finally:
            stop_return.set()
            release.set()
            if starting is not None:
                await asyncio.gather(starting, return_exceptions=True)
            await asyncio.gather(first.close(), second.close())
    asyncio.run(scenario())


def test_delayed_follower_stop_does_not_overwrite_a_new_start(backend, tmp_path, monkeypatch):
    db, _ = backend
    first, second = servers_for(backend, tmp_path)

    async def scenario():
        entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
        original = second._stop_entry
        async def delayed(*args, **kwargs):
            if kwargs.get('persist') is False and not entered.is_set():
                entered.set()
                await release.wait()
                try:
                    return await original(*args, **kwargs)
                finally:
                    finished.set()
            return await original(*args, **kwargs)
        try:
            await first.start()
            await second.start()
            monkeypatch.setattr(second, '_stop_entry', delayed)
            await first.stop_entry('shared')
            await asyncio.wait_for(entered.wait(), 5)
            # Freeze further polls so this delayed observation is the only one
            # being applied while a newer control intent is committed.
            second.shared_control.cancel()
            await asyncio.gather(second.shared_control, return_exceptions=True)
            await first.start_entry('shared')
            release.set()
            await asyncio.wait_for(finished.wait(), 5)
            assert db.queue_usage(first.entries['shared']['queue'].key)['accepting']
            assert (await first.deploy('shared', {}))['outputs']['text'] == 'echo'
        finally:
            release.set()
            await asyncio.gather(first.close(), second.close())
    asyncio.run(scenario())


def test_control_writes_fence_old_intents_and_support_uncertain_commit_replay(db):
    key, group = uuid.uuid4().hex, uuid.uuid4().hex
    db.register_queue(key, 4)
    db.register_queue(group, 8)
    original = db.queue_usage(key)['control_token']
    stopped = db.set_queue_accepting(key, False, expected_token=original)
    resumed = uuid.uuid4().hex
    assert db.set_queue_accepting(key, True, expected_token=stopped, write_token=resumed) == resumed
    # Model a lost commit acknowledgement: replay the identical conditional write.
    assert db.set_queue_accepting(key, True, expected_token=stopped, write_token=resumed) == resumed
    newer_stop = db.set_queue_accepting(key, False)
    with pytest.raises(ServerError) as stale:
        db.set_queue_accepting(key, True, expected_token=stopped, write_token=resumed)
    assert stale.value.code == 'shared_control_changed'
    assert db.queue_usage(key)['control_token'] == newer_stop
    assert not db.queue_usage(key)['accepting']
    # Even a repeated Stop invalidates initialization started from that state.
    repeat = db.set_queue_accepting(key, False)
    assert repeat != newer_stop
    with pytest.raises(ServerError):
        db.set_queue_accepting(key, True, expected_token=newer_stop)
    assert db.set_queue_accepting(key, True, expected_token=repeat)


def test_resume_requires_confirmed_drain_in_the_same_transaction(db):
    key, group, identity = uuid.uuid4().hex, uuid.uuid4().hex, 'task_' + uuid.uuid4().hex
    db.register_queue(key, 4)
    db.register_queue(group, 8)
    db.create_shared_task(identity, key, key, group, {}, {}, 30)
    stopped = db.set_queue_accepting(key, False)
    with pytest.raises(ServerError) as busy:
        db.set_queue_accepting(key, True, expected_token=stopped)
    assert busy.value.code == 'entry_busy'
    assert not db.queue_usage(key)['accepting'] and db.queue_usage(key)['inflight'] == 1
    db.finish_task(identity, 'cancelled', time.time())
    assert db.set_queue_accepting(key, True, expected_token=stopped)
    assert db.queue_usage(key)['accepting'] and db.queue_usage(key)['inflight'] == 0


def test_slow_follower_initialization_does_not_block_other_entry_controls(backend, tmp_path, monkeypatch):
    database, redis_config = backend
    settings = entry_config()
    settings['execution']['isolation'] = 'thread'
    server = Server(resources=tmp_path, data_dir=tmp_path, database_url=database.url.render_as_string(hide_password=False), redis_config=redis_config)
    for name in ('slow', 'fast'):
        server.add(name=name, deploy=shared_deploy, config=settings)

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        try:
            await server.start()
            await server.stop_entry('slow')
            await server.stop_entry('fast')
            original = Worker.start
            async def delayed(worker):
                if worker.config.name == 'slow':
                    entered.set()
                    await release.wait()
                await original(worker)
            monkeypatch.setattr(Worker, 'start', delayed)
            for name in ('slow', 'fast'):
                database.set_queue_accepting(server.entries[name]['queue'].key, True)
            await asyncio.wait_for(entered.wait(), 5)
            await until(lambda: server.describe('fast')['status'] == 'ready', timeout=2)
            assert server.describe('slow')['status'] == 'starting'
            assert (await server.deploy('fast', {}))['outputs']['text'] == 'echo'
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())


def test_concurrent_control_preparations_have_one_confirmed_winner(db):
    key = uuid.uuid4().hex
    db.register_queue(key, 4)
    original = db.queue_usage(key)['control_token']
    barrier = threading.Barrier(2)
    def control(accepting):
        barrier.wait(timeout=5)
        try:
            return accepting, db.set_queue_accepting(key, accepting, expected_token=original)
        except ServerError as exc:
            return exc.code
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(control, [True, False]))
    winners = [item for item in results if isinstance(item, tuple)]
    assert len(winners) == 1 and results.count('shared_control_changed') == 1
    row = db.queue_usage(key)
    assert (row['accepting'], row['control_token']) == winners[0]


def test_control_token_migration_preserves_stopped_queue_and_capacity(db):
    key, group, identity = uuid.uuid4().hex, uuid.uuid4().hex, 'task_' + uuid.uuid4().hex
    db.register_queue(key, 4)
    db.register_queue(group, 8)
    db.create_shared_task(identity, key, key, group, {}, {}, 30)
    db.set_queue_accepting(key, False)
    with db.engine.begin() as connection:
        connection.execute(text('ALTER TABLE server_queue_limits DROP COLUMN control_token'))
        connection.execute(text("UPDATE alembic_version SET version_num = '0014_request_ids'"))
    upgraded = DB(db.url).init_schema()
    try:
        row = upgraded.queue_usage(key)
        assert row['control_token'] == '0' and not row['accepting']
        assert row['capacity'] == 4 and row['inflight'] == 1
        column = next(item for item in inspect(upgraded.engine).get_columns('server_queue_limits') if item['name'] == 'control_token')
        assert not column['nullable'] and column['type'].length == 32
        upgraded.finish_task(identity, 'cancelled', time.time())
        assert upgraded.set_queue_accepting(key, True, expected_token='0')
        assert upgraded.queue_usage(key)['inflight'] == 0
    finally:
        upgraded.close()


def test_first_factory_discovery_cannot_resume_a_stop_it_could_not_observe(backend, tmp_path):
    database, redis_config = backend
    options = dict(name='factory', graph_factory=text_factory, resources=tmp_path, data_dir=tmp_path,
                   database_url=database.url.render_as_string(hide_password=False), redis_config=redis_config)
    first, second = Server(**options), Server(**options)

    async def scenario():
        try:
            await first.start()
            await first.stop_entry('factory')
            # The second instance has never discovered the factory's schema.
            with pytest.raises(ServerError) as discovered:
                await second.start_entry('factory')
            assert discovered.value.code == 'shared_control_changed'
            assert not database.queue_usage(first.entries['factory']['queue'].key)['accepting']
            assert not second.entries['factory']['workers']
            assert (await second.start_entry('factory'))['status'] == 'ready'
            assert (await second.deploy('factory', {'source': 'restored'}))['outputs'] == {'sink': 'restored'}
        finally:
            await asyncio.gather(first.close(), second.close())
    asyncio.run(scenario())


def test_factory_schema_drift_does_not_silently_switch_shared_queue_on_restart(backend, tmp_path, monkeypatch):
    database, redis_config = backend
    server = Server(name='factory', graph_factory=text_factory, resources=tmp_path, data_dir=tmp_path,
                    database_url=database.url.render_as_string(hide_password=False), redis_config=redis_config)

    async def scenario():
        try:
            await server.start()
            original_key = server.entries['factory']['queue'].key
            original_config = server.entries['factory']['config'].model_dump()
            await server.stop_entry('factory')
            changed = text_graph()
            changed['node_repository_'][1]['name_'] = 'different_sink'
            monkeypatch.setattr('test_shared_controls.text_graph', lambda: changed)
            with pytest.raises(ServerError) as drift:
                await server.start_entry('factory')
            assert drift.value.code == 'shared_configuration_changed'
            assert server.entries['factory']['queue'].key == original_key
            assert server.entries['factory']['config'].model_dump() == original_config
            assert not database.queue_usage(original_key)['accepting']
            assert not server.entries['factory']['workers']
        finally:
            await server.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('accepting', [True, False])
def test_control_commit_acknowledgement_loss_retries_same_intent(backend, tmp_path, monkeypatch, accepting):
    database, _ = backend
    first, second = servers_for(backend, tmp_path)

    async def scenario():
        try:
            await first.start()
            await second.start()
            if accepting:
                await first.stop_entry('shared')
                await until(lambda: second.describe('shared')['status'] == 'stopped')
            original = first.db.set_queue_accepting
            writes = []
            def lose_ack(key, value, **options):
                result = original(key, value, **options)
                writes.append((key, value, options))
                if len(writes) == 1:
                    raise OSError('Injected loss of a successful control commit acknowledgement')
                return result
            monkeypatch.setattr(first.db, 'set_queue_accepting', lose_ack)
            result = await (first.start_entry('shared') if accepting else first.stop_entry('shared'))
            assert result['status'] == ('ready' if accepting else 'stopped')
            assert len(writes) == 2 and writes[0] == writes[1]
            assert writes[0][2]['write_token']
            row = database.queue_usage(first.entries['shared']['queue'].key)
            assert row['accepting'] == accepting and row['control_token'] == writes[0][2]['write_token']
            await until(lambda: second.describe('shared')['status'] == result['status'])
        finally:
            await asyncio.gather(first.close(), second.close())
    asyncio.run(scenario())


@pytest.mark.parametrize('accepting', [True, False])
def test_retry_after_lost_ack_does_not_overwrite_a_newer_control(backend, tmp_path, monkeypatch, accepting):
    database, _ = backend
    first, second = servers_for(backend, tmp_path)

    async def scenario():
        try:
            await first.start()
            await second.start()
            if accepting:
                await first.stop_entry('shared')
                await until(lambda: second.describe('shared')['status'] == 'stopped')
            original = first.db.set_queue_accepting
            written = False
            def supersede(key, value, **options):
                nonlocal written
                result = original(key, value, **options)
                if not written:
                    written = True
                    database.set_queue_accepting(key, not value)
                    raise OSError('Commit succeeded, then a newer control superseded it')
                return result
            monkeypatch.setattr(first.db, 'set_queue_accepting', supersede)
            with pytest.raises(ServerError) as stale:
                await (first.start_entry('shared') if accepting else first.stop_entry('shared'))
            assert stale.value.code == 'shared_control_changed'
            assert database.queue_usage(first.entries['shared']['queue'].key)['accepting'] == (not accepting)
            if accepting:
                assert first.describe('shared')['status'] == 'stopped' and not first.entries['shared']['workers']
            else:
                assert (await first.deploy('shared', {}))['outputs']['text'] == 'echo'
        finally:
            await asyncio.gather(first.close(), second.close())
    asyncio.run(scenario())


def test_shutdown_reports_unconfirmed_shared_control_and_only_closes_local_workers(backend, tmp_path, monkeypatch):
    database, _ = backend
    first, second = servers_for(backend, tmp_path)

    async def scenario():
        attempted = threading.Event()
        stopping = None
        try:
            await first.start()
            await second.start()
            def unavailable(*args, **kwargs):
                attempted.set()
                raise OSError('Control database temporarily unavailable')
            monkeypatch.setattr(first.db, 'set_queue_accepting', unavailable)
            stopping = asyncio.create_task(first.stop_entry('shared'))
            assert await asyncio.to_thread(attempted.wait, 5)
            await asyncio.wait_for(first.close(), 5)
            with pytest.raises(ServerError) as unconfirmed:
                await stopping
            assert unconfirmed.value.code == 'shared_control_unconfirmed'
            assert not first.entries['shared']['workers']
            assert database.queue_usage(second.entries['shared']['queue'].key)['accepting']
            assert (await second.deploy('shared', {}))['outputs']['text'] == 'echo'
        finally:
            if stopping is not None:
                await asyncio.gather(stopping, return_exceptions=True)
            await asyncio.gather(first.close(), second.close())
    asyncio.run(scenario())


def test_start_rejects_known_shared_drain_before_loading_another_model(backend, tmp_path, monkeypatch):
    first, second = servers_for(backend, tmp_path)
    release, marker = tmp_path / 'release', tmp_path / 'entered'

    async def scenario():
        stopping = None
        try:
            await first.start()
            identity, future = await first.submit('shared', {'release': str(release), 'marker': str(marker)})
            await until(marker.exists)
            stopping = asyncio.create_task(first.stop_entry('shared'))
            await until(lambda: first.describe('shared')['status'] == 'stopping')
            starts = []
            original = Worker.start
            async def record(worker):
                starts.append(worker)
                await original(worker)
            monkeypatch.setattr(Worker, 'start', record)
            with pytest.raises(ServerError) as draining:
                await second.start_entry('shared')
            assert draining.value.code == 'entry_busy'
            assert not starts and second.describe('shared')['status'] == 'stopped'
            assert not first.db.get_task(identity)['cancel_requested']
            release.touch()
            assert (await future)['outputs']['text'] == 'echo'
            await stopping
            assert (await second.start_entry('shared'))['status'] == 'ready'
            assert len(starts) >= 1
        finally:
            release.touch()
            if stopping is not None:
                await asyncio.gather(stopping, return_exceptions=True)
            await asyncio.gather(first.close(), second.close())
    asyncio.run(scenario())
