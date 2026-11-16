"""Restart restores queued work with its original configuration, never lost execution."""
import asyncio
from pathlib import Path

import pytest
from sqlalchemy import update

from nndeploy.server import Server
from nndeploy.server.db import tasks
from nndeploy.server.worker import Worker
from test_redis_queue import backend, redis_service
from test_shared_server import shared_deploy, until
from test_shared_reload import config
from test_shared_repair import options
from test_server_api import text_graph


def recorded_deploy(values):
    if values['marker']:
        with Path(values['marker']).open('a') as output:
            output.write(values['text'] + '\n')
    return shared_deploy(dict(values, marker=''))


async def leave_queued_old_work(server, new_config, requests):
    """Detach old consumers after switch to model a confirmed process exit.

    Real Workers initialize and close; no accepted task is executed by the
    detached consumers. DB and Redis retain the committed switch and backlog.
    """
    original = server._consume_shared
    detach = asyncio.Event()
    old = server.entries[next(iter(server.entries))]
    name = old['config'].name
    async def paused(entry, worker):
        if entry is old:
            await detach.wait()
        else:
            await original(entry, worker)
    server._consume_shared = paused
    reloading = None
    try:
        await server.start()
        server.shared_control.cancel()
        await asyncio.gather(server.shared_control, return_exceptions=True)
        key = old['queue'].key
        accepted = [await server.submit(name, values) for values in requests]
        reloading = asyncio.create_task(server.reload_entry(name, config=new_config))
        await until(lambda: server.entries[name]['queue'].key != key)
        detach.set()
        await reloading
        for _, observer in accepted:
            observer.cancel()
        await server.close()
        return key, [identity for identity, _ in accepted]
    finally:
        detach.set()
        if reloading is not None:
            await asyncio.gather(reloading, return_exceptions=True)
        await server.close()


@pytest.mark.parametrize('isolation', ['thread', 'process'])
def test_restarted_replicas_finish_old_queued_tasks_with_original_config(backend, tmp_path, isolation):
    database, _ = backend
    settings = options(backend, tmp_path)
    first = Server(name='model', deploy=recorded_deploy, config=config('old', isolation), **settings)
    resumed = [Server(name='model', deploy=recorded_deploy, config=config('old', isolation), **settings) for _ in range(2)]
    release, marker = tmp_path / 'release', tmp_path / 'calls'
    async def scenario():
        try:
            old_key, identities = await leave_queued_old_work(first, config('new', isolation),
                [{'marker': str(marker), 'release': str(release)}, {'marker': str(marker)}])
            assert all(database.get_task(identity)['state'] == 'pending' for identity in identities)
            for server in resumed:
                await server.start()
            await until(marker.exists)
            assert (await resumed[0].deploy('model', {}))['outputs']['text'] == 'new'
            release.touch()
            await until(lambda: all(database.get_task(identity)['state'] == 'succeeded' for identity in identities))
            assert marker.read_text().splitlines() == ['old', 'old']
            assert all(database.get_task(identity)['queue_key'] == old_key for identity in identities)
            assert all(database.get_task(identity)['outputs']['text'] == 'old' for identity in identities)
            await until(lambda: not database.shared.draining(resumed[0].entries['model']['shared_key']))
            assert database.queue_usage(resumed[0].entries['model']['queue'].group_key)['inflight'] == 0
        finally:
            release.touch()
            await asyncio.gather(first.close(), *(server.close() for server in resumed))
    asyncio.run(scenario())


def test_native_graph_recovery_uses_old_field_bindings_after_restart(backend, tmp_path):
    database, _ = backend
    settings = options(backend, tmp_path)
    def binding(prefix):
        return {'inputs': {prefix + '_input': {'type': 'string', 'node': ['source'], 'required': False, 'default': prefix}},
                'outputs': {prefix + '_output': {'type': 'string', 'edge': 'text'}}}
    first = Server(name='echo', workflow=text_graph(), config=binding('old'), **settings)
    async def scenario():
        resumed = None
        try:
            old_key, identities = await leave_queued_old_work(first, binding('new'), [{'old_input': 'accepted-before-reload'}])
            resumed = Server(**settings)
            await resumed.start()
            assert (await resumed.deploy('echo', {}))['outputs'] == {'new_output': 'new'}
            await until(lambda: database.get_task(identities[0])['state'] == 'succeeded')
            old = database.get_task(identities[0])
            assert old['outputs'] == {'old_output': 'accepted-before-reload'} and old['queue_key'] == old_key
            await until(lambda: not database.shared.draining(resumed.entries['echo']['shared_key']))
        finally:
            await first.close()
            if resumed is not None:
                await resumed.close()
    asyncio.run(scenario())


def test_queued_recovery_does_not_reexecute_stalled_running_task_or_release_its_slot(backend, tmp_path):
    database, _ = backend
    settings = options(backend, tmp_path)
    first = Server(name='model', deploy=recorded_deploy, config=config('old'), **settings)
    resumed = Server(name='model', deploy=recorded_deploy, config=config('old'), **settings)
    marker = tmp_path / 'calls'
    async def scenario():
        lost = None
        try:
            old_key, identities = await leave_queued_old_work(first, config('new'),
                [{'text': 'uncertain', 'marker': str(marker)}, {'text': 'queued', 'marker': str(marker)}])
            lost = identities[0]
            assert database.claim_task(lost, old_key, 'uncertain-worker', 1)
            with database.engine.begin() as connection:
                connection.execute(update(tasks).where(tasks.c.task_id == lost).values(lease_expires_at=0))
            await resumed.start()
            await until(lambda: database.get_task(identities[1])['state'] == 'succeeded')
            await until(lambda: not resumed.retiring_entries.get('model'))
            assert marker.read_text().splitlines() == ['queued']
            record = database.get_task(lost)
            assert record['state'] == 'running' and record['attempt'] == 1 and record['worker_token'] == 'uncertain-worker'
            assert database.queue_usage(old_key)['inflight'] == 1
            assert database.queue_usage(resumed.entries['model']['queue'].group_key)['inflight'] == 1
            assert (await resumed.deploy('model', {}))['outputs']['text'] == 'new'
        finally:
            if lost is not None and database.get_task(lost)['state'] == 'running':
                database.finish_task(lost, 'failed', database.clock(), error={'code': 'test_cleanup'}, worker_token='uncertain-worker')
            await asyncio.gather(first.close(), resumed.close())
    asyncio.run(scenario())


def test_old_worker_load_failure_retains_tasks_and_current_config_stays_available(backend, tmp_path, monkeypatch):
    database, _ = backend
    settings = options(backend, tmp_path)
    first = Server(name='model', deploy=recorded_deploy, config=config('old'), **settings)
    resumed = Server(name='model', deploy=recorded_deploy, config=config('old'), **settings)
    async def scenario():
        original = Worker.start
        failed = asyncio.Event()
        async def fail_old(worker):
            if worker.config.inputs['text'].default == 'old':
                failed.set()
                raise RuntimeError('old model temporarily unavailable')
            await original(worker)
        try:
            old_key, identities = await leave_queued_old_work(first, config('new'), [{}])
            monkeypatch.setattr(Worker, 'start', fail_old)
            await resumed.start()
            await asyncio.wait_for(failed.wait(), 5)
            await until(lambda: not resumed.retiring_entries.get('model'))
            assert database.get_task(identities[0])['state'] == 'pending'
            assert database.queue_usage(old_key)['inflight'] == 1
            assert (await resumed.deploy('model', {}))['outputs']['text'] == 'new'
            monkeypatch.setattr(Worker, 'start', original)
            await until(lambda: database.get_task(identities[0])['state'] == 'succeeded')
            assert database.get_task(identities[0])['outputs']['text'] == 'old'
        finally:
            monkeypatch.setattr(Worker, 'start', original)
            await asyncio.gather(first.close(), resumed.close())
    asyncio.run(scenario())


def test_recovery_snapshot_retired_by_another_replica_is_a_successful_noop(backend, tmp_path, monkeypatch):
    database, _ = backend
    settings = options(backend, tmp_path)
    first = Server(name='model', deploy=recorded_deploy, config=config('old'), **settings)
    resumed = Server(name='model', deploy=recorded_deploy, config=config('old'), **settings)
    async def scenario():
        raced = asyncio.Event()
        try:
            old_key, identities = await leave_queued_old_work(first, config('new'), [{}])
            original = resumed.shared_runtime.queue
            def finish_elsewhere(name, entry, **kwargs):
                if entry.get('shared_draining'):
                    assert database.claim_task(identities[0], old_key, 'other-replica', 30)
                    database.finish_task(identities[0], 'succeeded', database.clock(), {'text': 'old'}, worker_token='other-replica')
                    database.shared.discard(entry['shared_key'], old_key)
                    raced.set()
                return original(name, entry, **kwargs)
            monkeypatch.setattr(resumed.shared_runtime, 'queue', finish_elsewhere)
            await resumed.start()
            await asyncio.wait_for(raced.wait(), 5)
            await until(lambda: 'model' not in resumed.entry_controls and not resumed.retiring_entries.get('model'))
            assert resumed.entries['model']['error'] is None
            assert (await resumed.deploy('model', {}))['outputs']['text'] == 'new'
        finally:
            await asyncio.gather(first.close(), resumed.close())
    asyncio.run(scenario())


def test_completed_old_result_replays_after_restart_without_loading_or_executing_model_again(backend, tmp_path, monkeypatch):
    from sqlalchemy.exc import OperationalError
    database, _ = backend
    settings = options(backend, tmp_path)
    servers = [Server(name='model', deploy=recorded_deploy, config=config('old'), **settings) for _ in range(3)]
    first, failing, recovered = servers
    marker = tmp_path / 'calls'
    async def scenario():
        try:
            old_key, identities = await leave_queued_old_work(first, config('new'), [{'marker': str(marker)}])
            original_finish = failing.db.finish_task
            def unavailable(identity, *args, **kwargs):
                if identity == identities[0]:
                    raise OperationalError('finish old task', {}, RuntimeError('injected database outage'))
                return original_finish(identity, *args, **kwargs)
            monkeypatch.setattr(failing.db, 'finish_task', unavailable)
            await failing.start()
            directory = failing.data_dir / 'queue-results' / old_key
            await until(lambda: bool(list(directory.glob('*.json'))))
            await until(lambda: not failing.retiring_entries.get('model'))
            assert database.get_task(identities[0])['state'] == 'running'
            await failing.close()
            original_start = Worker.start
            async def forbid_old(worker):
                assert worker.config.inputs['text'].default != 'old', 'Completed result must not reload its model'
                await original_start(worker)
            monkeypatch.setattr(Worker, 'start', forbid_old)
            await recovered.start()
            await until(lambda: database.get_task(identities[0])['state'] == 'succeeded')
            await until(lambda: not list(directory.glob('*.json')))
            assert marker.read_text().splitlines() == ['old']
            assert database.get_task(identities[0])['outputs']['text'] == 'old'
            assert database.queue_usage(old_key)['inflight'] == 0
            assert (await recovered.deploy('model', {}))['outputs']['text'] == 'new'
        finally:
            await asyncio.gather(*(server.close() for server in servers))
    asyncio.run(scenario())
