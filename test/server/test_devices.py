"""Resident resource budgets against actual Workers and all supported SQL DBs."""
import asyncio
import copy
import multiprocessing
from pathlib import Path
import sys
import threading
import time
import uuid

import httpx
import pytest
from pydantic import ValidationError
from sqlalchemy import select, text, update

from nndeploy.server import Server
from nndeploy.server.config import ExecutionConfig, suggest_config
from nndeploy.server.control_guard import ControlGuard
from nndeploy.server.db import DB
from nndeploy.server.devices import DeviceLimits, scopes, claims
from nndeploy.server.errors import ServerError
from test_db import db
from test_shared_server import until, shared_deploy
from test_shared_reload import config as shared_config
from test_redis_queue import backend, redis_service
from test_server_api import text_graph


def budget(workers=2, memory=100, *, host=None):
    value = {'devices': {'cpu': {'max_workers': workers, 'memory_bytes': memory}}}
    if host is not None:
        value.update(scope='host', host_id=host)
        value['devices']['cpu']['identity'] = 'cpu'
    return value


def configuration(memory=40, workers=1, isolation='thread'):
    value = shared_config('old', isolation)
    value['execution'].update(devices=['cpu'], memory_bytes={'cpu': memory}, workers=workers)
    return value


def make(database, directory, settings=None):
    return Server(resources=directory, data_dir=directory, device_limits=settings or budget(), admin_token='test',
                  database_url=database.url.render_as_string(hide_password=False))


def used(server, device='cpu'):
    return server.devices.snapshot()[device]['used']


def test_device_configuration_is_explicit_and_does_not_change_old_fingerprints():
    assert 'devices' not in ExecutionConfig().model_dump()
    assert 'memory_bytes' not in ExecutionConfig().model_dump()
    for value in [dict(devices=['cpu', 'cpu']), dict(devices=['']), dict(memory_bytes={'cpu': 1}),
                  dict(devices=['cpu'], memory_bytes={'cpu': -1}),
                  dict(devices=['cpu'], workers=2, memory_bytes={'cpu': 2**63 - 1})]:
        with pytest.raises(ValidationError):
            ExecutionConfig.model_validate(value)
    for value in [dict(scope='host', devices={'cpu': {'max_workers': 1}}),
                  dict(scope='host', host_id='host', devices={'cpu': {'max_workers': 1}}),
                  dict(devices={'cpu': {'max_workers': 0}}),
                  dict(devices={'a': {'identity': 'same', 'max_workers': 1}, 'b': {'identity': 'same', 'max_workers': 1}})]:
        with pytest.raises(ValidationError):
            DeviceLimits.model_validate(value)
    from nndeploy.server.app import cli
    assert cli(['--api-only', '--workflow', 'graph.json', '--device-limits', 'limits.json']).device_limits == 'limits.json'


def test_model_capacity_is_reserved_before_factory_and_failed_load_is_isolated(db, tmp_path):
    server = make(db, tmp_path, budget(workers=2, memory=100))
    calls = []
    def handler():
        calls.append('factory')
        return lambda inputs: inputs
    server.add(name='first', deploy=shared_deploy, config=configuration(60))
    server.add(name='second', handler_factory=handler, config=configuration(50))
    async def scenario():
        try:
            await server.start()
            assert server.describe('first')['status'] == 'ready'
            assert server.describe('second')['status'] == 'failed' and not calls
            assert used(server) == {'workers': 1, 'memory_bytes': 60}
            assert (await server.deploy('first', {}))['outputs']['text'] == 'old'
            await server.stop_entry('first')
            await server.start_entry('second')
            assert calls == ['factory'] and used(server)['memory_bytes'] == 50
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                assert (await client.get('/api/server?include_resources=true')).status_code == 401
                response = await client.get('/api/server?include_resources=true', headers={'Authorization': 'Bearer test'})
                assert response.json()['devices']['cpu']['used']['workers'] == 1
        finally:
            await server.close()
    asyncio.run(scenario())
    with db.engine.connect() as connection:
        assert not connection.execute(select(claims).where(claims.c.scope == server.devices.scope)).first()


def test_failed_reload_preserves_old_graph_and_budget(db, tmp_path):
    server = make(db, tmp_path, budget(workers=1, memory=100))
    cfg = suggest_config(text_graph())
    cfg['execution'] = configuration(40)['execution']
    server.add(name='model', workflow=text_graph(), config=cfg)
    async def scenario():
        try:
            await server.start()
            original = server.entries['model']['workers'][0]
            with pytest.raises(ServerError) as error:
                await server.reload_entry('model')
            assert error.value.code == 'device_capacity_exceeded'
            assert server.entries['model']['workers'][0] is original
            assert server.describe('model')['status'] == 'ready'
            assert (await server.deploy('model', {'source': 'still ready'}))['outputs']['sink'] == 'still ready'
            assert used(server) == {'workers': 1, 'memory_bytes': 40}
        finally:
            await server.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('isolation', ['thread', 'process'])
def test_reload_keeps_old_and_new_memory_until_running_request_finishes(db, tmp_path, isolation):
    server = make(db, tmp_path, budget(workers=3, memory=100))
    server.add(name='model', deploy=shared_deploy, config=configuration(40, isolation=isolation))
    marker, release = tmp_path / 'entered', tmp_path / 'release'
    async def scenario():
        reloading = None
        try:
            await server.start()
            identity, future = await server.submit('model', {'marker': str(marker), 'release': str(release)})
            await until(marker.exists)
            reloading = asyncio.create_task(server.reload_entry('model'))
            await until(lambda: used(server)['workers'] == 2)
            assert used(server)['memory_bytes'] == 80 and not future.done()
            server.add(name='other', deploy=shared_deploy, config=configuration(30))
            with pytest.raises(ServerError) as error:
                await server.start_entry('other')
            assert error.value.code == 'device_capacity_exceeded'
            release.touch()
            assert (await future)['outputs']['text'] == 'old'
            await reloading
            assert used(server)['memory_bytes'] == 40
            await server.start_entry('other')
            assert used(server)['memory_bytes'] == 70
        finally:
            release.touch()
            if reloading is not None:
                await asyncio.gather(reloading, return_exceptions=True)
            await server.close()
    asyncio.run(scenario())


def test_cross_server_host_budget_is_atomic_and_default_workspaces_are_independent(db, tmp_path):
    host = uuid.uuid4().hex
    servers = [make(db, tmp_path / str(i), budget(1, 40, host=host)) for i in range(2)]
    for i, server in enumerate(servers):
        server.add(name=f'model{i}', deploy=shared_deploy, config=configuration())
    async def scenario():
        try:
            results = await asyncio.gather(*(server.start() for server in servers), return_exceptions=True)
            assert sum(isinstance(value, ServerError) and value.code == 'device_capacity_exceeded' for value in results) == 1
            assert [used(server) for server in servers] == [{'workers': 1, 'memory_bytes': 40}] * 2
            ready = next(i for i, server in enumerate(servers) if server.describe(f'model{i}')['status'] == 'ready')
            await servers[ready].stop_entry(f'model{ready}')
            await servers[1-ready].start_entry(f'model{1-ready}')
            assert used(servers[0])['workers'] == 1
        finally:
            await asyncio.gather(*(server.close() for server in servers))
        independent = [make(db, tmp_path / f'independent-{i}', budget(1, 40)) for i in range(2)]
        try:
            for server in independent:
                server.add(name='model', deploy=shared_deploy, config=configuration())
            await asyncio.gather(*(server.start() for server in independent))
            assert all(used(server)['workers'] == 1 for server in independent)
        finally:
            await asyncio.gather(*(server.close() for server in independent))
    asyncio.run(scenario())


def test_multidevice_reservation_is_atomic_and_stable_identity_unifies_aliases(db, tmp_path):
    host = uuid.uuid4().hex
    settings = budget(2, 100, host=host)
    settings['devices']['accelerator'] = {'identity': 'physical-test-device', 'max_workers': 1, 'memory_bytes': 10}
    first = make(db, tmp_path / 'first', settings)
    second_settings = copy.deepcopy(settings)
    second_settings['devices']['alias'] = second_settings['devices'].pop('accelerator')
    second = make(db, tmp_path / 'second', second_settings)
    first.add(name='cpu', deploy=shared_deploy, config=configuration(20))
    both = configuration(20)
    both['execution'].update(devices=['cpu', 'accelerator'], memory_bytes={'cpu': 20, 'accelerator': 11})
    first.add(name='too-large', deploy=shared_deploy, config=both)
    async def scenario():
        try:
            await first.start()
            assert used(first)['workers'] == 1 and used(first, 'accelerator')['workers'] == 0
            both['execution']['memory_bytes']['accelerator'] = 10
            first.entries.pop('too-large')
            first.add(name='both', deploy=shared_deploy, config=both)
            await first.start_entry('both')
            alias = configuration(10)
            alias['execution'].update(devices=['alias'], memory_bytes={'alias': 1})
            second.add(name='alias', deploy=shared_deploy, config=alias)
            with pytest.raises(ServerError) as error:
                await second.start()
            assert error.value.code == 'device_capacity_exceeded'
            assert used(second, 'alias') == {'workers': 1, 'memory_bytes': 10}
        finally:
            await asyncio.gather(first.close(), second.close())
    asyncio.run(scenario())


@pytest.mark.parametrize('stage', ['reserve', 'release'])
def test_commit_ack_loss_retries_without_duplicate_resident_reservation(db, tmp_path, monkeypatch, stage):
    server = make(db, tmp_path)
    server.add(name='model', deploy=shared_deploy, config=configuration())
    original = getattr(server.devices, '_' + stage)
    failed = []
    def uncertain(*args):
        value = original(*args)
        if not failed:
            failed.append(True)
            raise OSError('Commit acknowledgement lost')
        return value
    monkeypatch.setattr(server.devices, '_' + stage, uncertain)
    async def scenario():
        try:
            await server.start()
            assert used(server)['workers'] == 1
            await server.stop_entry('model')
            assert used(server)['workers'] == 0 and failed
        finally:
            await server.close()
    asyncio.run(scenario())


def test_initializer_failure_releases_reserved_budget_and_worker_count_is_multiplied(db, tmp_path):
    calls = []
    class Handler:
        def init(self):
            calls.append(True)
            if len(calls) == 2:
                raise ValueError('Second model failed')
        def deploy(self, inputs):
            return inputs
    server = make(db, tmp_path)
    server.add(name='model', handler_factory=Handler, config=configuration(workers=2))
    async def scenario():
        try:
            with pytest.raises(Exception):
                await server.start()
            assert calls == [True, True] and used(server)['workers'] == 0
            server.entries.pop('model')
            server.add(name='too-many', handler_factory=Handler, config=configuration(1, workers=3))
            with pytest.raises(ServerError):
                await server.start_entry('too-many')
            assert len(calls) == 2 and used(server)['workers'] == 0
        finally:
            await server.close()
    asyncio.run(scenario())


def test_policy_mismatch_and_missing_declarations_fail_before_model_loading(db, tmp_path):
    host = uuid.uuid4().hex
    first = make(db, tmp_path / 'first', budget(1, 100, host=host))
    second = make(db, tmp_path / 'second', budget(2, 100, host=host))
    first.add(name='model', deploy=shared_deploy, config=configuration())
    second.add(name='model', deploy=shared_deploy, config=configuration())
    async def scenario():
        try:
            await first.start()
            with pytest.raises(ServerError) as error:
                await second.start()
            assert error.value.code == 'device_policy_conflict'
            for name, cfg, code in [
                ('missing', shared_config('old'), 'device_declaration_required'),
                ('memory', {**shared_config('old'), 'execution': {'devices': ['cpu']}}, 'memory_budget_required'),
                ('unknown', {**shared_config('old'), 'execution': {'devices': ['missing']}}, 'device_not_configured')]:
                second.add(name=name, deploy=shared_deploy, config=cfg)
                with pytest.raises(ServerError) as error:
                    await second.start_entry(name)
                assert error.value.code == code
        finally:
            await asyncio.gather(first.close(), second.close())
    asyncio.run(scenario())


def test_shared_reload_reserves_every_replica_and_releases_only_its_own_claim(backend, tmp_path):
    database, redis = backend
    options = dict(resources=tmp_path, data_dir=tmp_path, database_url=database.url.render_as_string(hide_password=False), redis_config=redis,
                   device_limits=budget(4, 160))
    servers = [Server(name='model', deploy=shared_deploy, config=configuration(), **options) for _ in range(2)]
    async def scenario():
        try:
            await asyncio.gather(*(server.start() for server in servers))
            assert used(servers[0])['workers'] == 2
            await servers[0].reload_entry('model')
            await until(lambda: servers[0].entries['model']['queue'].key == servers[1].entries['model']['queue'].key)
            await until(lambda: used(servers[0])['workers'] == 2)
            await servers[0].close()
            assert used(servers[1])['workers'] == 1
            assert (await servers[1].deploy('model', {}))['outputs']['text'] == 'old'
        finally:
            await asyncio.gather(*(server.close() for server in servers))
    asyncio.run(scenario())


def test_upgrade_preserves_existing_workflow_and_has_no_invented_allocations(db):
    identity = uuid.uuid4().hex
    db.create_task(identity, 'before-budgets', time.time(), {}, {})
    with db.engine.begin() as connection:
        claims.drop(connection)
        scopes.drop(connection)
        connection.execute(text("UPDATE alembic_version SET version_num = '0020_control_requests'"))
    upgraded = DB(db.url).init_schema()
    try:
        assert upgraded.get_task(identity)['state'] == 'pending'
        with upgraded.engine.connect() as connection:
            assert not connection.execute(select(claims)).first()
    finally:
        upgraded.close()


def _running_budgeted_server(options, isolation, marker, release, accepted, warmup=False):
    async def scenario():
        cfg = configuration(isolation=isolation)
        cfg['execution']['temporary_memory_bytes'] = {'cpu': 50}
        if warmup:
            cfg['execution']['warmup'] = [{'inputs': {'marker': marker, 'release': release}}]
            Path(accepted).write_text('initializing')
        server = Server(name='model', deploy=shared_deploy, config=cfg, **options)
        await server.start()
        if not warmup:
            identity, future = await server.submit('model', {'marker': marker, 'release': release})
            Path(accepted).write_text(identity)
            await future
        await asyncio.Event().wait()
    asyncio.run(scenario())


@pytest.mark.skipif(sys.platform not in ('linux', 'darwin'), reason='Physical recovery requires kernel boot UUID')
@pytest.mark.parametrize('isolation', ['thread', 'process'])
@pytest.mark.parametrize('warmup', [False, True])
def test_real_owner_death_preserves_budget_until_all_compute_exits(db, tmp_path, isolation, warmup):
    from test_shared_crash import _exited
    host = uuid.uuid4().hex
    settings = budget(1, 90, host=host)
    settings['devices']['cpu']['max_concurrency'] = 1
    options = dict(resources=tmp_path / 'original', data_dir=tmp_path / 'original', device_limits=settings,
                   database_url=db.url.render_as_string(hide_password=False))
    marker, release, accepted = [tmp_path / name for name in ('running', 'release', 'accepted')]
    owner = multiprocessing.get_context('spawn').Process(target=_running_budgeted_server,
        args=(options, isolation, str(marker), str(release), str(accepted), warmup))
    replacement = make(db, tmp_path / 'replacement', settings)
    replacement.add(name='replacement', deploy=shared_deploy, config=configuration())
    async def scenario():
        child = None
        try:
            owner.start()
            await until(lambda: accepted.exists() and marker.exists())
            child = int(marker.read_text()) if isolation == 'process' else None
            assert used(replacement)['workers'] == 1
            assert replacement.devices.snapshot()['cpu']['execution']['running'] == 1
            assert replacement.devices.snapshot()['cpu']['memory']['temporary_bytes'] == 50
            owner.kill()
            await asyncio.to_thread(owner.join, 5)
            if child is not None:
                assert not _exited(child)
                # No lease expiry or missing controller frees the child's model.
                for _ in range(3):
                    replacement.devices.recover()
                assert used(replacement)['workers'] == 1
                assert replacement.devices.snapshot()['cpu']['execution']['running'] == 1
                assert replacement.devices.snapshot()['cpu']['memory']['temporary_bytes'] == 50
                with pytest.raises(ServerError) as error:
                    await replacement.start_entry('replacement')
                assert error.value.code == 'device_capacity_exceeded'
                release.touch()
                await until(lambda: _exited(child))
            for _ in range(3):
                replacement.devices.recover()
            assert used(replacement)['workers'] == 0
            assert replacement.devices.snapshot()['cpu']['execution'] == {'running': 0, 'waiting': 0}
            assert replacement.devices.snapshot()['cpu']['memory']['temporary_bytes'] == 0
            await replacement.start_entry('replacement')
            assert (await replacement.deploy('replacement', {}))['outputs']['text'] == 'old'
            assert used(replacement)['workers'] == 1
        finally:
            release.touch()
            if owner.pid is not None:
                if owner.is_alive():
                    owner.kill()
                await asyncio.to_thread(owner.join, 5)
            if child is not None:
                await until(lambda: _exited(child))
            await replacement.close()
    asyncio.run(scenario())


@pytest.mark.skipif(sys.platform not in ('linux', 'darwin'), reason='Physical recovery requires kernel boot UUID')
def test_initializer_and_failed_shutdown_keep_resource_claim(db, tmp_path):
    entered, release = threading.Event(), threading.Event()
    class Handler:
        def init(self):
            entered.set()
            assert release.wait(10)
        def deploy(self, inputs):
            return inputs
    server = make(db, tmp_path, budget(1, 40))
    server.add(name='model', handler_factory=Handler, config=configuration())
    async def scenario():
        starting = None
        try:
            starting = asyncio.create_task(server.start())
            await until(entered.is_set)
            assert used(server)['workers'] == 1
            assert server.devices.recover() == 0
            assert used(server)['workers'] == 1
            release.set()
            await starting
            worker = server.entries['model']['workers'][0]
            original = worker.close
            async def unconfirmed():
                raise RuntimeError('Native release was not confirmed')
            worker.close = unconfirmed
            with pytest.raises(RuntimeError):
                await server.stop_entry('model')
            for _ in range(3):
                server.devices.recover()
            assert used(server)['workers'] == 1
            worker.close = original
        finally:
            release.set()
            if starting is not None:
                await asyncio.gather(starting, return_exceptions=True)
            # Physical close can be proved here, independent of the cached
            # failed management operation; normal retry semantics are separate.
            for worker in server.entries['model']['workers']:
                await worker.close()
            await server._release_entry_files(server.entries['model'])
            server.entries['model']['workers'] = []
            await server.close()
    asyncio.run(scenario())


@pytest.mark.skipif(sys.platform not in ('linux', 'darwin'), reason='Physical recovery requires kernel boot UUID')
def test_shutdown_database_outage_keeps_claim_until_confirmed_recovery(db, tmp_path, monkeypatch):
    server = make(db, tmp_path, budget(1, 40))
    server.add(name='model', deploy=shared_deploy, config=configuration())
    async def scenario():
        restored = None
        try:
            await server.start()
            def offline(*args):
                raise OSError('Database unavailable after physical worker close')
            monkeypatch.setattr(server.devices, '_release', offline)
            with pytest.raises(ServerError) as error:
                await asyncio.wait_for(server.close(), 5)
            assert error.value.code == 'device_claim_unconfirmed'
            restored = make(db, tmp_path, budget(1, 40))
            assert used(restored)['workers'] == 1
            restored.add(name='model', deploy=shared_deploy, config=configuration())
            await restored.start()
            assert used(restored)['workers'] == 1
            assert (await restored.deploy('model', {}))['outputs']['text'] == 'old'
        finally:
            if restored:
                await restored.close()
    asyncio.run(scenario())


@pytest.mark.skipif(sys.platform not in ('linux', 'darwin'), reason='Physical recovery requires kernel boot UUID')
def test_recovery_failure_is_retryable_and_revoked_guard_rejects_late_children(db, tmp_path, monkeypatch):
    server = make(db, tmp_path, budget(1, 40))
    server.add(name='model', deploy=shared_deploy, config=configuration())
    entry = server.entries['model']
    async def scenario():
        try:
            await server.devices.acquire('model', entry)
            guard = entry['device_guard']
            guard.close()  # No Worker was ever constructed in this initializer.
            with monkeypatch.context() as patch:
                def failed(*args):
                    raise OSError('Commit failed')
                patch.setattr(server.devices, '_release', failed)
                assert server.devices.recover() == 0
            assert used(server)['workers'] == 1
            with pytest.raises(RuntimeError):
                ControlGuard.join(guard.path)
            for _ in range(3):
                server.devices.recover()
            assert used(server)['workers'] == 0
            await server.devices.release(entry)
        finally:
            await server.close()
    asyncio.run(scenario())


@pytest.mark.skipif(sys.platform not in ('linux', 'darwin'), reason='Physical recovery requires kernel boot UUID')
def test_one_failed_recovery_does_not_block_other_models_and_copied_evidence_is_rejected(db, tmp_path, monkeypatch):
    import shutil
    server = make(db, tmp_path, budget(2, 100))
    tokens = ['1' * 32, '2' * 32]
    guards = [ControlGuard.create(server.data_dir, token, token) for token in tokens]
    try:
        for token in tokens:
            server.devices._reserve(token, token, {'cpu': {'workers': 1, 'memory_bytes': 40}})
        for guard in guards:
            guard.close()
        original = server.devices._release
        with monkeypatch.context() as patch:
            def failure(token):
                if token == tokens[0]:
                    raise OSError('First result unavailable')
                return original(token)
            patch.setattr(server.devices, '_release', failure)
            assert server.devices.recover() == 1
        assert used(server)['workers'] == 1
        copied = tmp_path / 'copied'
        (copied / 'control-locks').mkdir(parents=True)
        shutil.copyfile(guards[0].path, copied / 'control-locks' / tokens[0])
        with db.engine.begin() as connection:
            connection.execute(update(claims).where(claims.c.token == tokens[0]).values(directory=str(copied)))
        for _ in range(3):
            assert server.devices.recover() == 0
        assert used(server)['workers'] == 1
        with db.engine.begin() as connection:
            connection.execute(update(claims).where(claims.c.token == tokens[0]).values(directory=str(server.data_dir)))
        for _ in range(3):
            server.devices.recover()
        assert used(server)['workers'] == 0
        with pytest.raises(RuntimeError):
            ControlGuard.join(guards[0].path)
    finally:
        for guard in guards:
            guard.close()
        asyncio.run(server.close())


def test_declared_budgets_require_a_configured_coordinator(db, tmp_path):
    server = Server(name='model', deploy=shared_deploy, config=configuration(), resources=tmp_path, data_dir=tmp_path,
                    database_url=db.url.render_as_string(hide_password=False))
    async def scenario():
        try:
            with pytest.raises(ServerError) as error:
                await server.start()
            assert error.value.code == 'device_limits_required'
        finally:
            await server.close()
    asyncio.run(scenario())


def test_cpu_budget_example_runs_both_endpoints_and_has_reload_headroom(tmp_path):
    import runpy
    example = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'demo/server/device_limits.py'))
    server = example['create_server'](tmp_path, resources=tmp_path)
    async def scenario():
        try:
            await server.start()
            assert (await server.deploy('uppercase', {'text': 'nndeploy'}))['outputs']['text'] == 'NNDEPLOY'
            assert (await server.deploy('lowercase', {'text': 'Server'}))['outputs']['text'] == 'server'
            await server.reload_entry('uppercase')
            assert used(server)['workers'] == 2
        finally:
            await server.close()
    asyncio.run(scenario())


def test_unlimited_memory_policy_still_keeps_sql_aggregate_representable(db, tmp_path):
    server = make(db, tmp_path, budget(2, None))
    server.add(name='first', deploy=shared_deploy, config=configuration(2**63 - 1))
    server.add(name='second', deploy=shared_deploy, config=configuration(1))
    async def scenario():
        try:
            await server.start()
            assert server.describe('first')['status'] == 'ready'
            assert server.describe('second')['status'] == 'failed'
            assert used(server) == {'workers': 1, 'memory_bytes': 2**63 - 1}
        finally:
            await server.close()
    asyncio.run(scenario())
