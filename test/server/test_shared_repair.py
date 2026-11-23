"""Repair confirmed initial failures without replacing established shared entries."""
import asyncio
import copy
import queue
from types import SimpleNamespace
import uuid

import httpx
import pytest
from sqlalchemy import inspect, text, update

from nndeploy.server import Server
from nndeploy.server.db import DB
from nndeploy.server.errors import ServerError
from nndeploy.server.worker import Worker
from test_db import db
from test_redis_queue import backend, redis_service
from test_shared_configs import initial, ready
from test_shared_reload import config, replicas
from test_shared_server import shared_deploy, until
from test_server_api import text_graph
from test_control import configuration


def options(backend, tmp_path):
    database, redis = backend
    return dict(resources=tmp_path / 'shared', data_dir=tmp_path / 'shared', database_url=database.url.render_as_string(hide_password=False), redis_config=redis)


@pytest.mark.parametrize('restart', [False, True])
def test_failed_workflow_can_be_corrected_in_place_or_on_new_server(backend, tmp_path, restart):
    database, _ = backend
    settings = options(backend, tmp_path)
    broken = text_graph()
    broken['node_repository_'][0]['key_'] = 'missing.Node'
    first = Server(name='echo', workflow=broken, config=configuration('old'), **settings)
    async def scenario():
        replacement = restored = None
        try:
            with pytest.raises(Exception):
                await first.start_entry('echo')
            state = database.shared.get(first.entries['echo']['shared_key'])
            assert state['load_failed'] and not state['initialized']
            if restart:
                await first.close()
                replacement = Server(name='echo', workflow=text_graph(), config=configuration('corrected'), **settings)
                await replacement.start()
            else:
                replacement = first
                await replacement.reload_entry('echo', workflow=text_graph(), config=configuration('corrected'))
            assert (await replacement.deploy('echo', {}))['outputs']['text'] == 'corrected'
            state = database.shared.get(replacement.entries['echo']['shared_key'])
            assert state['initialized'] and not state['load_failed'] and state['pending_queue'] is None
            await replacement.close()
            restored = Server(**settings)
            await restored.start()
            assert (await restored.deploy('echo', {}))['outputs']['text'] == 'corrected'
        finally:
            await first.close()
            if replacement is not None:
                await replacement.close()
            if restored is not None:
                await restored.close()
    asyncio.run(scenario())


def test_partial_worker_initialization_failure_allows_repair_only_after_all_workers_close(backend, tmp_path, monkeypatch):
    database, _ = backend
    bad = config('old')
    bad['execution']['workers'] = 2
    first = Server(name='model', deploy=shared_deploy, config=bad, **options(backend, tmp_path))
    second = Server(name='model', deploy=shared_deploy, config=config('new'), **options(backend, tmp_path))
    async def scenario():
        original = Worker.start
        attempts = []
        async def fail_second(worker):
            await original(worker)
            if worker.config.inputs['text'].default == 'old':
                attempts.append(worker)
                if len(attempts) == 2:
                    raise RuntimeError('second worker failed')
        monkeypatch.setattr(Worker, 'start', fail_second)
        try:
            with pytest.raises(RuntimeError, match='second worker'):
                await first.start_entry('model')
            state = database.shared.get(first.entries['model']['shared_key'])
            assert state['load_failed'] and not state['initialized']
            assert not first.entries['model']['workers'] and not first.entries['model']['shared_holders']
            await second.start()
            assert (await second.deploy('model', {}))['outputs']['text'] == 'new'
        finally:
            await asyncio.gather(first.close(), second.close())
    asyncio.run(scenario())


def test_initial_failure_is_not_inferred_from_missing_or_closed_workers(db):
    entry, old, new = initial(db)
    key = entry['entry_key']
    holder = ready(db, entry, old, None)
    db.shared.initialization_failed(key, old)
    assert not db.shared.get(key)['load_failed']
    with pytest.raises(ServerError):
        db.shared.register(entry['group_key'], 'model', new, 'different', {})
    db.shared.confirm_ready(key, old, [holder])
    db.release_files(holder)
    db.shared.initialization_failed(key, old)
    state = db.shared.get(key)
    assert state['initialized'] and not state['load_failed']
    with pytest.raises(ServerError):
        db.shared.register(entry['group_key'], 'model', new, 'different', {})


def test_repair_and_original_configuration_retry_cannot_both_initialize(db):
    entry, old, new = initial(db)
    key = entry['entry_key']
    db.shared.initialization_failed(key, old)
    state = db.shared.get(key)
    holder = ready(db, entry, old, None)
    with pytest.raises(ServerError) as live:
        db.shared.begin_reload(key, state['control_token'], new, 'new', {}, write_token=uuid.uuid4().hex,
                               repair=True, expected_queue_token=state['queue_control_token'])
    assert live.value.code == 'shared_control_changed'
    db.release_files(holder)
    db.shared.initialization_failed(key, old)
    token = uuid.uuid4().hex
    db.shared.begin_reload(key, state['control_token'], new, 'new', {}, write_token=token,
                           repair=True, expected_queue_token=state['queue_control_token'])
    with pytest.raises(ServerError) as retry:
        db.shared.hold_worker(key, old, uuid.uuid4().hex)
    assert retry.value.code == 'shared_control_changed'
    assert db.shared.get(key)['pending_queue'] == new


def test_stop_before_preparation_is_not_overwritten_by_reload(backend, tmp_path, monkeypatch):
    database, _ = backend
    first, second = replicas(backend, tmp_path)
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        reloading = None
        try:
            await first.start()
            await second.start()
            original = first.shared_runtime.snapshot
            async def delayed(entry):
                if entry['config'].inputs['text'].default == 'new':
                    entered.set()
                    await release.wait()
                return await original(entry)
            monkeypatch.setattr(first.shared_runtime, 'snapshot', delayed)
            reloading = asyncio.create_task(first.reload_entry('model', config=config('new')))
            await asyncio.wait_for(entered.wait(), 5)
            await second.stop_entry('model')
            release.set()
            with pytest.raises(ServerError) as stale:
                await reloading
            assert stale.value.code == 'shared_control_changed'
            state = database.shared.get(first.entries['model']['shared_key'])
            assert state['pending_queue'] is None and not state['accepting']
            await until(lambda: first.describe('model')['status'] == 'stopped')
        finally:
            release.set()
            if reloading is not None:
                await asyncio.gather(reloading, return_exceptions=True)
            await asyncio.gather(first.close(), second.close())
    asyncio.run(scenario())


def test_failed_initial_entry_stays_stopped_until_explicit_repair_start(backend, tmp_path):
    database, _ = backend
    settings = options(backend, tmp_path)
    broken = text_graph()
    broken['node_repository_'][0]['key_'] = 'missing.Node'
    first = Server(name='echo', workflow=broken, config=configuration('old'), **settings)
    second = Server(name='echo', workflow=text_graph(), config=configuration('fixed'), **settings)
    async def scenario():
        try:
            with pytest.raises(Exception):
                await first.start_entry('echo')
            await first.stop_entry('echo')
            with pytest.raises(ServerError) as stopped:
                await second.start_entry('echo', resume=False)
            assert stopped.value.code == 'shared_control_changed'
            assert not database.shared.get(first.entries['echo']['shared_key'])['accepting']
            await second.start_entry('echo')
            assert (await second.deploy('echo', {}))['outputs']['text'] == 'fixed'
        finally:
            await asyncio.gather(first.close(), second.close())
    asyncio.run(scenario())


def test_old_shared_schema_is_conservatively_established_on_upgrade(db):
    entry, old, new = initial(db)
    with db.engine.begin() as connection:
        connection.execute(text('ALTER TABLE server_shared_configs DROP COLUMN initialized'))
        connection.execute(text('ALTER TABLE server_shared_configs DROP COLUMN load_failed'))
        connection.execute(text("UPDATE alembic_version SET version_num = '0016_shared_configs'"))
    reopened = DB(db.url).init_schema()
    try:
        state = reopened.shared.get(entry['entry_key'])
        assert state['initialized'] and not state['load_failed']
        reopened.shared.initialization_failed(entry['entry_key'], old)
        with pytest.raises(ServerError):
            reopened.shared.register(entry['group_key'], 'model', new, 'different', {})
        assert {'initialized', 'load_failed'} <= {column['name'] for column in inspect(reopened.engine).get_columns('server_shared_configs')}
    finally:
        reopened.close()


@pytest.mark.parametrize('restart_legacy', [False, True])
@pytest.mark.parametrize('follow_during_attach', [False, True])
def test_editor_repairs_failed_first_start_without_losing_entry_identity(backend, tmp_path, monkeypatch, restart_legacy, follow_during_attach):
    from nndeploy.server.template import WorkflowTemplateManager
    monkeypatch.setattr(WorkflowTemplateManager, 'init_templates', lambda: tmp_path)
    database, redis = backend
    def create():
        return Server(SimpleNamespace(resources=str(tmp_path), debug=True), queue.Queue(), queue.Queue(), queue.Queue(),
            admin_token='editor-test', redis_config=redis, database_url=database.url.render_as_string(hide_password=False))
    server = create()
    async def scenario():
        nonlocal server
        release = asyncio.Event()
        try:
            await server.start()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app, raise_app_exceptions=False), base_url='http://test') as client:
                headers = {'Authorization': 'Bearer editor-test'}
                broken = copy.deepcopy(text_graph())
                broken['node_repository_'][1]['key_'] = 'missing.Node'
                failed = await client.post('/api/workflow/server/start', json={'id': '', 'businessContent': broken}, headers=headers)
                assert failed.status_code == 500, failed.text
                name = next(iter(server.entries))
                identity = (await client.get('/api/workflows', headers=headers)).json()['result'][0]['id']
                if restart_legacy:
                    from nndeploy.server.shared import configs
                    # Migration treats legacy configurations as established.
                    # An explicit editor Update must still repair one without
                    # first reloading its broken graph or weakening boot checks.
                    with database.engine.begin() as connection:
                        connection.execute(update(configs).where(configs.c.queue_key == server.entries[name]['queue'].key)
                                           .values(initialized=True, load_failed=False))
                    await server.close()
                    server = create()
                    await server.start()
                    assert not server.entries
            if follow_during_attach:
                from nndeploy.server.shared_runtime import SharedRuntime
                original_attach, original_start = SharedRuntime.attach_workflow, Worker.start
                entered = asyncio.Event()
                async def blocked_old_start(worker, *args, **kwargs):
                    workflow = worker.executor.source.get('workflow') or {}
                    if any(node.get('key_') == 'missing.Node' for node in workflow.get('node_repository_', [])):
                        entered.set()
                        await release.wait()
                    return await original_start(worker, *args, **kwargs)
                async def competing_attach(runtime, name):
                    result = await original_attach(runtime, name)
                    if result:
                        # Run the real background follower exactly while the
                        # editor has attached the old entry but not repaired it.
                        follower = runtime.server.own(runtime.server._follow_shared_control(name, runtime.server.entries[name], True))
                        observed = asyncio.create_task(entered.wait())
                        try:
                            await asyncio.wait_for(asyncio.wait([follower, observed], return_when=asyncio.FIRST_COMPLETED), 5)
                        finally:
                            observed.cancel()
                            await asyncio.gather(observed, return_exceptions=True)
                    return result
                monkeypatch.setattr(SharedRuntime, 'attach_workflow', competing_attach)
                monkeypatch.setattr(Worker, 'start', blocked_old_start)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app, raise_app_exceptions=False), base_url='http://test') as client:
                repaired = await client.post('/api/workflow/server/start', json={'id': identity, 'businessContent': text_graph()}, headers=headers)
                assert repaired.status_code == 200, repaired.text
                assert repaired.json()['name'] == name
                result = await client.post(repaired.json()['deploy_path'], json={'inputs': {'source': 'corrected'}})
                assert result.status_code == 200 and result.json()['outputs']['sink'] == 'corrected', result.text
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())


def test_editor_does_not_forget_workers_whose_failed_initialization_cleanup_is_unconfirmed(backend, tmp_path, monkeypatch):
    from nndeploy.server.template import WorkflowTemplateManager
    monkeypatch.setattr(WorkflowTemplateManager, 'init_templates', lambda: tmp_path)
    database, redis = backend
    server = Server(SimpleNamespace(resources=str(tmp_path), debug=True), queue.Queue(), queue.Queue(), queue.Queue(),
        admin_token='editor-test', redis_config=redis, database_url=database.url.render_as_string(hide_password=False))
    original_start, original_close = Worker.start, Worker.close
    async def fail_start(worker):
        await original_start(worker)
        raise RuntimeError('injected startup failure')
    async def fail_close(worker):
        raise RuntimeError('physical worker exit is not confirmed')
    async def scenario():
        try:
            await server.start()
            monkeypatch.setattr(Worker, 'start', fail_start)
            monkeypatch.setattr(Worker, 'close', fail_close)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app, raise_app_exceptions=False), base_url='http://test') as client:
                headers = {'Authorization': 'Bearer editor-test'}
                failed = await client.post('/api/workflow/server/start', json={'id': '', 'businessContent': text_graph()}, headers=headers)
                assert failed.status_code == 500
                name = next(iter(server.entries))
                original_entry = server.entries[name]
                assert original_entry['workers'] and original_entry['shared_holders']
                identity = (await client.get('/api/workflows', headers=headers)).json()['result'][0]['id']
                retry = await client.post('/api/workflow/server/start', json={
                    'id': identity, 'businessContent': text_graph()}, headers=headers)
                assert retry.status_code == 409 and retry.json()['error']['code'] == 'entry_busy', retry.text
                assert server.entries[name] is original_entry and original_entry['workers']
                assert not database.shared.get(original_entry['shared_key'])['load_failed']
        finally:
            monkeypatch.setattr(Worker, 'start', original_start)
            monkeypatch.setattr(Worker, 'close', original_close)
            await server.close()
    asyncio.run(scenario())
