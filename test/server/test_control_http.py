"""Durable receipts and bounded waits for the existing management routes."""
import asyncio
import json
import multiprocessing
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace
import uuid

import httpx
import pytest
from sqlalchemy import update, select, text

from nndeploy.server import Server
from nndeploy.server.control import receipts, RETENTION_SECONDS
from nndeploy.server.errors import ServerError
from nndeploy.server.db import DB
from test_db import db
from test_redis_queue import backend, redis_service
from test_shared_server import shared_deploy, until
from test_shared_reload import config
from test_server_api import text_graph


AUTH = {'Authorization': 'Bearer management-token'}


def make(database, directory):
    return Server(resources=directory, data_dir=directory, database_url=database.url.render_as_string(hide_password=False), admin_token='management-token')


async def complete(client, url):
    for _ in range(300):
        response = await client.get(url, headers=AUTH)
        if response.status_code != 202:
            return response
        await asyncio.sleep(.02)
    raise AssertionError('Control did not complete')


def test_bounded_start_and_concurrent_key_replay_load_one_model(db, tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = []
    class Handler:
        def init(self):
            calls.append('init')
            entered.set()
            assert release.wait(10)
        def deploy(self, values):
            return values
    server = make(db, tmp_path)
    async def scenario():
        try:
            await server.start()
            server.add(name='model', handler_factory=Handler)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                headers = dict(AUTH, **{'Idempotency-Key': 'same-start'})
                start = time.monotonic()
                values = await asyncio.gather(*(client.post('/api/server/model/start?wait_timeout_seconds=0', headers=headers) for _ in range(8)))
                assert time.monotonic() - start < 3
                assert {value.status_code for value in values} == {202}
                assert len({value.json()['request_id'] for value in values}) == 1
                location = values[0].headers['location']
                assert values[0].json()['status_url'] == location
                assert (await client.get(location)).status_code == 401
                await until(entered.is_set)
                assert calls == ['init']
                release.set()
                result = await complete(client, location)
                assert result.status_code == 200 and result.json()['status'] == 'ready'
                repeated = await client.post('/api/server/model/start', headers=headers)
                assert repeated.content == result.content and calls == ['init']
                assert repeated.headers['idempotency-retention-seconds'] == str(RETENTION_SECONDS)
                await until(lambda: not list((server.data_dir / 'control-locks').iterdir()))
                conflict = await client.post('/api/server/model/stop', headers=headers)
                assert conflict.status_code == 409 and conflict.json()['error']['code'] == 'idempotency_conflict'
                assert (await server.deploy('model', {'value': 1}))['outputs'] == {'value': 1}
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())


def test_disconnect_after_acceptance_does_not_cancel_control_and_restart_replays_result(db, tmp_path):
    server = make(db, tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def start(name):
        calls.append(name)
        entered.set()
        await release.wait()
        return {'name': name, 'status': 'ready', 'original': True}
    server.start_entry = start
    async def scenario():
        restored = None
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                request = asyncio.create_task(client.post('/api/server/model/start', headers=dict(AUTH, **{'Idempotency-Key': 'disconnect'})))
                await entered.wait()
                request.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await request
                release.set()
                response = await client.post('/api/server/model/start', headers=dict(AUTH, **{'Idempotency-Key': 'disconnect'}))
                assert response.status_code == 200 and calls == ['model']
                result, location = response.content, response.headers['location']
            await server.close()
            restored = make(db, tmp_path)
            async def forbidden(name):
                pytest.fail('Completed control was executed after restart')
            restored.start_entry = forbidden
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=restored.app), base_url='http://test') as client:
                response = await client.post('/api/server/model/start', headers=dict(AUTH, **{'Idempotency-Key': 'disconnect'}))
                assert response.content == result and response.status_code == 200
                assert (await client.get(location, headers=AUTH)).content == result
        finally:
            release.set()
            await server.close()
            if restored:
                await restored.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('stage', ['reserve', 'finish'])
def test_control_commit_ack_loss_uses_the_same_receipt_and_does_not_repeat_action(db, tmp_path, monkeypatch, stage):
    server = make(db, tmp_path)
    calls, failed = [], []
    async def action(name):
        calls.append(name)
        return {'name': name, 'status': 'ready'}
    server.start_entry = action
    original = getattr(server.controls, stage)
    def uncertain(*args):
        value = original(*args)
        if not failed:
            failed.append(True)
            raise OSError('Commit acknowledgement was lost')
        return value
    monkeypatch.setattr(server.controls, stage, uncertain)
    async def scenario():
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                response = await client.post('/api/server/model/start', headers=dict(AUTH, **{'Idempotency-Key': 'ack'}))
                assert response.status_code == 200 and calls == ['model'] and failed
                assert (await client.get(response.headers['location'], headers=AUTH)).content == response.content
        finally:
            await server.close()
    asyncio.run(scenario())


def test_failed_result_is_replayed_and_different_reload_body_is_rejected(db, tmp_path):
    server = make(db, tmp_path)
    calls = []
    async def reload(name, **kwargs):
        calls.append(kwargs)
        raise ServerError(422, 'invalid_model', 'Model could not be initialized')
    server.reload_entry = reload
    async def scenario():
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                headers = dict(AUTH, **{'Idempotency-Key': 'reload'})
                body = {'config': {'execution': {'workers': 2}}}
                response = await client.post('/api/server/model/reload', json=body, headers=headers)
                assert response.status_code == 422
                repeated = await client.post('/api/server/model/reload', json=body, headers=headers)
                assert repeated.content == response.content and len(calls) == 1
                conflict = await client.post('/api/server/model/reload', json={'config': {}}, headers=headers)
                assert conflict.status_code == 409 and len(calls) == 1
                invalid = await client.post('/api/server/model/reload', json={'config': '/private/file'}, headers=AUTH)
                assert invalid.status_code == 422
                assert (await client.get('/api/server?request_id=invalid', headers=AUTH)).status_code == 422
                assert (await client.post('/api/server/model/start?wait_timeout_seconds=61', headers=AUTH)).status_code == 422
        finally:
            await server.close()
    asyncio.run(scenario())


def test_receipts_are_scoped_and_expiry_never_deletes_pending_or_unconfirmed_records(db, tmp_path):
    server = make(db, tmp_path)
    controls = server.controls
    ids = ['req_' + uuid.uuid4().hex for _ in range(4)]
    alice = controls.reserve(ids[0], 'alice', 'model', 'start', {}, 'key')
    bob = controls.reserve(ids[1], 'bob', 'model', 'start', {}, 'key')
    other = controls.reserve(ids[2], 'alice', 'other', 'start', {}, 'key')
    pending = controls.reserve(ids[3], 'alice', 'pending', 'start', {}, 'key')
    try:
        assert len({row['identity'] for row in (alice, bob, other, pending)}) == 4
        with pytest.raises(ServerError):
            controls.get(ids[0], 'bob')
        controls.finish(alice, 200, {'done': True})
        with db.engine.begin() as connection:
            connection.execute(update(receipts).where(receipts.c.identity == alice['identity']).values(expires_at=db.clock(connection) - 1))
            connection.execute(update(receipts).where(receipts.c.identity == bob['identity']).values(state='unconfirmed'))
        assert controls.cleanup() == 1
        assert controls.get(ids[1], 'bob')['state'] == 'unconfirmed'
        assert controls.get(ids[3], 'alice')['state'] == 'running'  # no exit evidence
        replacement = controls.reserve('req_' + uuid.uuid4().hex, 'alice', 'model', 'stop', {}, 'key')
        assert replacement['request_id'] != alice['request_id']
    finally:
        asyncio.run(server.close())


@pytest.mark.parametrize('commit_failure', [False, True])
def test_shutdown_joins_control_and_commits_known_results_without_waiting_forever(db, tmp_path, monkeypatch, commit_failure):
    server = make(db, tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    async def start(name):
        entered.set()
        await release.wait()
        return {'name': name, 'status': 'ready'}
    server.start_entry = start
    original = server.controls.finish
    def finish(*args):
        calls.append(True)
        if commit_failure:
            raise OSError('Database unavailable during shutdown')
        return original(*args)
    monkeypatch.setattr(server.controls, 'finish', finish)
    async def scenario():
        stopping = restored = None
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                response = await client.post('/api/server/model/start?wait_timeout_seconds=0', headers=AUTH)
                assert response.status_code == 202
                await entered.wait()
                stopping = asyncio.create_task(server.close())
                await until(lambda: server.shutting_down)
                assert not stopping.done()
                release.set()
                await asyncio.wait_for(asyncio.shield(stopping), 5)
                assert calls == [True]
                restored = make(db, tmp_path)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=restored.app), base_url='http://test') as other:
                    result = await other.get(response.headers['location'], headers=AUTH)
                    # A known response is retained locally before the failed
                    # final DB write, and replayed after restart without Start.
                    assert result.status_code == 200 and result.json()['status'] == 'ready'
        finally:
            release.set()
            await server.close()
            if restored:
                await restored.close()
    asyncio.run(scenario())


def test_upgrade_adds_control_history_without_changing_existing_tasks(db, tmp_path):
    from alembic.config import Config
    from alembic.script import ScriptDirectory
    from nndeploy.server import migrations
    migration_config = Config()
    migration_config.set_main_option('script_location', str(Path(migrations.__file__).parent))
    current_head = ScriptDirectory.from_config(migration_config).get_current_head()
    identity = uuid.uuid4().hex
    db.create_task(identity, 'old-entry', time.time(), {'text': 'preserved'}, {})
    with db.engine.begin() as connection:
        receipts.drop(connection)
        connection.execute(text("UPDATE alembic_version SET version_num = '0019_task_runtimes'"))
    upgraded = DB(db.url).init_schema()
    try:
        record = upgraded.get_task(identity)
        assert record['state'] == 'pending' and record['inputs'] == {'text': 'preserved'}
        with upgraded.engine.connect() as connection:
            assert not connection.execute(select(receipts)).first()
            assert connection.execute(text('SELECT version_num FROM alembic_version')).scalar_one() == current_head
    finally:
        upgraded.close()


def test_shared_replicas_replay_one_reload_even_with_different_workspaces(backend, tmp_path, monkeypatch):
    database, redis = backend
    # Private configuration/journal directories, common managed file storage.
    # Shared queues intentionally reject replicas with unrelated file stores.
    files = tmp_path / 'files'
    files.mkdir()
    for index in range(2):
        directory = tmp_path / str(index)
        directory.mkdir()
        (directory / 'files').symlink_to(files, target_is_directory=True)
    servers = [Server(name='model', deploy=shared_deploy, config=config('old'), resources=tmp_path / str(index), data_dir=tmp_path / str(index),
        database_url=database.url.render_as_string(hide_password=False), redis_config=redis, admin_token='management-token') for index in range(2)]
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    for server in servers:
        original = server.reload_entry
        async def reload(name, _original=original, **kwargs):
            calls.append(name)
            entered.set()
            await release.wait()
            return await _original(name, **kwargs)
        monkeypatch.setattr(server, 'reload_entry', reload)
    async def scenario():
        try:
            for server in servers:
                await server.start()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=servers[0].app), base_url='http://test') as first, \
                       httpx.AsyncClient(transport=httpx.ASGITransport(app=servers[1].app), base_url='http://test') as second:
                headers = dict(AUTH, **{'Idempotency-Key': 'shared-reload'})
                response = await first.post('/api/server/model/reload?wait_timeout_seconds=0', json={'config': config('new')}, headers=headers)
                await entered.wait()
                duplicate = await second.post('/api/server/model/reload?wait_timeout_seconds=0', json={'config': config('new')}, headers=headers)
                assert response.status_code == duplicate.status_code == 202
                assert response.json()['request_id'] == duplicate.json()['request_id'] and calls == ['model']
                release.set()
                completed = await complete(second, response.headers['location'])
                assert completed.status_code == 200 and calls == ['model']
                assert (await first.post('/api/server/model/reload', json={'config': config('new')}, headers=headers)).content == completed.content
        finally:
            release.set()
            await asyncio.gather(*(server.close() for server in servers))
    asyncio.run(scenario())


def test_editor_new_workflow_retries_save_and_start_only_once(db, tmp_path, monkeypatch):
    from nndeploy.server.template import WorkflowTemplateManager
    monkeypatch.setattr(WorkflowTemplateManager, 'init_templates', lambda: tmp_path)
    server = Server(SimpleNamespace(resources=str(tmp_path), debug=True), queue.Queue(), queue.Queue(), queue.Queue(),
        database_url=db.url.render_as_string(hide_password=False), admin_token='management-token')
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []
    original = server.start_workflow
    async def start(body):
        calls.append(body)
        entered.set()
        await release.wait()
        return await original(body)
    monkeypatch.setattr(server, 'start_workflow', start)
    async def scenario():
        try:
            await server.start()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                body = {'id': '', 'businessContent': text_graph()}
                headers = dict(AUTH, **{'Idempotency-Key': 'new-workflow'})
                started = await client.post('/api/workflow/server/start?wait_timeout_seconds=0', json=body, headers=headers)
                await entered.wait()
                repeated = await client.post('/api/workflow/server/start?wait_timeout_seconds=0', json=body, headers=headers)
                assert started.status_code == repeated.status_code == 202
                assert started.json()['request_id'] == repeated.json()['request_id'] and len(calls) == 1
                release.set()
                completed = await complete(client, started.headers['location'])
                assert completed.status_code == 200, completed.text
                info = completed.json()
                repeated = await client.post('/api/workflow/server/start', json=body, headers=headers)
                assert repeated.content == completed.content and len(calls) == 1
                workflows = (await client.get('/api/workflows', headers=AUTH)).json()['result']
                assert [item['id'] for item in workflows] == [info['workflow_id']]
                assert json.loads(server.db.get_workflow_path(info['workflow_id']).read_text()) == body['businessContent']
                result = await client.post(info['deploy_path'], json={'inputs': {'source': 'once'}})
                assert result.status_code == 200 and result.json()['outputs']['sink'] == 'once'
                changed = await client.post('/api/workflow/server/start', json=dict(body, businessContent={}), headers=headers)
                assert changed.status_code == 409 and len(calls) == 1
                # Preserve the editor's existing HTTP errors through receipts.
                missing = dict(body, id='missing-workflow')
                headers['Idempotency-Key'] = 'missing-workflow'
                failed = await client.post('/api/workflow/server/start', json=missing, headers=headers)
                assert failed.status_code == 404
                assert (await client.post('/api/workflow/server/start', json=missing, headers=headers)).content == failed.content
                assert len(calls) == 2
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())


def _interrupted_control(url, directory, marker):
    async def scenario():
        server = Server(resources=directory, data_dir=directory, database_url=url, admin_token='management-token')
        async def action():
            Path(marker).write_text('started')
            await asyncio.Event().wait()
        await server.controls.request('req_' + 'a' * 32, 'management', 'model', 'start', {}, 'interrupted', action, 0)
        await asyncio.Event().wait()
    asyncio.run(scenario())


@pytest.mark.skipif(__import__('sys').platform not in ('linux', 'darwin'), reason='Kernel boot UUID required')
def test_dead_controller_remains_unconfirmed_and_key_never_reexecutes(db, tmp_path):
    marker = tmp_path / 'started'
    directory = tmp_path / 'server'
    process = multiprocessing.get_context('spawn').Process(target=_interrupted_control,
        args=(db.url.render_as_string(hide_password=False), directory, marker))
    async def scenario():
        server = None
        try:
            process.start()
            await until(marker.exists)
            process.kill()
            await asyncio.to_thread(process.join, 5)
            server = make(db, directory)
            async def forbidden(name):
                pytest.fail('An interrupted control was executed again')
            server.start_entry = forbidden
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                response = await client.post('/api/server/model/start', headers=dict(AUTH, **{'Idempotency-Key': 'interrupted'}))
                assert response.status_code == 503 and response.json()['error']['code'] == 'control_unconfirmed'
                assert response.json()['error']['accepted'] is True
                assert (await client.get(response.headers['location'], headers=AUTH)).status_code == 503
        finally:
            if process.pid is not None:
                if process.is_alive():
                    process.kill()
                await asyncio.to_thread(process.join, 5)
            if server:
                await server.close()
    asyncio.run(scenario())
