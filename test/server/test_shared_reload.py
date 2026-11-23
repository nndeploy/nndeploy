"""Shared Reload through real Server replicas, native Graphs and Redis."""
import asyncio
import copy

import httpx
import pytest

from nndeploy.server import Server
from nndeploy.server.errors import ServerError
from nndeploy.server.worker import Worker
from test_redis_queue import backend, redis_service
from test_shared_server import shared_deploy, entry_config, until
from test_server_api import text_graph
from test_control import configuration


def config(value, isolation='thread'):
    result = entry_config()
    result['inputs']['text']['default'] = value
    result['execution']['isolation'] = isolation
    return result


def replicas(backend, tmp_path, *, isolation='thread'):
    db, redis = backend
    options = dict(resources=tmp_path / 'shared', data_dir=tmp_path / 'shared', database_url=db.url.render_as_string(hide_password=False),
                   redis_config=redis, admin_token='test-admin')
    return [Server(name='model', deploy=shared_deploy, config=config('old', isolation), **options) for _ in range(2)]


@pytest.mark.parametrize('isolation', ['thread', 'process'])
def test_new_requests_switch_while_running_and_queued_old_tasks_drain(backend, tmp_path, isolation):
    db, _ = backend
    first, second = replicas(backend, tmp_path, isolation=isolation)
    release = tmp_path / 'release'
    async def scenario():
        reloading = None
        try:
            await first.start()
            await second.start()
            old_key = first.entries['model']['queue'].key
            old = []
            for index in range(2):
                marker = tmp_path / f'marker-{index}'
                old.append(await first.submit('model', {'release': str(release), 'marker': str(marker)}))
                await until(marker.exists)
            old.append(await first.submit('model', {}))
            assert db.get_task(old[-1][0])['state'] == 'pending'
            reloading = asyncio.create_task(first.reload_entry('model', config=config('new', isolation)))
            await until(lambda: first.entries['model']['queue'].key != old_key)
            await until(lambda: second.entries['model']['queue'].key != old_key)
            assert not reloading.done()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=first.app), base_url='http://test') as client:
                assert (await client.get('/health/ready')).status_code == 200
            assert (await first.deploy('model', {}))['outputs']['text'] == 'new'
            assert (await second.deploy('model', {}))['outputs']['text'] == 'new'
            assert all(db.get_task(identity)['queue_key'] == old_key for identity, _ in old)
            release.touch()
            results = await asyncio.gather(*(future for _, future in old))
            assert [row['outputs']['text'] for row in results] == ['old'] * 3
            assert (await reloading)['status'] == 'ready'
            await until(lambda: not db.shared.draining(first.entries['model']['shared_key']))
            assert db.queue_usage(first.entries['model']['queue'].group_key)['inflight'] == 0
            # Reusing the same config still gets fresh workers without reopening
            # a retired queue or confusing its old task identities.
            before = first.entries['model']['queue'].key
            await first.reload_entry('model')
            assert first.entries['model']['queue'].key != before
            assert (await first.deploy('model', {}))['outputs']['text'] == 'new'
        finally:
            release.touch()
            if reloading is not None:
                await asyncio.gather(reloading, return_exceptions=True)
            await asyncio.gather(first.close(), second.close())
    asyncio.run(scenario())


def test_stale_replica_rejects_before_binding_and_then_follows_active_config(backend, tmp_path):
    first, second = replicas(backend, tmp_path)
    async def scenario():
        try:
            await first.start()
            await second.start()
            second.shared_control.cancel()
            await asyncio.gather(second.shared_control, return_exceptions=True)
            await first.reload_entry('model', config=config('new'))
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=second.app), base_url='http://test') as client:
                assert (await client.get('/health/ready')).status_code == 503
                response = await client.post('/v1/server/model/deploy', json={'inputs': {'text': 42}})
                assert response.status_code == 409, response.text
                assert response.json()['error']['code'] == 'entry_changed'
            old = second.entries['model']
            await second._follow_shared_configuration('model', old, await second.shared_runtime.current(old))
            assert (await second.deploy('model', {}))['outputs']['text'] == 'new'
            await second.stop_entry('model')
            await until(lambda: first.describe('model')['status'] == 'stopped')
            await first.start_entry('model')
            assert (await first.deploy('model', {}))['outputs']['text'] == 'new'
        finally:
            await asyncio.gather(first.close(), second.close())
    asyncio.run(scenario())


def test_initialization_failure_preserves_active_entry_and_releases_candidate(backend, tmp_path, monkeypatch):
    db, _ = backend
    first, second = replicas(backend, tmp_path)
    async def scenario():
        try:
            await first.start()
            await second.start()
            old_key = first.entries['model']['queue'].key
            original = Worker.start
            async def fail(worker):
                await original(worker)
                if worker.config.inputs['text'].default == 'broken':
                    raise RuntimeError('injected warmup failure')
            monkeypatch.setattr(Worker, 'start', fail)
            with pytest.raises(RuntimeError, match='warmup'):
                await first.reload_entry('model', config=config('broken'))
            active = db.shared.get(first.entries['model']['shared_key'])
            assert active['active_queue'] == old_key and active['pending_queue'] is None
            assert (await first.deploy('model', {}))['outputs']['text'] == 'old'
            assert (await second.deploy('model', {}))['outputs']['text'] == 'old'
            assert not first.retiring_entries.get('model')
            await first.reload_entry('model', config=config('new'))
            assert (await first.deploy('model', {}))['outputs']['text'] == 'new'
        finally:
            await asyncio.gather(first.close(), second.close())
    asyncio.run(scenario())


def test_remote_stop_during_warmup_prevents_candidate_activation(backend, tmp_path, monkeypatch):
    db, _ = backend
    first, second = replicas(backend, tmp_path)
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        reloading = None
        try:
            await first.start()
            await second.start()
            original = Worker.start
            async def delayed(worker):
                if worker.config.inputs['text'].default == 'new':
                    entered.set()
                    await release.wait()
                await original(worker)
            monkeypatch.setattr(Worker, 'start', delayed)
            old_key = first.entries['model']['queue'].key
            reloading = asyncio.create_task(first.reload_entry('model', config=config('new')))
            await asyncio.wait_for(entered.wait(), 5)
            await second.stop_entry('model')
            release.set()
            with pytest.raises(ServerError) as changed:
                await reloading
            assert changed.value.code == 'shared_control_changed'
            state = db.shared.get(first.entries['model']['shared_key'])
            assert state['active_queue'] == old_key and state['pending_queue'] is None
            assert not state['accepting']
            await until(lambda: first.describe('model')['status'] == 'stopped')
        finally:
            release.set()
            if reloading is not None:
                await asyncio.gather(reloading, return_exceptions=True)
            await asyncio.gather(first.close(), second.close())
    asyncio.run(scenario())


def test_concurrent_reload_has_one_owner_and_caller_cancellation_does_not_abort(backend, tmp_path, monkeypatch):
    db, _ = backend
    first, second = replicas(backend, tmp_path)
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        reloading = None
        try:
            await first.start()
            await second.start()
            original = Worker.start
            async def delayed(worker):
                if worker.config.inputs['text'].default == 'new':
                    entered.set()
                    await release.wait()
                await original(worker)
            monkeypatch.setattr(Worker, 'start', delayed)
            reloading = asyncio.create_task(first.reload_entry('model', config=config('new')))
            await asyncio.wait_for(entered.wait(), 5)
            with pytest.raises(ServerError) as busy:
                await second.reload_entry('model', config=config('other'))
            assert busy.value.code == 'shared_control_changed'
            reloading.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reloading
            assert db.shared.get(first.entries['model']['shared_key'])['pending_queue'] is not None
            release.set()
            await until(lambda: first.entries['model']['config'].inputs['text'].default == 'new')
            await until(lambda: second.entries['model']['config'].inputs['text'].default == 'new')
            assert (await second.deploy('model', {}))['outputs']['text'] == 'new'
        finally:
            release.set()
            await asyncio.gather(first.close(), second.close())
    asyncio.run(scenario())


def test_activate_commit_ack_loss_retries_without_reinitializing_workers(backend, tmp_path, monkeypatch):
    first, second = replicas(backend, tmp_path)
    async def scenario():
        try:
            await first.start()
            original = first.db.shared.activate
            calls = []
            def uncertain(*args):
                result = original(*args)
                calls.append(args)
                if len(calls) == 1:
                    raise OSError('injected commit acknowledgement loss')
                return result
            monkeypatch.setattr(first.db.shared, 'activate', uncertain)
            await first.reload_entry('model', config=config('new'))
            assert len(calls) == 2 and calls[0] == calls[1]
            assert (await first.deploy('model', {}))['outputs']['text'] == 'new'
        finally:
            await asyncio.gather(first.close(), second.close())
    asyncio.run(scenario())


def test_native_workflow_restores_active_config_without_a_second_saved_snapshot(backend, tmp_path):
    db, redis = backend
    options = dict(resources=tmp_path / 'graph', data_dir=tmp_path / 'graph', database_url=db.url.render_as_string(hide_password=False), redis_config=redis)
    first = Server(name='echo', workflow=text_graph(), config=configuration('old'), **options)
    async def scenario():
        restored = None
        try:
            await first.start()
            original_identity = first.entries['echo']['source']['identity']
            changed = text_graph()
            changed['name_'] = 'updated_graph'
            await first.reload_entry('echo', workflow=changed, config=configuration('new'))
            assert first.entries['echo']['source']['identity'] != original_identity
            assert (await first.deploy('echo', {}))['outputs'] == {'text': 'new'}
            saved = db.load_server_configs(first.config_scope)
            assert len(saved) == 1 and 'workflow' not in saved[0]
            await first.close()
            restored = Server(**options)
            await restored.start()
            assert (await restored.deploy('echo', {}))['outputs'] == {'text': 'new'}
        finally:
            await first.close()
            if restored is not None:
                await restored.close()
    asyncio.run(scenario())


def test_late_stop_cannot_acknowledge_stopping_a_retired_configuration(backend, tmp_path, monkeypatch):
    db, _ = backend
    first, second = replicas(backend, tmp_path)
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        stopping = None
        try:
            await first.start()
            await second.start()
            second.shared_control.cancel()
            await asyncio.gather(second.shared_control, return_exceptions=True)
            original = second.shared_runtime.check
            async def delayed(entry):
                result = await original(entry)
                entered.set()
                await release.wait()
                return result
            monkeypatch.setattr(second.shared_runtime, 'check', delayed)
            stopping = asyncio.create_task(second.stop_entry('model'))
            await asyncio.wait_for(entered.wait(), 5)
            await first.reload_entry('model', config=config('new'))
            release.set()
            with pytest.raises(ServerError) as stale:
                await stopping
            assert stale.value.code == 'entry_changed'
            assert db.shared.get(first.entries['model']['shared_key'])['accepting']
            assert (await first.deploy('model', {}))['outputs']['text'] == 'new'
        finally:
            release.set()
            if stopping is not None:
                await asyncio.gather(stopping, return_exceptions=True)
            await asyncio.gather(first.close(), second.close())
    asyncio.run(scenario())


def test_shared_workflow_files_release_only_after_every_replica_exits_old_config(backend, tmp_path):
    import io
    from test_server_api import image_graph
    from test_storage import picture
    from test_config_files import assert_protected
    db, redis = backend
    options = dict(resources=tmp_path / 'files', data_dir=tmp_path / 'files', database_url=db.url.render_as_string(hide_password=False), redis_config=redis)
    first, second = Server(**options), Server(**options)
    old_file = first.storage.save(io.BytesIO(picture()), 'image/png')['file_id']
    new_file = first.storage.save(io.BytesIO(picture()), 'image/png')['file_id']
    def configuration(file):
        return {'inputs': {'image': {'type': 'image', 'node': ['decode'], 'required': False, 'default': {'file_id': file}}},
                'outputs': {'image': {'type': 'image', 'node': ['encode']}}}
    for server in (first, second):
        server.add(name='image', workflow=image_graph(), config=configuration(old_file))
    async def scenario():
        try:
            await first.start()
            await second.start()
            second.shared_control.cancel()
            await asyncio.gather(second.shared_control, return_exceptions=True)
            await first.reload_entry('image', config=configuration(new_file))
            assert_protected(db, old_file)
            assert_protected(db, new_file)
            old = second.entries['image']
            await second._follow_shared_configuration('image', old, await second.shared_runtime.current(old))
            db.mark_file_deleting(old_file)
            assert db.get_file(old_file)['state'] == 'deleting'
            assert (await second.deploy('image', {}))['outputs']['image']['file_id']
            assert_protected(db, new_file)
        finally:
            await asyncio.gather(first.close(), second.close())
    asyncio.run(scenario())


def test_editor_can_update_a_running_or_stopped_shared_workflow(backend, tmp_path, monkeypatch):
    import queue
    from types import SimpleNamespace
    from nndeploy.server.template import WorkflowTemplateManager
    monkeypatch.setattr(WorkflowTemplateManager, 'init_templates', lambda: tmp_path)
    db, redis = backend
    server = Server(SimpleNamespace(resources=str(tmp_path), debug=True), queue.Queue(), queue.Queue(), queue.Queue(),
        admin_token='editor-test', redis_config=redis, database_url=db.url.render_as_string(hide_password=False))
    async def scenario():
        try:
            await server.start()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app, raise_app_exceptions=False), base_url='http://test') as client:
                headers = {'Authorization': 'Bearer editor-test'}
                body = {'id': '', 'businessContent': text_graph()}
                started = await client.post('/api/workflow/server/start', json=body, headers=headers)
                assert started.status_code == 200, started.text
                info = started.json()
                body['id'] = info['workflow_id']
                name = info['name']
                for stop in (False, True):
                    if stop:
                        assert (await client.post(f'/api/server/{name}/stop', headers=headers)).status_code == 200
                        broken = copy.deepcopy(body)
                        broken['businessContent']['node_repository_'][0]['key_'] = 'missing.Node'
                        failed = await client.post('/api/workflow/server/start', json=broken, headers=headers)
                        assert failed.status_code == 500, failed.text
                        assert server.entries[name]['status'] == 'stopped'
                        assert not db.shared.status(server.entries[name]['shared_key'])['accepting']
                    old = server.entries[name]['queue'].key
                    body['businessContent']['node_repository_'][0].setdefault('frontend_params', {})['description'] = f'updated-{stop}'
                    updated = await client.post('/api/workflow/server/start', json=body, headers=headers)
                    assert updated.status_code == 200, updated.text
                    assert updated.json()['workflow_id'] == info['workflow_id']
                    assert server.entries[name]['queue'].key != old
                    result = await client.post(info['deploy_path'], json={'inputs': {'source': 'editor'}})
                    assert result.status_code == 200 and result.json()['outputs']['sink'] == 'editor', result.text
        finally:
            await server.close()
    asyncio.run(scenario())


def test_http_reload_uses_existing_admin_route_and_code_restart_follows_active(backend, tmp_path):
    first, second = replicas(backend, tmp_path)
    async def scenario():
        try:
            await first.start()
            await first.reload_entry('model', config=config('new'))
            # This instance still has the original code/config registration.
            await second.start()
            assert (await second.deploy('model', {}))['outputs']['text'] == 'new'
            before = second.entries['model']['queue'].key
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=second.app), base_url='http://test') as client:
                assert (await client.post('/api/server/model/reload')).status_code == 401
                response = await client.post('/api/server/model/reload', headers={'Authorization': 'Bearer test-admin'})
                assert response.status_code == 200, response.text
            assert second.entries['model']['queue'].key != before
            assert (await second.deploy('model', {}))['outputs']['text'] == 'new'
        finally:
            await asyncio.gather(first.close(), second.close())
    asyncio.run(scenario())


def test_reload_between_control_observations_does_not_stop_the_stale_replica(backend, tmp_path, monkeypatch):
    first, second = replicas(backend, tmp_path)
    release = tmp_path / 'release'
    async def scenario():
        observed, resume = asyncio.Event(), asyncio.Event()
        reloading = None
        try:
            await first.start()
            await second.start()
            old_key = first.entries['model']['queue'].key
            tasks = []
            for i in range(2):
                marker = tmp_path / f'held-{i}'
                tasks.append(await first.submit('model', {'release': str(release), 'marker': str(marker)}))
                await until(marker.exists)
            original = second.shared_runtime.status
            intercepted = False
            async def stale_once(entry):
                nonlocal intercepted
                value = await original(entry)
                if not intercepted and value['active_queue'] == old_key:
                    intercepted = True
                    observed.set()
                    await resume.wait()
                return value
            monkeypatch.setattr(second.shared_runtime, 'status', stale_once)
            await asyncio.wait_for(observed.wait(), 5)
            reloading = asyncio.create_task(first.reload_entry('model', config=config('new')))
            await until(lambda: first.entries['model']['queue'].key != old_key)
            resume.set()
            await until(lambda: second.entries['model']['queue'].key != old_key, timeout=3)
            assert not release.exists()
            assert (await second.deploy('model', {}))['outputs']['text'] == 'new'
            release.touch()
            assert all(result['outputs']['text'] == 'old' for result in await asyncio.gather(*(f for _, f in tasks)))
            await reloading
        finally:
            resume.set()
            release.touch()
            if reloading is not None:
                await asyncio.gather(reloading, return_exceptions=True)
            await asyncio.gather(first.close(), second.close())
    asyncio.run(scenario())
