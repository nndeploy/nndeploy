"""Saved workflows and live runtimes protect files independently of tasks."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import json
import threading
import uuid

from fastapi.testclient import TestClient
from PIL import Image
import pytest
from sqlalchemy import event

from nndeploy.server import Server
from nndeploy.server.errors import ServerError
from nndeploy.server.resources import server_scope
from test_db import db
from test_server_api import image_graph


def file_record(database):
    identity = 'file_' + uuid.uuid4().hex
    database.create_file(dict(file_id=identity, storage_key=identity, media_type='text/plain', size=1, created_at=1))
    return identity


def config_value(file, enabled=True):
    return dict(config={'name': 'image', 'inputs': {'image': {'type': 'image', 'node': ['decode'],
                    'required': False, 'default': {'file_id': file}}},
                'outputs': {'image': {'type': 'image', 'node': ['encode']}}}, workflow=image_graph(), enabled=enabled)


def assert_protected(database, file):
    with pytest.raises(ServerError) as failure:
        database.mark_file_deleting(file)
    assert failure.value.code == 'file_in_use'
    assert database.get_file(file)['state'] == 'ready'


def test_saved_config_replacement_atomically_switches_references(db):
    scope = uuid.uuid4().hex
    first, second = file_record(db), file_record(db)
    original, replacement = config_value(first), config_value(second, False)
    db.save_server_config(scope, 'image', original)
    assert_protected(db, first)
    assert db.load_server_configs(scope) == [original]
    assert db.load_server_configs(uuid.uuid4().hex) == []
    # An old private JSON left after upgrade must not revert a newer DB config.
    db.save_server_config(scope, 'image', replacement, only_if_absent=True)
    assert db.load_server_configs(scope) == [original]
    db.save_server_config(scope, 'image', replacement)
    assert db.load_server_configs(scope) == [replacement]
    db.mark_file_deleting(first)
    db.delete_file(first)
    assert_protected(db, second)
    broken = config_value('file_missing_' + uuid.uuid4().hex)
    with pytest.raises(ServerError):
        db.save_server_config(scope, 'image', broken)
    assert db.load_server_configs(scope) == [replacement]
    assert_protected(db, second)


def test_runtime_references_are_independent_and_survive_one_owner_exit(db):
    file = file_record(db)
    first, second = uuid.uuid4().hex, uuid.uuid4().hex
    for holder in (first, second):
        db.hold_files(holder, {'default': {'file_id': file}})
        db.hold_files(holder, {'default': {'file_id': file}})
    assert_protected(db, file)
    db.release_files(first)
    assert_protected(db, file)
    db.release_files(second)
    db.mark_file_deleting(file)
    with pytest.raises(ServerError):
        db.hold_files(uuid.uuid4().hex, {'file_id': file})
    db.delete_file(file)


@pytest.mark.parametrize('kind', ['config', 'runtime'])
def test_resource_reference_and_delete_race_cannot_accept_a_deleted_file(db, kind):
    file, holder = file_record(db), uuid.uuid4().hex
    barrier = threading.Barrier(2)
    def retain():
        barrier.wait(5)
        try:
            if kind == 'runtime':
                db.hold_files(holder, {'file_id': file})
            else:
                db.save_server_config(holder, 'image', config_value(file))
            return 'retained'
        except ServerError:
            return 'unavailable'
    def remove():
        barrier.wait(5)
        try:
            db.mark_file_deleting(file)
            return 'deleting'
        except ServerError:
            return 'referenced'
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(retain), pool.submit(remove)]
        outcome = tuple(job.result() for job in pending)
    assert outcome in (('retained', 'referenced'), ('unavailable', 'deleting'))
    if outcome[0] == 'retained':
        assert_protected(db, file)


def test_config_failure_rolls_back_data_and_resource_references(db):
    scope = uuid.uuid4().hex
    first, second = file_record(db), file_record(db)
    original = config_value(first)
    db.save_server_config(scope, 'image', original)
    def fail(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith('INSERT INTO server_config_file_refs'):
            raise RuntimeError('injected configuration write failure')
    event.listen(db.engine, 'after_cursor_execute', fail)
    try:
        with pytest.raises(RuntimeError, match='injected'):
            db.save_server_config(scope, 'image', config_value(second))
    finally:
        event.remove(db.engine, 'after_cursor_execute', fail)
    assert db.load_server_configs(scope) == [original]
    assert_protected(db, first)
    db.mark_file_deleting(second)
    db.delete_file(second)


def image_file(server, color):
    picture = io.BytesIO()
    Image.new('RGB', (4, 6), color).save(picture, format='PNG')
    picture.seek(0)
    return server.storage.save(picture, 'image/png')['file_id']


def test_default_file_is_protected_before_first_task_and_across_stop_and_restart(tmp_path):
    server = Server(data_dir=tmp_path, admin_token='admin')
    file = image_file(server, (10, 20, 30))
    value = config_value(file)
    server.add(workflow=value['workflow'], config=value['config'])
    headers = {'Authorization': 'Bearer admin'}
    with TestClient(server.app) as client:
        assert client.delete('/v1/files/' + file, headers=headers).status_code == 409
        assert client.post('/api/server/image/stop', headers=headers).status_code == 200
        assert client.delete('/v1/files/' + file, headers=headers).status_code == 409
    restored = Server(data_dir=tmp_path, admin_token='admin')
    with TestClient(restored.app) as client:
        assert client.get('/v1/server/image/schema').json()['status'] == 'stopped'
        assert client.delete('/v1/files/' + file, headers=headers).status_code == 409
        assert client.post('/api/server/image/start', headers=headers).status_code == 200
        result = client.post('/v1/server/image/deploy', json={})
        assert result.status_code == 200, result.text
        content = client.get('/v1/files/' + result.json()['outputs']['image']['file_id'] + '/content')
        with Image.open(io.BytesIO(content.content)) as image:
            assert image.size == (4, 6) and image.getpixel((0, 0)) == (10, 20, 30)
        from test_retention import expire
        expire(restored.db, result.json()['task_id'])
        assert client.post('/api/server/cleanup-tasks', headers=headers).json()['deleted_tasks'] == 1
        assert client.delete('/v1/files/' + file, headers=headers).status_code == 409


def test_reload_keeps_old_default_until_old_worker_really_closes(tmp_path):
    server = Server(data_dir=tmp_path)
    first, second = image_file(server, (1, 2, 3)), image_file(server, (40, 50, 60))
    server.add(workflow=image_graph(), config=config_value(first)['config'])
    async def scenario():
        await server.start()
        old = server.entries['image']['workers'][0]
        close = old.close
        entered, release = asyncio.Event(), asyncio.Event()
        async def held_close():
            entered.set()
            await release.wait()
            await close()
        old.close = held_close
        reloading = asyncio.create_task(server.reload_entry('image', config=config_value(second)['config']))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            assert server.db.load_server_configs(server.config_scope)[0]['config']['inputs']['image']['default'] == {'file_id': second}
            assert_protected(server.db, first)
            assert_protected(server.db, second)
            release.set()
            await asyncio.wait_for(reloading, 5)
            server.storage.remove(first)
            assert server.db.get_file(first) is None
        finally:
            release.set()
            await reloading
            await server.close()
    asyncio.run(scenario())
    with TestClient(Server(data_dir=tmp_path).app) as client:
        result = client.post('/v1/server/image/deploy', json={})
        assert result.status_code == 200
        with Image.open(io.BytesIO(client.get('/v1/files/' + result.json()['outputs']['image']['file_id'] + '/content').content)) as image:
            assert image.getpixel((0, 0)) == (40, 50, 60)


def test_failed_reload_releases_candidate_files_and_preserves_saved_defaults(tmp_path):
    server = Server(data_dir=tmp_path)
    first, second = image_file(server, (1, 2, 3)), image_file(server, (4, 5, 6))
    server.add(workflow=image_graph(), config=config_value(first)['config'])
    async def scenario():
        await server.start()
        try:
            broken = image_graph()
            broken['node_repository_'][0]['key_'] = 'missing.Node'
            with pytest.raises(Exception):
                await server.reload_entry('image', workflow=broken, config=config_value(second)['config'])
            assert_protected(server.db, first)
            server.storage.remove(second)
            assert server.db.load_server_configs(server.config_scope)[0]['config']['inputs']['image']['default'] == {'file_id': first}
        finally:
            await server.close()
    asyncio.run(scenario())


def test_two_code_servers_hold_their_own_default_files(tmp_path):
    first = Server(data_dir=tmp_path)
    file = image_file(first, (1, 2, 3))
    second = Server(data_dir=tmp_path)
    config = {'inputs': {'file': {'type': 'file', 'required': False, 'default': {'file_id': file}}}}
    for server in (first, second):
        server.add(name='code', deploy=lambda values: values, config=config)
    async def scenario():
        await first.start()
        await second.start()
        await first.close()
        try:
            assert_protected(second.db, file)
        finally:
            await second.close()
        second.storage.remove(file)
        second.db.close()
    asyncio.run(scenario())


def test_defaults_are_protected_while_handler_initializes(tmp_path):
    entered, release = threading.Event(), threading.Event()
    class Handler:
        def init(self):
            entered.set()
            assert release.wait(5)
        def deploy(self, values):
            return values
    server = Server(data_dir=tmp_path)
    file = image_file(server, (1, 2, 3))
    server.add(name='code', handler_factory=Handler,
               config={'inputs': {'file': {'type': 'file', 'required': False, 'default': {'file_id': file}}}})
    async def scenario():
        starting = asyncio.create_task(server.start())
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            assert_protected(server.db, file)
        finally:
            release.set()
            await starting
            await server.close()
        server.storage.remove(file)
        server.db.close()
    asyncio.run(scenario())


def test_failed_worker_release_keeps_runtime_file_protection(tmp_path):
    server = Server(data_dir=tmp_path)
    file = image_file(server, (1, 2, 3))
    server.add(name='code', deploy=lambda values: values,
               config={'inputs': {'file': {'type': 'file', 'required': False, 'default': {'file_id': file}}}})
    async def scenario():
        await server.start()
        entry = server.entries['code']
        close = entry['workers'][0].close
        async def fail():
            raise RuntimeError('worker release not confirmed')
        entry['workers'][0].close = fail
        try:
            with pytest.raises(RuntimeError, match='not confirmed'):
                await server.close()
            assert_protected(server.db, file)
        finally:
            # The test explicitly confirms actual release before removing the
            # hold. A production process crash must not expire it on a timer.
            await close()
            server.db.release_files(entry['file_holder'])
            server.storage.remove(file)
            server.db.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('scheduling', ['serial', 'engine', 'concurrent'])
def test_partial_initialization_with_failed_cleanup_does_not_release_defaults(tmp_path, scheduling):
    class Handler:
        def init(self):
            raise RuntimeError('initialization failed')
        def deinit(self):
            raise RuntimeError('release not confirmed')
        def deploy(self, values):
            return values
    class AsyncHandler(Handler):
        async def deploy(self, values):
            return values
    server = Server(data_dir=tmp_path)
    file = image_file(server, (1, 2, 3))
    server.add(name='code', handler_factory=AsyncHandler if scheduling == 'engine' else Handler,
               config={'execution': {'scheduling': scheduling},
                       'inputs': {'file': {'type': 'file', 'required': False, 'default': {'file_id': file}}}})
    async def scenario():
        try:
            with pytest.raises(RuntimeError, match='release not confirmed'):
                await server.start()
            assert_protected(server.db, file)
        finally:
            with pytest.raises(RuntimeError, match='release not confirmed'):
                await server.close()
            # This handler allocated no external resources. Only the fixture
            # knows that; production code retains an uncertain runtime's hold.
            server.db.release_files(server.entries['code']['file_holder'])
            server.storage.remove(file)
            server.db.close()
    asyncio.run(scenario())


def test_legacy_config_import_is_one_way_and_preserves_stopped_workflow(tmp_path):
    server = Server(data_dir=tmp_path)
    first, second = image_file(server, (1, 2, 3)), image_file(server, (4, 5, 6))
    saved = config_value(first, enabled=False)
    directory = tmp_path / 'entries'
    directory.mkdir()
    legacy = directory / 'image.json'
    legacy.write_text(json.dumps(saved))
    with TestClient(server.app):
        assert server.entries['image']['status'] == 'stopped'
        assert_protected(server.db, first)
        server.db.save_server_config(server.config_scope, 'image', config_value(second, False))
    assert json.loads(legacy.read_text()) == saved
    reopened = Server(data_dir=tmp_path)
    with TestClient(reopened.app):
        assert reopened.entries['image']['config'].inputs['image'].default == {'file_id': second}
        assert_protected(reopened.db, second)
        reopened.storage.remove(first)


def test_workspace_identity_survives_concurrent_start_and_directory_move(tmp_path):
    original = tmp_path / 'original'
    original.mkdir()
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(lambda _: server_scope(original), range(24)))
    assert len(set(ids)) == 1
    renamed = tmp_path / 'renamed'
    original.rename(renamed)
    assert server_scope(renamed) == ids[0]
    another = tmp_path / 'another'
    another.mkdir()
    assert server_scope(another) != ids[0]
