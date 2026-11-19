"""Service metadata reuses durable admission; races do not execute twice."""
import time
from concurrent.futures import ThreadPoolExecutor
import pytest
from nndeploy.server.db import DB
from nndeploy.server.errors import ServerError
from test_redis_queue import backend, redis_service


def test_cross_connection_exclusive_admission_and_external_id(tmp_path):
    path=tmp_path/'tasks.db'
    first=DB(path).init_schema();second=DB(path).init_schema()
    def submit(index):
        db=first if index%2 else second
        try:
            return db.create_task(f'task_{index}', 'videoedit',time.time(),{}, {},
                service={'scope':'same','external_id':str(index),'admission_group':'gpu'})
        except ServerError as error:
            assert error.code=='service_busy'
            return None
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            accepted=[identity for identity in pool.map(submit,range(16)) if identity]
        assert len(accepted)==1
        record=first.get_task(accepted[0]);external=record['service']['external_id']
        first.finish_task(accepted[0],'succeeded',time.time(),{})
        with pytest.raises(ServerError,match='Task ID already exists'):
            second.create_task('duplicate','videoedit',time.time(),{}, {},
                service={'scope':'same','external_id':external,'admission_group':'gpu'})
        assert second.external_task('videoedit',external,None,'same')['task_id']==accepted[0]
        assert not second.service_busy('same','gpu')
    finally:
        first.close();second.close()


def test_secret_store_concurrent_initialization(tmp_path):
    from nndeploy.server.secret_store import SecretStore
    def seal(index):
        return SecretStore(tmp_path).seal({'secret': str(index)})
    with ThreadPoolExecutor(max_workers=8) as pool:
        encrypted = list(pool.map(seal, range(32)))
    store = SecretStore(tmp_path)
    assert [store.open(value) for value in encrypted] == [{'secret': str(i)} for i in range(32)]
    assert not (tmp_path/'request.key').stat().st_mode & 0o077


def test_progress_claim_does_not_emit_after_terminal_transition(tmp_path, monkeypatch):
    from nndeploy.server import callbacks
    db = DB(tmp_path/'tasks.db').init_schema()
    try:
        db.create_task('task_progress', 'videoedit', time.time(), {}, {},
            callback_url='http://localhost/callback', callback_scope='scope',
            service={'scope': 'scope', 'progress_callbacks': True})
        db.start_task('task_progress', time.time())
        db.append_task_event('task_progress', {'progress': 25})
        claimed = callbacks.claim_progress(db, 'scope')
        assert claimed is not None
        db.finish_task('task_progress', 'succeeded', time.time(), {})
        assert callbacks.claim(db, 'scope') is None  # progress delivery owns the callback lease
        monkeypatch.setattr(callbacks, 'send', lambda value: pytest.fail('Stale progress callback sent'))
        callbacks.deliver(db, claimed)
        assert callbacks.claim(db, 'scope')['state'] == 'succeeded'
    finally:
        db.close()


@pytest.mark.parametrize('extension,media', [('.avi', 'video/x-msvideo'),
    ('.mov', 'video/quicktime'), ('.mkv', 'video/x-matroska'), ('.m4v', 'video/x-m4v')])
def test_video_containers_and_configured_limits(tmp_path, extension, media):
    import shutil
    import subprocess
    from nndeploy.server.storage import LocalStorage
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('FFmpeg required')
    source = tmp_path / ('source' + extension)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=size=16x16:rate=5',
                    '-t', '1', '-c:v', 'mpeg4', str(source)], check=True)
    db = DB(tmp_path/'tasks.db').init_schema()
    storage = LocalStorage(tmp_path/'files', db, config={'media_max_duration_seconds': 2, 'media_max_dimension': 32})
    try:
        with source.open('rb') as stream:
            result = storage.save(stream, media)
        path, record = storage.resolve(result['file_id'])
        assert path.suffix == extension and record['media_type'] == media
        storage.config.media_max_duration_seconds = .1
        with source.open('rb') as stream, pytest.raises(ServerError, match='Invalid or unsupported'):
            storage.save(stream, media)
    finally:
        storage.close()
        db.close()


def test_shared_queue_keeps_service_identity_and_admission(backend):
    import asyncio
    from nndeploy.server.task_queue import RedisTaskQueue
    db, config = backend
    async def scenario():
        queues = [RedisTaskQueue(db, config, name, 'model-config', 1) for name in ('normal', 'dmd')]
        try:
            for queue in queues:
                await queue.start()
            metadata = dict(scope=config['namespace'], external_id='one', admission_group='videoedit', deadline=time.time()+60)
            accepted = await queues[0].put({}, service=metadata)
            with pytest.raises(ServerError, match='A task is running'):
                await queues[1].put({}, service={**metadata, 'external_id': 'two'})
            received = await queues[0].get('worker')
            assert received['service'] == metadata
            await queues[0].finish(received, outputs={})
            assert db.external_task('normal', 'one', None, config['namespace'])['task_id'] == accepted['task_id']
            with pytest.raises(ServerError, match='Task ID already exists'):
                await queues[0].put({}, service=metadata)
            await queues[1].put({}, service={**metadata, 'external_id': 'two'})
            received = await queues[1].get('worker')
            await queues[1].finish(received, outputs={})
        finally:
            for queue in queues:
                await queue.close()
    asyncio.run(scenario())


def test_resource_lookup_and_cursor_keep_scope_owner_and_entry_boundaries(tmp_path):
    db = DB(tmp_path/'resources.db').init_schema()
    try:
        for identity, name, owner, scope in (
            ('task_a', 'edit', 'alice', 'shared'), ('task_b', 'vsr', 'alice', 'shared'),
            ('task_c', 'vsr', 'bob', 'shared'), ('task_d', 'vsr', 'alice', 'other'),
            ('task_e', 'private', 'alice', 'shared')):
            db.create_task(identity, name, 123, owner=owner,
                           service={'scope': scope, 'external_id': 'same'})
        names = ['edit', 'vsr']
        with pytest.raises(ServerError) as ambiguous:
            db.resource_task(names, 'same', 'shared', owner='alice')
        assert ambiguous.value.code == 'ambiguous_task'
        assert db.resource_task(['vsr'], 'same', 'shared', owner='alice')['task_id'] == 'task_b'
        assert db.resource_task(names, 'task_c', 'shared', owner='alice') is None
        assert db.resource_task(names, 'task_d', 'shared', owner='alice') is None
        assert db.resource_task(names, 'task_e', 'shared', owner='alice') is None
        assert db.resource_task([], 'task_a', 'shared', owner='alice') is None
        page = db.service_tasks(names, 'shared', owner='alice', order='asc', limit=1)
        assert [row['task_id'] for row in page] == ['task_a']
        page = db.service_tasks(names, 'shared', owner='alice', order='asc', after=page[0]['task_id'], limit=1)
        assert [row['task_id'] for row in page] == ['task_b']
        assert db.service_tasks(names, 'shared', owner='alice', after='task_c') == []
        with pytest.raises(ServerError) as ambiguous:
            db.service_tasks(names, 'shared', owner='alice', after='same')
        assert ambiguous.value.code == 'ambiguous_task'
        assert db.resource_task(names, 'task_c', 'shared', administrator=True)['owner'] == 'bob'
        assert len(db.service_tasks(names, 'shared', administrator=True)) == 3
    finally:
        db.close()
