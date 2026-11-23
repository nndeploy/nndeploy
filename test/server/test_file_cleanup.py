"""File expiry must respect storage ownership, live readers and task commits."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import threading
import time
import uuid

from fastapi.testclient import TestClient
import pytest
import requests
from sqlalchemy import select, update

from nndeploy.server import Server
from nndeploy.server.api.deploy import FileContentResponse
from nndeploy.server.db import files, file_holds
from nndeploy.server.errors import ServerError
from nndeploy.server.storage import LocalStorage
from test_db import db
from test_retention import expire as expire_task
from test_storage import local, remote, s3_service, picture


def expire(database, *identities, signed=False):
    with database.engine.begin() as connection:
        values = dict(purge_after=database.clock(connection) - 1, cleanup_after=None)
        if signed:
            values['protected_until'] = database.clock(connection) - 1
        connection.execute(update(files).where(files.c.file_id.in_(identities)).values(**values))


def upload(storage, **kwargs):
    return storage.save(io.BytesIO(b'file lifecycle'), 'text/plain', **kwargs)['file_id']


def running_task(database, owner=None):
    identity = 'task_' + uuid.uuid4().hex
    database.create_task(identity, uuid.uuid4().hex, time.time(), owner=owner)
    database.start_task(identity, time.time())
    return identity


def test_expiry_is_opt_in_and_scoped_to_its_local_directory(db, tmp_path):
    first = LocalStorage(tmp_path / 'first', db, config={'retention_seconds': 60.0})
    second = LocalStorage(tmp_path / 'second', db)
    before = db.clock()
    file, permanent = upload(first), upload(second)
    assert before + 60 <= db.get_file(file)['purge_after'] <= db.clock() + 60
    assert db.get_file(permanent)['purge_after'] is None
    assert first.cleanup()['deleted'] == 0
    expire(db, file, permanent)
    with pytest.raises(ServerError) as wrong:
        second.remove(file)
    assert wrong.value.code == 'storage_mismatch'
    assert first.cleanup()['deleted'] == 1
    assert db.get_file(permanent) is not None
    assert second.cleanup()['deleted'] == 1
    assert first.cleanup()['scanned'] == 0
    with pytest.raises(ValueError):
        first.cleanup(0)


def test_tasks_saved_configs_and_live_holds_all_block_expired_content(db, tmp_path):
    storage = LocalStorage(tmp_path / 'files', db)
    a, b, c, free = [upload(storage) for _ in range(4)]
    task, scope, holder = 'task_' + uuid.uuid4().hex, uuid.uuid4().hex, uuid.uuid4().hex
    db.create_task(task, uuid.uuid4().hex, time.time(), {'file_id': a})
    db.save_server_config(scope, 'saved', {'workflow': {}, 'config': {'default': {'file_id': b}}})
    db.hold_files(holder, {'file_id': c})
    expire(db, a, b, c, free)
    assert storage.cleanup()['deleted'] == 1
    assert all(db.get_file(file)['state'] == 'ready' for file in (a, b, c))
    db.finish_task(task, 'cancelled', time.time())
    assert storage.cleanup()['deleted'] == 0
    expire_task(db, task)
    db.cleanup_tasks()
    db.save_server_config(scope, 'saved', {'workflow': {}, 'config': {}})
    db.release_files(holder)
    assert storage.cleanup()['deleted'] == 3


def test_output_is_referenced_in_the_same_transaction_and_failure_reclaims_only_outputs(db, tmp_path):
    storage = LocalStorage(tmp_path / 'files', db)
    input_file = upload(storage, owner='alice')
    task = 'task_' + uuid.uuid4().hex
    db.create_task(task, uuid.uuid4().hex, time.time(), {'file_id': input_file}, owner='alice')
    db.start_task(task, time.time())
    output = upload(storage, owner='alice', task_id=task)
    expire(db, output, input_file)
    assert storage.cleanup()['deleted'] == 0
    for options in ({'attempt': 2, 'owner': 'alice'}, {'owner': 'bob'}, {'worker_token': 'stale', 'owner': 'alice'}):
        with pytest.raises(ServerError) as stale:
            upload(storage, task_id=task, **options)
        assert stale.value.code == 'execution_ownership_lost'
    assert db.get_file(output)['producer_task_id'] == task
    db.finish_task(task, 'failed', time.time())
    assert storage.cleanup()['deleted'] == 1
    assert db.get_file(output) is None
    assert db.get_file(input_file) is not None
    with pytest.raises(ServerError):
        upload(storage, owner='alice', task_id=task)


def test_lost_result_acknowledgement_never_discards_committed_success(db, tmp_path, monkeypatch):
    storage = LocalStorage(tmp_path / 'files', db)
    task = running_task(db)
    output = upload(storage, task_id=task)
    finish = db.finish_task
    def lost(*args, **kwargs):
        finish(*args, **kwargs)
        raise RuntimeError('lost acknowledgement')
    monkeypatch.setattr(db, 'finish_task', lost)
    with pytest.raises(RuntimeError):
        db.finish_task(task, 'succeeded', time.time(), {'file_id': output})
    storage.discard_output(output, task)
    expire(db, output)
    assert storage.cleanup()['deleted'] == 0
    assert storage.resolve(output)[0].read_bytes() == b'file lifecycle'
    expire_task(db, task)
    db.cleanup_tasks()
    assert storage.cleanup()['deleted'] == 1


def test_failed_upload_ack_preserves_output_until_task_outcome_is_known(db, tmp_path, monkeypatch):
    storage = LocalStorage(tmp_path / 'files', db)
    task = running_task(db)
    create = db.create_file
    def lost(record, **kwargs):
        create(record, **kwargs)
        raise RuntimeError('lost file acknowledgement')
    monkeypatch.setattr(db, 'create_file', lost)
    with pytest.raises(RuntimeError):
        upload(storage, task_id=task)
    with db.engine.connect() as connection:
        output = connection.execute(select(files.c.file_id).where(files.c.producer_task_id == task)).scalar_one()
    expire(db, output)
    assert storage.cleanup()['deleted'] == 0
    db.finish_task(task, 'failed', time.time())
    assert storage.cleanup()['deleted'] == 1


def test_failure_is_backed_off_and_other_files_continue(db, tmp_path, monkeypatch):
    storage = LocalStorage(tmp_path / 'files', db)
    bad, good = upload(storage), upload(storage)
    expire(db, bad, good)
    remove = storage._delete_content
    def fail(identity, *args):
        if identity == bad:
            raise OSError('temporary storage failure')
        return remove(identity, *args)
    monkeypatch.setattr(storage, '_delete_content', fail)
    result = storage.cleanup()
    assert result == {'scanned': 2, 'deleted': 1, 'skipped': 0, 'failed': 1}
    assert db.get_file(bad)['state'] == 'deleting'
    assert db.get_file(bad)['cleanup_after'] > db.clock()
    assert storage.cleanup()['scanned'] == 0
    with pytest.raises(ServerError) as gone:
        storage.resolve(bad)
    assert gone.value.status_code == 404
    with pytest.raises(ServerError):
        db.hold_files(uuid.uuid4().hex, {'file_id': bad})
    monkeypatch.setattr(storage, '_delete_content', remove)
    expire(db, bad)
    assert storage.cleanup()['deleted'] == 1


def test_context_read_blocks_automatic_and_manual_deletion(db, tmp_path):
    storage = LocalStorage(tmp_path / 'files', db)
    file = upload(storage)
    expire(db, file)
    with storage.open(file) as content:
        assert storage.cleanup()['deleted'] == 0
        with pytest.raises(ServerError) as active:
            storage.remove(file)
        assert active.value.code == 'file_in_use'
        assert content.read() == b'file lifecycle'
    assert storage.cleanup()['deleted'] == 1


def test_unscoped_legacy_file_is_not_automatically_deleted(db, tmp_path):
    storage = LocalStorage(tmp_path / 'files', db)
    file = upload(storage)
    with db.engine.begin() as connection:
        connection.execute(update(files).where(files.c.file_id == file).values(storage_scope=None, purge_after=0))
    assert storage.cleanup()['scanned'] == 0
    assert storage.resolve(file)[0].exists()
    other = LocalStorage(tmp_path / 'wrong', db)
    with pytest.raises(ServerError):
        other.remove(file)
    assert db.get_file(file) is not None
    storage.remove(file)


def test_cleanup_race_with_task_admission_never_loses_an_accepted_input(db, tmp_path):
    storage = LocalStorage(tmp_path / 'files', db)
    file, task = upload(storage), 'task_' + uuid.uuid4().hex
    expire(db, file)
    barrier = threading.Barrier(2)
    def admit():
        barrier.wait(5)
        try:
            db.create_task(task, uuid.uuid4().hex, time.time(), {'file_id': file})
            return True
        except ServerError as error:
            assert error.code == 'invalid_file'
            return False
    def cleanup():
        barrier.wait(5)
        return storage.cleanup()
    with ThreadPoolExecutor(max_workers=2) as pool:
        accepted, cleaning = pool.submit(admit), pool.submit(cleanup)
        admitted, result = accepted.result(), cleaning.result()
    assert result['failed'] == 0
    if admitted:
        assert result['deleted'] == 0
        assert storage.resolve(file)[0].read_bytes() == b'file lifecycle'
        assert db.get_task(task)['inputs'] == {'file_id': file}
    else:
        assert result['deleted'] == 1 and db.get_task(task) is None


def test_concurrent_cleaners_count_only_committed_deletions(db, tmp_path):
    storage = LocalStorage(tmp_path / 'files', db)
    identities = [upload(storage) for _ in range(8)]
    expire(db, *identities)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: storage.cleanup(limit=3), range(12)))
    final = storage.cleanup(limit=3)
    assert all(result['scanned'] <= 3 and result['failed'] == 0 for result in [*results, final])
    assert sum(result['deleted'] for result in [*results, final]) == 8
    assert all(db.get_file(identity) is None for identity in identities)


def test_verified_legacy_deletion_can_retry_after_content_is_gone(db, tmp_path, monkeypatch):
    storage = LocalStorage(tmp_path / 'files', db)
    file = upload(storage)
    path = storage.resolve(file)[0]
    with db.engine.begin() as connection:
        connection.execute(update(files).where(files.c.file_id == file).values(storage_scope=None))
    original = db.delete_file
    def fail(*args):
        raise RuntimeError('metadata removal temporarily unavailable')
    monkeypatch.setattr(db, 'delete_file', fail)
    with pytest.raises(RuntimeError):
        storage.remove(file)
    assert not path.exists()
    assert db.get_file(file)['storage_scope'] == storage.scope
    assert db.get_file(file)['state'] == 'deleting'
    monkeypatch.setattr(db, 'delete_file', original)
    assert storage.cleanup()['deleted'] == 1


def test_http_reader_is_protected_until_last_byte_and_cancellation_releases_hold(tmp_path):
    server = Server(resources=tmp_path, data_dir=tmp_path)
    file = server.storage.save(io.BytesIO(b'x' * 200000), 'application/octet-stream')['file_id']
    expire(server.db, file)
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        messages = []
        async def send(message):
            assert message['type'] != 'http.response.pathsend'
            if message['type'] == 'http.response.body':
                entered.set()
                await release.wait()
            messages.append(message)
        async def receive():
            return {'type': 'http.request', 'body': b''}
        response = FileContentResponse(server, file)
        reading = asyncio.create_task(response({'type': 'http', 'method': 'GET', 'headers': [],
                    'extensions': {'http.response.pathsend': {}}}, receive, send))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            assert (await server.cleanup_files())['deleted'] == 0
            reading.cancel()
            with pytest.raises(asyncio.CancelledError):
                await reading
            assert (await server.cleanup_files())['deleted'] == 1
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())


def test_cancel_during_reader_registration_waits_before_releasing(tmp_path, monkeypatch):
    server = Server(resources=tmp_path, data_dir=tmp_path)
    file = upload(server.storage)
    entered, release = threading.Event(), threading.Event()
    hold = server.db.hold_files
    def delayed(*args):
        entered.set()
        assert release.wait(5)
        hold(*args)
    monkeypatch.setattr(server.db, 'hold_files', delayed)
    async def scenario():
        async def unused(*args):
            raise AssertionError('Cancelled reader must not send content')
        reading = asyncio.create_task(FileContentResponse(server, file)(
            {'type': 'http', 'method': 'GET', 'headers': []}, unused, unused))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            reading.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await reading
            with server.db.engine.connect() as connection:
                assert connection.execute(select(file_holds).where(file_holds.c.file_id == file)).first() is None
            expire(server.db, file)
            assert (await server.cleanup_files())['deleted'] == 1
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())


def test_cleanup_management_and_range_response(tmp_path):
    server = Server(resources=tmp_path, data_dir=tmp_path, admin_token='admin')
    with TestClient(server.app) as client:
        file = upload(server.storage)
        response = client.get('/v1/files/' + file + '/content', headers={'Range': 'bytes=0-3'})
        assert response.status_code == 206 and response.content == b'file'
        expire(server.db, file)
        assert client.post('/api/server/cleanup-files').status_code == 401
        response = client.post('/api/server/cleanup-files', headers={'Authorization': 'Bearer admin'})
        assert response.status_code == 200 and response.json()['deleted'] == 1
        assert client.get('/v1/files/' + file + '/content').status_code == 404


def test_s3_signed_url_protection_and_metadata_only_lookup(remote, monkeypatch):
    storage, client = remote
    file = upload(storage)
    path, record = storage.resolve(file)
    path.unlink()
    def no_download(*args):
        raise AssertionError('Metadata must not download content')
    monkeypatch.setattr(storage.backend, 'open', no_download)
    assert storage.metadata(file)['file_id'] == file
    url = storage.signed_get(file, 60)
    expire(storage.db, file)
    assert storage.cleanup()['deleted'] == 0
    assert requests.get(url, timeout=5).content == b'file lifecycle'
    expire(storage.db, file, signed=True)
    assert storage.cleanup()['deleted'] == 1
    assert requests.get(url, timeout=5).status_code == 404


def test_s3_deletion_recovers_after_content_deleted_but_db_write_fails(remote, monkeypatch):
    storage, client = remote
    file = upload(storage)
    url = storage.signed_get(file, 60)
    expire(storage.db, file, signed=True)
    remove = storage.db.delete_file
    def fail(*args):
        raise RuntimeError('DB unavailable after remote deletion')
    monkeypatch.setattr(storage.db, 'delete_file', fail)
    assert storage.cleanup()['failed'] == 1
    assert requests.get(url, timeout=5).status_code == 404
    assert storage.db.get_file(file)['state'] == 'deleting'
    monkeypatch.setattr(storage.db, 'delete_file', remove)
    expire(storage.db, file, signed=True)
    assert storage.cleanup()['deleted'] == 1
    assert storage.db.get_file(file) is None
