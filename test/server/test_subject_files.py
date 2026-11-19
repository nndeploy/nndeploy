"""File quotas include uploads and pending deletion, across storage instances."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import json
import multiprocessing
import os
import threading
import time
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, func, text, update, event
from sqlalchemy.exc import SQLAlchemyError

from nndeploy.server import Server
from nndeploy.server.db import DB, files
from nndeploy.server.errors import ServerError
from nndeploy.server.storage import LocalStorage
from test_db import db
from test_auth import credentials
from test_storage import s3_service, remote, local, picture


def test_concurrent_upload_count_and_bytes_across_instances(db, tmp_path):
    owner = uuid.uuid4().hex
    storages = [LocalStorage(tmp_path / str(i), db, config={
        'max_files_per_subject': 3, 'max_storage_bytes_per_subject': 16}) for i in range(2)]
    barrier = threading.Barrier(8)

    def upload(i):
        storage = storages[i % 2]
        barrier.wait(timeout=10)
        try:
            return storage, storage.save(io.BytesIO(b'12345678'), 'text/plain', owner=owner)
        except ServerError as exc:
            assert exc.code in ('subject_files_full', 'subject_storage_full')
            return None
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = [result for result in pool.map(upload, range(8)) if result]
        assert len(results) == 2
        with db.engine.connect() as connection:
            used = connection.execute(select(func.count(), func.sum(files.c.size)).where(files.c.owner_key == DB.subject_key(owner))).one()
        assert tuple(used) == (2, 16)
        assert all(not list(s.root.glob('.pending-*')) and not list(s.root.glob('.upload-*')) for s in storages)
        first, record = results[0]
        assert 'owner_key' not in record
        first.remove(record['file_id'])
        assert storages[1].save(io.BytesIO(b'12345678'), 'text/plain', owner=owner)['size'] == 8
        # Exact subjects remain distinct under case-insensitive SQL collations.
        assert storages[0].save(io.BytesIO(b'12345678'), 'text/plain', owner=owner.upper())['size'] == 8
    finally:
        for storage in storages:
            storage.close()


def test_upload_reserves_count_before_read_and_is_not_public(db, tmp_path):
    owner = uuid.uuid4().hex
    storage = LocalStorage(tmp_path, db, config={'max_files_per_subject': 1})
    reading, release = threading.Event(), threading.Event()

    class Held(io.BytesIO):
        def read(self, size=-1):
            reading.set()
            assert release.wait(timeout=10)
            return super().read(size)

    class Unread(io.BytesIO):
        def read(self, size=-1):
            pytest.fail('Quota rejection must happen before reading file content')

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(storage.save, Held(b'hello'), 'text/plain', owner=owner)
            try:
                assert reading.wait(timeout=5)
                with db.engine.connect() as connection:
                    identity = connection.execute(select(files.c.file_id).where(files.c.owner_key == DB.subject_key(owner))).scalar_one()
                assert db.get_file(identity)['state'] == 'uploading'
                for operation in (storage.metadata, storage.remove):
                    with pytest.raises(ServerError) as hidden:
                        operation(identity)
                    assert hidden.value.status_code == 404
                with pytest.raises(ServerError) as full:
                    storage.save(Unread(b'new'), 'text/plain', owner=owner)
                assert full.value.code == 'subject_files_full'
            finally:
                release.set()
            assert future.result()['size'] == 5
    finally:
        release.set()
        storage.close()


def test_rejected_content_releases_reservation_and_committed_ack_loss_keeps_it(db, tmp_path, monkeypatch):
    owner = uuid.uuid4().hex
    storage = LocalStorage(tmp_path, db, config={'max_files_per_subject': 1, 'max_storage_bytes_per_subject': 5})
    try:
        with pytest.raises(ServerError) as too_large:
            storage.save(io.BytesIO(b'123456'), 'text/plain', owner=owner)
        assert too_large.value.code == 'subject_storage_full'
        with pytest.raises(ServerError):
            storage.save(io.BytesIO(b'bad'), 'image/png', owner=owner)
        assert not list(tmp_path.glob('.pending-*')) and not list(tmp_path.glob('.upload-*'))
        create = db.create_file

        def lost(record, **kwargs):
            create(record, **kwargs)
            raise RuntimeError('commit acknowledgement lost')

        monkeypatch.setattr(db, 'create_file', lost)
        with pytest.raises(RuntimeError, match='acknowledgement'):
            storage.save(io.BytesIO(b'12345'), 'text/plain', owner=owner)
        monkeypatch.setattr(db, 'create_file', create)
        with db.engine.connect() as connection:
            identity = connection.execute(select(files.c.file_id).where(files.c.owner_key == DB.subject_key(owner))).scalar_one()
        assert storage.resolve(identity)[0].read_bytes() == b'12345'
        with pytest.raises(ServerError) as full:
            storage.save(io.BytesIO(b'new'), 'text/plain', owner=owner)
        assert full.value.code == 'subject_files_full'
        storage.remove(identity)
        assert storage.save(io.BytesIO(b'new'), 'text/plain', owner=owner)['size'] == 3
    finally:
        storage.close()


def _crash_upload(database_url, root, owner):
    class Crash:
        read_once = False
        def read(self, size):
            if self.read_once:
                os._exit(17)
            self.read_once = True
            return b'a' * 1024**2
    storage = LocalStorage(root, DB(database_url), config={'max_files_per_subject': 1})
    storage.save(Crash(), 'text/plain', owner=owner)


def test_dead_writer_reservation_recovered_only_after_content_cleanup(db, tmp_path):
    owner = uuid.uuid4().hex
    process = multiprocessing.get_context('spawn').Process(target=_crash_upload,
        args=(db.url.render_as_string(hide_password=False), str(tmp_path), owner))
    process.start()
    process.join(timeout=15)
    try:
        assert process.exitcode == 17
        storage = LocalStorage(tmp_path, db, config={'max_files_per_subject': 1})
        try:
            with db.engine.connect() as connection:
                record = connection.execute(select(files).where(files.c.owner_key == DB.subject_key(owner))).mappings().one()
            assert record['state'] == 'uploading' and record['size'] == 1024**2
            assert (tmp_path / record['storage_key']).read_bytes() == b'a' * 1024**2
            with pytest.raises(ServerError) as full:
                storage.save(io.BytesIO(b'new'), 'text/plain', owner=owner)
            assert full.value.code == 'subject_files_full'
            storage.cleanup()
            assert db.get_file(record['file_id']) is None
            assert not (tmp_path / record['storage_key']).exists()
            assert storage.save(io.BytesIO(b'new'), 'text/plain', owner=owner)['size'] == 3
        finally:
            storage.close()
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        process.close()


def test_remote_cleanup_failure_retains_quota_until_reconciled(remote, monkeypatch):
    original, client = remote
    storage = LocalStorage(original.root, original.db, config=original.config.model_copy(update={
        'max_files_per_subject': 1, 'max_storage_bytes_per_subject': 5}))
    owner = uuid.uuid4().hex
    create, delete = storage.db.create_file, storage.backend.delete

    def fail(*args, **kwargs):
        raise RuntimeError('injected storage failure')
    try:
        monkeypatch.setattr(storage.db, 'create_file', fail)
        monkeypatch.setattr(storage.backend, 'delete', fail)
        with pytest.raises(RuntimeError):
            storage.save(io.BytesIO(b'hello'), 'text/plain', owner=owner)
        journal = next(storage.root.glob('.pending-*'))
        value = json.loads(journal.read_text())
        key = storage.config.prefix + '/' + value['storage_key']
        assert client.head_object(Bucket=storage.config.bucket, Key=key)['ContentLength'] == 5
        monkeypatch.setattr(storage.db, 'create_file', create)
        with pytest.raises(ServerError) as full:
            storage.save(io.BytesIO(b'new'), 'text/plain', owner=owner)
        assert full.value.code == 'subject_files_full'
        monkeypatch.setattr(storage.backend, 'delete', delete)
        assert storage.recover_pending() == 1
        assert not client.list_objects_v2(Bucket=storage.config.bucket, Prefix=key).get('Contents')
        assert storage.save(io.BytesIO(b'new'), 'text/plain', owner=owner)['size'] == 3
    finally:
        storage.close()


def test_process_graph_output_uses_same_file_quota(db, tmp_path):
    from test_server_api import image_graph
    owner = uuid.uuid4().hex
    server = Server(name='image', workflow=image_graph(), resources=tmp_path, data_dir=tmp_path,
        database_url=db.url.render_as_string(hide_password=False), storage_config={'max_files_per_subject': 1},
        config={'inputs': {'image': {'type': 'image', 'node': ['decode']}},
                'outputs': {'image': {'type': 'image', 'node': ['encode']}}, 'execution': {'isolation': 'process'}})
    uploaded = server.storage.save(io.BytesIO(picture()), 'image/png', owner=owner)

    async def scenario():
        try:
            await server.start()
            identity, future = await server.submit('image', {'image': {'file_id': uploaded['file_id']}}, owner=owner)
            with pytest.raises(ServerError):
                await future
            assert db.get_task(identity)['state'] == 'failed'
            with db.engine.connect() as connection:
                assert connection.execute(select(func.count()).select_from(files).where(files.c.owner_key == DB.subject_key(owner))).scalar_one() == 1
            assert server.storage.resolve(uploaded['file_id'])[0].read_bytes() == picture()
        finally:
            await server.close()
    asyncio.run(scenario())


def test_migration_includes_existing_file_bytes(db, tmp_path):
    owner = uuid.uuid4().hex
    original = LocalStorage(tmp_path, db)
    record = original.save(io.BytesIO(b'existing'), 'text/plain', owner=owner)
    original.close()
    with db.engine.begin() as connection:
        connection.execute(update(files).where(files.c.file_id == record['file_id']).values(owner_key=None))
        connection.execute(text("UPDATE alembic_version SET version_num = '0026_subject_task_quotas'"))
    upgraded = DB(db.url).init_schema()
    storage = LocalStorage(tmp_path, upgraded, config={'max_storage_bytes_per_subject': 8})
    try:
        assert upgraded.get_file(record['file_id'])['owner_key'] == DB.subject_key(owner)
        with pytest.raises(ServerError) as full:
            storage.save(io.BytesIO(b'new'), 'text/plain', owner=owner)
        assert full.value.code == 'subject_storage_full'
        storage.remove(record['file_id'])
        assert storage.save(io.BytesIO(b'new'), 'text/plain', owner=owner)['size'] == 3
    finally:
        storage.close()
        upgraded.close()


def test_file_http_quota_shared_by_identity_and_released_after_delete(db, tmp_path, credentials):
    auth, headers = credentials
    for key in auth['keys']:
        key['subject'] += '-' + uuid.uuid4().hex
    servers = [Server(resources=tmp_path / str(i), data_dir=tmp_path / str(i), auth_config=auth,
        database_url=db.url.render_as_string(hide_password=False),
        storage_config={'max_files_per_subject': 1, 'max_storage_bytes_per_subject': 5}) for i in range(2)]
    with TestClient(servers[0].app) as a, TestClient(servers[1].app) as b:
        def upload(client, person, content):
            return client.post('/v1/files', headers=headers[person], files={'file': ('value.txt', content, 'text/plain')})
        first = upload(a, 'alice', b'hello')
        assert first.status_code == 201, first.text
        denied = upload(b, 'alice', b'new')
        assert denied.status_code == 429 and denied.json()['error']['code'] == 'subject_files_full'
        assert upload(b, 'bob', b'new').status_code == 201
        assert a.delete('/v1/files/' + first.json()['file_id'], headers=headers['alice']).status_code == 200
        denied = upload(b, 'alice', b'123456')
        assert denied.status_code == 429 and denied.json()['error']['code'] == 'subject_storage_full'
        assert upload(b, 'alice', b'new').status_code == 201


@pytest.mark.parametrize('phase', ['reserve_file', 'create_file'])
def test_uncertain_database_reply_waits_for_actual_commit_before_cleanup(db, tmp_path, monkeypatch, phase):
    owner = uuid.uuid4().hex
    storage = LocalStorage(tmp_path, db, config={'max_files_per_subject': 1})
    committing, release, inspecting = threading.Event(), threading.Event(), threading.Event()
    writer = None
    errors = []
    original = getattr(db, phase)
    reconcile = db.reconcile_file_upload

    def held_commit(connection):
        if threading.current_thread() is writer:
            committing.set()
            assert release.wait(timeout=10)

    def uncertain(*args, **kwargs):
        nonlocal writer
        def complete():
            try:
                original(*args, **kwargs)
            except BaseException as exc:
                errors.append(exc)
        # Model a database still completing the transaction after the caller
        # loses its response. These are real SQL transactions and row locks.
        writer = threading.Thread(target=complete)
        writer.start()
        assert committing.wait(timeout=5)
        raise SQLAlchemyError('database acknowledgement unknown')

    def inspect(*args):
        inspecting.set()
        return reconcile(*args)

    event.listen(db.engine, 'commit', held_commit)
    monkeypatch.setattr(db, phase, uncertain)
    monkeypatch.setattr(db, 'reconcile_file_upload', inspect)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(storage.save, io.BytesIO(b'hello'), 'text/plain', owner=owner)
            try:
                assert inspecting.wait(timeout=5)
                value = json.loads(next(tmp_path.glob('.pending-*')).read_text())
                # Reconciliation must wait, not delete bytes using an old MVCC
                # snapshot and only then block trying to release the row.
                time.sleep(.1)
                assert not future.done()
                if phase == 'create_file':
                    assert (tmp_path / value['storage_key']).read_bytes() == b'hello'
            finally:
                release.set()
            with pytest.raises(ServerError) as failed:
                future.result(timeout=5)
            assert failed.value.status_code == 503
            assert failed.value.status_url == '/v1/files/' + value['file_id']
        writer.join(timeout=5)
        assert not writer.is_alive() and not errors
        with db.engine.connect() as connection:
            records = list(connection.execute(select(files).where(files.c.owner_key == DB.subject_key(owner))).mappings())
        if phase == 'create_file':
            assert len(records) == 1 and records[0]['state'] == 'ready'
            assert storage.resolve(records[0]['file_id'])[0].read_bytes() == b'hello'
        else:
            assert records == []
        assert not list(tmp_path.glob('.pending-*'))
    finally:
        release.set()
        if writer is not None:
            writer.join(timeout=5)
        event.remove(db.engine, 'commit', held_commit)
        storage.close()
