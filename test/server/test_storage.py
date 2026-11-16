import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import time
import wave

import pytest
import requests
from PIL import Image

from nndeploy.server.db import DB
from nndeploy.server.errors import ServerError
from nndeploy.server.storage import LocalStorage


def picture():
    data = io.BytesIO()
    Image.new('RGB', (7, 9), (24, 60, 91)).save(data, format='PNG')
    return data.getvalue()


@pytest.fixture
def local(tmp_path):
    db = DB(tmp_path / 'files.db').init_schema()
    storage = LocalStorage(tmp_path / 'files', db)
    yield storage
    storage.close()
    db.close()


@pytest.mark.parametrize('data,media', [(picture(), 'image/png'), ('中文'.encode(), 'text/plain'),
                                      (b'{"value":1}', 'application/json'), (b'\x00\x01binary', 'application/octet-stream')])
def test_local_media_integrity_and_delete(local, data, media):
    record = local.save(io.BytesIO(data), media)
    path, stored = local.resolve(record['file_id'])
    assert path.read_bytes() == data
    assert record['sha256'] == hashlib.sha256(data).hexdigest()
    assert stored['media_type'] == media
    local.remove(record['file_id'])
    assert not path.exists() and local.db.get_file(record['file_id']) is None


@pytest.mark.parametrize('data,media', [(b'fake', 'image/png'), (picture(), 'image/jpeg'),
                                      (b'{"x":1,"x":2}', 'application/json'), (b'{"x":NaN}', 'application/json'),
                                      (b'\xff', 'text/plain'), (b'\x00', 'text/plain')])
def test_invalid_media_never_published(local, data, media):
    with pytest.raises(ServerError):
        local.save(io.BytesIO(data), media)
    assert {path.name for path in local.root.iterdir()} == {'.server-id'}


def test_local_size_and_metadata_failure_cleanup(local, monkeypatch):
    local.max_bytes = 10
    with pytest.raises(ServerError) as failure:
        local.save(io.BytesIO(b'x' * 11), 'application/octet-stream')
    assert failure.value.status_code == 413
    local.max_bytes = 10000
    def fail(record):
        raise RuntimeError('DB unavailable')
    monkeypatch.setattr(local.db, 'create_file', fail)
    with pytest.raises(RuntimeError):
        local.save(io.BytesIO(picture()), 'image/png')
    assert {path.name for path in local.root.iterdir()} == {'.server-id'}


def test_audio_validation_roundtrip(local):
    if not shutil.which('ffprobe'):
        pytest.skip('ffprobe unavailable')
    data = io.BytesIO()
    with wave.open(data, 'wb') as sound:
        sound.setnchannels(1)
        sound.setsampwidth(2)
        sound.setframerate(16000)
        sound.writeframes(b'\x00\x00' * 1600)
    record = local.save(io.BytesIO(data.getvalue()), 'audio/wav')
    path, _ = local.resolve(record['file_id'])
    with wave.open(str(path), 'rb') as result:
        assert result.getnframes() == 1600
    with pytest.raises(ServerError):
        local.save(io.BytesIO(data.getvalue()), 'video/mp4')


@pytest.mark.parametrize('committed', [True, False])
def test_local_uncertain_commit_is_reconciled_without_data_loss(local, monkeypatch, committed):
    create, lookup = local.db.create_file, local.db.get_file
    recorded = []
    def uncertain(record):
        if committed:
            create(record)
        recorded.append(record)
        raise RuntimeError('commit acknowledgement lost')
    def unavailable(*args):
        raise RuntimeError('database offline')
    monkeypatch.setattr(local.db, 'create_file', uncertain)
    monkeypatch.setattr(local.db, 'get_file', unavailable)
    with pytest.raises(RuntimeError, match='acknowledgement'):
        local.save(io.BytesIO(picture()), 'image/png')
    assert len(list(local.root.glob('.pending-*.json'))) == 1
    assert (local.root / recorded[0]['storage_key']).read_bytes() == picture()
    monkeypatch.setattr(local.db, 'get_file', lookup)
    assert local.recover_pending() == 1
    assert not list(local.root.glob('.pending-*.json'))
    if committed:
        assert local.resolve(recorded[0]['file_id'])[0].read_bytes() == picture()
    else:
        assert {path.name for path in local.root.iterdir()} == {'.server-id'}


@pytest.fixture(scope='module')
def s3_service(tmp_path_factory):
    if os.environ.get('NNDEPLOY_TEST_S3') != '1':
        pytest.skip('Set NNDEPLOY_TEST_S3=1 to start a local MinIO process')
    executable = shutil.which('minio')
    if executable is None:
        pytest.fail('MinIO executable is required for requested S3 integration tests')
    import boto3
    root = tmp_path_factory.mktemp('minio')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    endpoint = f'http://127.0.0.1:{port}'
    environment = dict(os.environ, MINIO_ROOT_USER='nndeploy-local-test', MINIO_ROOT_PASSWORD='nndeploy-local-test-password',
                       MINIO_BROWSER='off')
    with (root / 'minio.log').open('w') as log:
        process = subprocess.Popen([executable, 'server', '--address', f'127.0.0.1:{port}', str(root / 'data')],
                                   env=environment, stdout=log, stderr=log)
        try:
            for _ in range(200):
                if process.poll() is not None:
                    pytest.fail('MinIO exited: ' + (root / 'minio.log').read_text())
                try:
                    if requests.get(endpoint + '/minio/health/live', timeout=.2).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(.05)
            else:
                pytest.fail('MinIO startup timed out')
            client = boto3.client('s3', endpoint_url=endpoint, region_name='us-east-1',
                aws_access_key_id=environment['MINIO_ROOT_USER'], aws_secret_access_key=environment['MINIO_ROOT_PASSWORD'])
            client.create_bucket(Bucket='nndeploy-tests')
            yield endpoint, environment, client
            client.close()
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@pytest.fixture
def remote(s3_service, local, monkeypatch):
    endpoint, environment, client = s3_service
    monkeypatch.setenv('NNDEPLOY_TEST_ACCESS', environment['MINIO_ROOT_USER'])
    monkeypatch.setenv('NNDEPLOY_TEST_SECRET', environment['MINIO_ROOT_PASSWORD'])
    config = dict(backend='s3', bucket='nndeploy-tests', region='us-east-1', endpoint=endpoint,
                  access_key_env='NNDEPLOY_TEST_ACCESS', secret_key_env='NNDEPLOY_TEST_SECRET', max_bytes=24 * 1024**2)
    storage = LocalStorage(local.root, local.db, config=config)
    yield storage, client
    storage.close()


def test_s3_upload_download_cache_restore_signed_url_and_delete(remote):
    storage, client = remote
    data = picture()
    record = storage.save(io.BytesIO(data), 'image/png')
    path, metadata = storage.resolve(record['file_id'])
    path.unlink()
    assert storage.resolve(record['file_id'])[0].read_bytes() == data
    url = storage.signed_get(record['file_id'], 60)
    assert requests.get(url, timeout=5).content == data
    storage.remove(record['file_id'])
    assert requests.get(url, timeout=5).status_code == 404
    assert not path.exists()


def test_s3_multipart_and_remote_integrity(remote):
    storage, client = remote
    data = b'large-test' * (1024 * 1024)
    record = storage.save(io.BytesIO(data), 'application/octet-stream')
    path, metadata = storage.resolve(record['file_id'])
    path.unlink()
    assert storage.resolve(record['file_id'])[0].read_bytes() == data
    path.unlink()
    client.put_object(Bucket='nndeploy-tests', Key=storage.backend.key(metadata['storage_key']), Body=b'changed')
    with pytest.raises(ServerError) as failure:
        storage.resolve(record['file_id'])
    assert failure.value.code == 'content_mismatch'
    assert not path.exists()
    storage.remove(record['file_id'])
    assert client.list_multipart_uploads(Bucket='nndeploy-tests').get('Uploads', []) == []


def test_s3_database_failure_removes_remote_object(remote, monkeypatch):
    storage, client = remote
    before = client.list_objects_v2(Bucket='nndeploy-tests').get('KeyCount', 0)
    def fail(record):
        raise RuntimeError('DB unavailable')
    monkeypatch.setattr(storage.db, 'create_file', fail)
    with pytest.raises(RuntimeError):
        storage.save(io.BytesIO(picture()), 'image/png')
    assert client.list_objects_v2(Bucket='nndeploy-tests').get('KeyCount', 0) == before
    assert {path.name for path in storage.root.iterdir()} == {'.server-id'}


def test_s3_failed_part_aborts_multipart(remote, monkeypatch):
    from botocore.exceptions import ClientError
    storage, client = remote
    def fail(**kwargs):
        raise ClientError({'Error': {'Code': 'InternalError', 'Message': 'injected'}}, 'UploadPart')
    monkeypatch.setattr(storage.backend.client, 'upload_part', fail)
    with pytest.raises(Exception):
        storage.save(io.BytesIO(b'x' * (9 * 1024**2)), 'application/octet-stream')
    assert client.list_multipart_uploads(Bucket='nndeploy-tests').get('Uploads', []) == []
    assert {path.name for path in storage.root.iterdir()} == {'.server-id'}


@pytest.mark.parametrize('provider,region,endpoint', [
    ('cos', 'ap-shanghai', None),
    ('oss', 'cn-hangzhou', 'https://oss-cn-hangzhou.aliyuncs.com'),
    ('obs', None, 'https://obs.cn-north-4.myhuaweicloud.com'),
])
def test_cloud_sdk_offline_signing_only(provider, region, endpoint, monkeypatch):
    """Checks actual installed SDK calls; this is NOT cloud compatibility acceptance."""
    from urllib.parse import urlsplit, parse_qs
    from nndeploy.server.storage.objects import ObjectBackend, StorageConfig
    pytest.importorskip({'cos': 'qcloud_cos', 'oss': 'alibabacloud_oss_v2', 'obs': 'obs'}[provider])
    monkeypatch.setenv('NNDEPLOY_OFFLINE_ACCESS', 'test-access')
    monkeypatch.setenv('NNDEPLOY_OFFLINE_SECRET', 'test-secret')
    config = StorageConfig(backend=provider, bucket='nndeploy-test-1234567890', region=region, endpoint=endpoint,
                           access_key_env='NNDEPLOY_OFFLINE_ACCESS', secret_key_env='NNDEPLOY_OFFLINE_SECRET')
    backend = ObjectBackend(config)
    try:
        url = urlsplit(backend.signed_get('file.png', 60))
        assert url.scheme == 'https' and url.path.endswith('/nndeploy/file.png')
        assert parse_qs(url.query)
        with pytest.raises(ValueError):
            backend.signed_get('file.png', 100000)
    finally:
        backend.close()


def test_s3_cleanup_failure_is_retried_from_local_journal(remote, monkeypatch):
    storage, client = remote
    create, delete = storage.db.create_file, storage.backend.delete
    def fail(*args, **kwargs):
        raise RuntimeError('temporary outage')
    monkeypatch.setattr(storage.db, 'create_file', fail)
    monkeypatch.setattr(storage.backend, 'delete', fail)
    with pytest.raises(RuntimeError):
        storage.save(io.BytesIO(picture()), 'image/png')
    assert len(list(storage.root.glob('.pending-*.json'))) == 1
    monkeypatch.setattr(storage.db, 'create_file', create)
    monkeypatch.setattr(storage.backend, 'delete', delete)
    assert storage.recover_pending() == 1
    assert {path.name for path in storage.root.iterdir()} == {'.server-id'}


def test_s3_uncertain_commit_preserves_recorded_content(remote, monkeypatch):
    storage, client = remote
    create = storage.db.create_file
    recorded = []
    def commit_then_disconnect(record):
        create(record)
        recorded.append(record)
        raise RuntimeError('commit acknowledgement lost')
    monkeypatch.setattr(storage.db, 'create_file', commit_then_disconnect)
    with pytest.raises(RuntimeError):
        storage.save(io.BytesIO(picture()), 'image/png')
    identity = recorded[0]['file_id']
    path, _ = storage.resolve(identity)
    path.unlink()
    assert storage.resolve(identity)[0].read_bytes() == picture()
    storage.remove(identity)
