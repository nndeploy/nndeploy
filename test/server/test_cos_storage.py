"""COS multipart lifecycle using the installed SDK with injected transport calls."""
import pytest

from nndeploy.server.storage.objects import ObjectBackend, StorageConfig


@pytest.fixture
def cos_backend(monkeypatch):
    pytest.importorskip('qcloud_cos')
    monkeypatch.setenv('COS_TEST_ACCESS', 'offline-access')
    monkeypatch.setenv('COS_TEST_SECRET', 'offline-secret')
    backend = ObjectBackend(StorageConfig(backend='cos', bucket='test-1234567890', region='ap-beijing',
        access_key_env='COS_TEST_ACCESS', secret_key_env='COS_TEST_SECRET'))
    yield backend
    backend.close()


def test_failed_cos_multipart_aborts_its_upload(cos_backend, tmp_path, monkeypatch):
    path = tmp_path / 'large.bin'
    path.write_bytes(b'x' * (9 * 1024**2))
    client = cos_backend.client
    monkeypatch.setattr(client, '_get_resumable_uploadid', lambda *args: None)
    monkeypatch.setattr(client, 'create_multipart_upload', lambda **kwargs: {'UploadId': 'owned-upload'})
    aborted = []
    monkeypatch.setattr(client, 'abort_multipart_upload', lambda **kwargs: aborted.append(kwargs))
    def fail(**kwargs):
        raise RuntimeError('Injected part failure')
    monkeypatch.setattr(client, 'upload_part', fail)
    with pytest.raises(Exception):
        cos_backend.put_file('large.bin', path, 'application/octet-stream')
    assert aborted == [{'Bucket': 'test-1234567890', 'Key': 'nndeploy/large.bin', 'UploadId': 'owned-upload'}]


def test_cos_complete_failure_aborts_and_preserves_original_error(cos_backend, tmp_path, monkeypatch, caplog):
    path = tmp_path / 'large.bin'
    path.write_bytes(b'x' * (9 * 1024**2))
    client = cos_backend.client
    monkeypatch.setattr(client, 'create_multipart_upload', lambda **kwargs: {'UploadId': 'owned-upload'})
    monkeypatch.setattr(client, 'upload_part', lambda **kwargs: {'ETag': str(kwargs['PartNumber'])})
    completed = []
    def complete(**kwargs):
        completed.append(kwargs['MultipartUpload'])
        raise RuntimeError('Completion response lost')
    def abort(**kwargs):
        raise RuntimeError('private-signed-url')
    monkeypatch.setattr(client, 'complete_multipart_upload', complete)
    monkeypatch.setattr(client, 'abort_multipart_upload', abort)
    with pytest.raises(RuntimeError, match='Completion response lost'):
        cos_backend.put_file('large.bin', path, 'application/octet-stream')
    assert completed == [{'Part': [{'PartNumber': 1, 'ETag': '1'}, {'PartNumber': 2, 'ETag': '2'}]}]
    assert 'COS multipart abort failed' in caplog.text
    assert 'private-signed-url' not in caplog.text
