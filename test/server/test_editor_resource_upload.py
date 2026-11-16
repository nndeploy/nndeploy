"""Browser model uploads use editor sessions and the existing Storage."""
import asyncio
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from nndeploy.server import Server
from test_editor_history import location
from test_model_resources import payload
from test_auth import credentials


@pytest.mark.parametrize('directory', [False, True])
def test_editor_session_uploads_managed_resources_without_opening_public_api(tmp_path, location, directory, credentials):
    server = Server(SimpleNamespace(resources=str(tmp_path), debug=True), database_url=location,
                    admin_token='editor-resource-test', auth_config=credentials[0])
    data, media = payload(directory)
    with TestClient(server.app) as client:
        upload = lambda headers=None: client.post('/api/files/upload',
            files={'file': ('model.zip' if directory else 'model.bin', data, media)}, headers=headers)
        assert upload().status_code == 401
        assert client.post('/api/auth/session', headers={'Authorization': 'Bearer editor-resource-test'}).status_code == 200
        assert upload().status_code == 403
        assert upload({'Origin': 'https://other.example'}).status_code == 403
        response = upload({'Origin': 'http://testserver'})
        assert response.status_code == 201, response.text
        record = response.json()['result']
        assert record['file_id'].startswith('file_') and record['size'] == len(data)
        assert record['filename'] == ('model.zip' if directory else 'model.bin')
        assert 'saved_path' not in record
        stored = server.db.get_file(record['file_id'])
        assert stored['owner'] == 'editor:' + server.config_scope
        with server.storage.open(record['file_id']) as source:
            assert source.read() == data
        assert client.get('/v1/files/' + record['file_id']).status_code == 401


def test_legacy_upload_with_path_keeps_original_response(tmp_path, location):
    server = Server(SimpleNamespace(resources=str(tmp_path), debug=True), database_url=location)
    with TestClient(server.app) as client:
        response = client.post('/api/files/upload', params={'file_path': str(tmp_path / 'models')},
                               files={'file': ('weights.bin', b'weights', 'application/octet-stream')})
        assert response.status_code == 201, response.text
        assert response.json()['result']['saved_path'] == str(tmp_path / 'models/weights.bin')
        assert 'file_id' not in response.json()['result']


def test_managed_upload_checks_media_and_storage_limit(tmp_path, location):
    server = Server(SimpleNamespace(resources=str(tmp_path), debug=True), database_url=location,
                    storage_config={'max_bytes': 32})
    with TestClient(server.app) as client:
        response = client.post('/api/files/upload', files={'file': ('model.zip', b'not zip', 'application/zip')})
        assert response.status_code == 422, response.text
        response = client.post('/api/files/upload', files={'file': ('model.bin', b'x' * 33, 'application/octet-stream')})
        assert response.status_code == 413, response.text


@pytest.mark.parametrize('route', ['/api/files/upload', '/v1/files'])
def test_cancelled_upload_waits_for_writer_then_removes_unreturned_file(tmp_path, location, monkeypatch, route):
    import httpx
    server = Server(SimpleNamespace(resources=str(tmp_path), debug=True), database_url=location)
    entered, release = threading.Event(), threading.Event()
    created = []
    original = server.storage.save
    def writing(stream, *args, **kwargs):
        entered.set()
        assert release.wait(5)
        assert not stream.closed
        record = original(stream, *args, **kwargs)
        created.append(record)
        return record
    monkeypatch.setattr(server.storage, 'save', writing)
    async def scenario():
        await server.start()
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://testserver') as client:
                request = asyncio.create_task(client.post(route, files={'file': ('weights.bin', b'weights', 'application/octet-stream')}))
                try:
                    assert await asyncio.to_thread(entered.wait, 3)
                    request.cancel()
                    await asyncio.sleep(.02)
                    assert not request.done()
                finally:
                    release.set()
                with pytest.raises(asyncio.CancelledError):
                    await request
                assert len(created) == 1
                assert server.db.get_file(created[0]['file_id']) is None
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())
