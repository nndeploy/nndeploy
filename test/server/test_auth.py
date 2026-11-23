"""Authorization exercised through HTTP, transactional DB writes and native Graphs."""
import io
import uuid

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from nndeploy.server import Server
from nndeploy.server.dependencies import Authentication
from nndeploy.server.errors import ServerError
from test_db import db
from test_server_api import image_graph


@pytest.fixture
def credentials(monkeypatch):
    tokens = {name: uuid.uuid4().hex for name in ('alice', 'bob', 'reader')}
    for name, value in tokens.items():
        monkeypatch.setenv('TEST_KEY_' + name.upper(), value)
    config = {'keys': [dict(subject=name, token_env='TEST_KEY_' + name.upper(), entries=['echo', 'image'],
        permissions=['deploy', 'tasks:read', 'tasks:cancel', 'files:read', 'files:write'] if name != 'reader' else ['tasks:read', 'files:read'])
        for name in tokens]}
    headers = {name: {'Authorization': 'Bearer ' + value} for name, value in tokens.items()}
    return config, headers


def test_identity_scopes_idempotency_tasks_and_entries(db, tmp_path, credentials):
    config, headers = credentials
    server = Server(name='echo', deploy=lambda value: value, resources=tmp_path, data_dir=tmp_path,
                    database_url=db.url.render_as_string(hide_password=False), auth_config=config, admin_token='admin')
    server.add(name='private', deploy=lambda value: value)
    with TestClient(server.app) as client:
        assert client.get('/health/live').status_code == 200
        assert client.get('/v1/models').status_code == 401
        assert client.get('/v1/models', headers={'Authorization': 'Bearer wrong'}).status_code == 401
        models = client.get('/v1/models', headers=headers['alice']).json()['models']
        assert [model['name'] for model in models] == ['echo']
        assert client.get('/v1/server/private/schema', headers=headers['alice']).status_code == 403
        assert client.post('/v1/server/private/deploy', json={}, headers=headers['alice']).status_code == 403
        assert client.post('/v1/server/echo/deploy', json={}, headers=headers['reader']).status_code == 403
        assert client.post('/api/server/echo/stop', headers=headers['alice']).status_code == 401
        key = uuid.uuid4().hex
        results = {}
        for person in ('alice', 'bob'):
            request_headers = dict(headers[person], **{'Idempotency-Key': key})
            result = client.post('/v1/server/echo/deploy', json={'inputs': {'person': person}}, headers=request_headers)
            assert result.status_code == 200, result.text
            results[person] = result.json()['task_id']
            again = client.post('/v1/server/echo/deploy', json={'inputs': {'person': person}}, headers=request_headers)
            assert again.json()['task_id'] == results[person]
            assert client.post('/v1/server/echo/deploy', json={}, headers=request_headers).status_code == 409
        assert results['alice'] != results['bob']
        for suffix in ('', '/cancel'):
            method = client.get if not suffix else client.post
            assert method('/v1/tasks/' + results['alice'] + suffix, headers=headers['bob']).status_code == 404
        own = client.get('/v1/tasks/' + results['alice'], headers=headers['alice'])
        assert own.status_code == 200 and 'owner' not in own.json()
        assert db.get_task(results['alice'])['owner'] == 'alice'
    with TestClient(Server(resources=tmp_path, data_dir=tmp_path, database_url=db.url.render_as_string(hide_password=False), auth_config=config).app) as client:
        assert client.get('/v1/tasks/' + results['alice'], headers=headers['alice']).status_code == 200
        assert client.get('/v1/tasks/' + results['alice'], headers=headers['bob']).status_code == 404


@pytest.mark.parametrize('isolation', ['thread', 'process'])
def test_native_graph_output_files_keep_owner_and_forbid_cross_user_references(db, tmp_path, credentials, isolation):
    config, headers = credentials
    server = Server(workflow=image_graph(), resources=tmp_path, data_dir=tmp_path, auth_config=config,
                    database_url=db.url.render_as_string(hide_password=False))
    server.entries['image']['config'].execution.isolation = isolation
    server.add(name='echo', deploy=lambda value: value)
    picture = io.BytesIO()
    Image.new('RGB', (3, 4), (11, 22, 33)).save(picture, format='PNG')
    with TestClient(server.app) as client:
        uploaded = client.post('/v1/files', files={'file': ('image.png', picture.getvalue(), 'image/png')}, headers=headers['alice'])
        assert uploaded.status_code == 201, uploaded.text
        identity = uploaded.json()['file_id']
        assert 'owner' not in uploaded.json()
        body = {'inputs': {'decode': {'file_id': identity}}}
        assert client.post('/v1/server/image/deploy', json=body, headers=headers['bob']).status_code == 422
        # The transactional check also covers nested references in untyped code inputs.
        assert client.post('/v1/server/echo/deploy', json={'inputs': {'nested': [{'file_id': identity}]}}, headers=headers['bob']).status_code == 422
        result = client.post('/v1/server/image/deploy', json=body, headers=headers['alice'])
        assert result.status_code == 200, result.text
        output = result.json()['outputs']['encode']['file_id']
        assert db.get_file(output)['owner'] == 'alice'
        for file_id in (identity, output):
            for suffix in ('', '/content'):
                assert client.get('/v1/files/' + file_id + suffix, headers=headers['bob']).status_code == 404
                assert client.get('/v1/files/' + file_id + suffix, headers=headers['alice']).status_code == 200
            assert client.post('/v1/files/' + file_id + '/download-url', headers=headers['bob']).status_code == 404
            assert client.delete('/v1/files/' + file_id, headers=headers['bob']).status_code == 404
        assert client.delete('/v1/files/' + output, headers=headers['alice']).status_code == 409
        unreferenced = client.post('/v1/files', files={'file': ('note.txt', b'private', 'text/plain')}, headers=headers['alice']).json()['file_id']
        assert client.delete('/v1/files/' + unreferenced, headers=headers['alice']).status_code == 200


def test_keys_fail_closed_and_never_retain_raw_tokens(monkeypatch):
    config = {'keys': [{'subject': 'alice', 'token_env': 'TEST_AUTH_MISSING'}]}
    monkeypatch.delenv('TEST_AUTH_MISSING', raising=False)
    with pytest.raises(ValueError, match='environment'):
        Authentication(config)
    token = uuid.uuid4().hex
    monkeypatch.setenv('TEST_AUTH_MISSING', token)
    auth = Authentication(config)
    assert token not in repr(auth.__dict__)
    assert auth.authenticate('Bearer ' + token).subject == 'alice'
    with pytest.raises(ServerError):
        auth.authenticate('')
    with pytest.raises(ValueError, match='distinct'):
        Authentication(config, admin_token=token)
