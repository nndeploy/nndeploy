"""Exercise the real editor routers, management sessions and progress sockets."""
import queue
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest
from starlette.websockets import WebSocketDisconnect

from nndeploy.server import Server
from test_auth import credentials
from test_server_api import text_graph


@pytest.fixture
def editor(tmp_path, monkeypatch, credentials):
    from nndeploy.server.template import WorkflowTemplateManager
    monkeypatch.setattr(WorkflowTemplateManager, 'init_templates', lambda: tmp_path)
    config, headers = credentials
    server = Server(SimpleNamespace(resources=str(tmp_path), debug=True), queue.Queue(), queue.Queue(), queue.Queue(),
                    admin_token='editor-test', auth_config=config)
    return server, headers


ADMIN = {'Authorization': 'Bearer editor-test'}
ORIGIN = {'Origin': 'http://testserver'}


def test_every_editor_http_route_requires_management_before_parsing(editor):
    server, headers = editor
    with TestClient(server.app) as client:
        paths = [(method, route.path) for route in server.app.routes if route.path.startswith('/api/') and
                 route.path != '/api/auth/session' for method in getattr(route, 'methods', [])]
        assert len(paths) >= 25
        for method, path in paths:
            assert client.request(method, path).status_code == 401, (method, path)
            assert client.request(method, path, headers=headers['alice']).status_code == 403, (method, path)
        assert client.get('/api/resources', headers=ADMIN).status_code == 200
        assert client.get('/api/resources', headers=ADMIN).headers['cache-control'] == 'no-store'
        assert client.post('/api/auth/session', headers=headers['alice']).status_code == 403
        assert client.get('/api/auth/session').json()['authenticated'] is False


def test_browser_session_covers_editor_management_but_not_public_api(editor):
    server, headers = editor
    with TestClient(server.app) as client:
        login = client.post('/api/auth/session', headers=dict(ADMIN, **ORIGIN))
        assert login.status_code == 200, login.text
        cookie = login.headers['set-cookie']
        assert 'HttpOnly' in cookie and 'SameSite=strict' in cookie and 'Path=/api' in cookie
        token = client.cookies.get(server.editor_access.cookie_name)
        assert token not in repr(server.editor_access.sessions)
        assert client.get('/api/workflows').status_code == 200
        assert client.get('/v1/models').status_code == 401
        assert client.get('/api/resources', headers={'Authorization': 'Bearer wrong'}).status_code == 401
        body = {'id': '', 'businessContent': text_graph()}
        assert client.post('/api/workflow/server/start', json=body).status_code == 403
        assert client.post('/api/workflow/server/start', json=body, headers={'Origin': 'https://evil.example'}).status_code == 403
        started = client.post('/api/workflow/server/start', json=body, headers=ORIGIN)
        assert started.status_code == 200, started.text
        name = started.json()['name']
        assert client.get('/api/server/' + name).status_code == 200
        assert client.post('/api/server/' + name + '/stop', headers=ORIGIN).status_code == 200
        assert client.delete('/api/auth/session').status_code == 403
        assert client.delete('/api/auth/session', headers=ORIGIN).status_code == 200
        assert client.get('/api/resources').status_code == 401
        client.cookies.set(server.editor_access.cookie_name, token, path='/api')
        assert client.get('/api/resources').status_code == 401


def test_sessions_expire_rotate_and_are_bounded(editor):
    server, _ = editor
    server.editor_access.max_sessions = 1
    with TestClient(server.app, base_url='https://testserver') as client:
        secure_origin = {'Origin': 'https://testserver'}
        login = client.post('/api/auth/session', headers=dict(ADMIN, **secure_origin))
        assert 'Secure' in login.headers['set-cookie']
        old = client.cookies.get(server.editor_access.cookie_name)
        assert client.post('/api/auth/session', headers=dict(ADMIN, **secure_origin)).status_code == 200
        fresh = client.cookies.get(server.editor_access.cookie_name)
        assert old != fresh and len(server.editor_access.sessions) == 1
        client.cookies.clear()
        assert client.post('/api/auth/session', headers=dict(ADMIN, **secure_origin)).status_code == 429
        client.cookies.set(server.editor_access.cookie_name, old, path='/api')
        assert client.get('/api/resources').status_code == 401
        client.cookies.clear()
        client.cookies.set(server.editor_access.cookie_name, fresh, path='/api')
        assert client.get('/api/resources').status_code == 200
        for key in server.editor_access.sessions:
            server.editor_access.sessions[key] = 0
        assert client.get('/api/resources').status_code == 401
        assert client.post('/api/auth/session', headers=dict(ADMIN, **secure_origin)).status_code == 200


@pytest.mark.parametrize('revoke', ['logout', 'expire'])
def test_progress_websocket_checks_origin_and_revokes_open_connections(editor, revoke):
    server, headers = editor
    with TestClient(server.app) as client:
        for auth in ({}, headers['alice'], {'Authorization': 'Bearer wrong'}, dict(ADMIN, Origin='https://evil.example')):
            with pytest.raises(WebSocketDisconnect) as error:
                with client.websocket_connect('/api/ws/progress', headers=auth):
                    pytest.fail('Unauthorized socket accepted')
            assert error.value.code == 1008
        assert client.post('/api/auth/session', headers=dict(ADMIN, **ORIGIN)).status_code == 200
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect('/api/ws/progress'):
                pytest.fail('Cookie socket without Origin accepted')
        with client.websocket_connect('/api/ws/progress', headers=ORIGIN) as ws:
            ws.send_json({'type': 'bind', 'task_id': 'example'})
            client.portal.call(server._broadcast, {'type': 'probe'})
            assert ws.receive_json() == {'type': 'probe'}
            if revoke == 'logout':
                assert client.delete('/api/auth/session', headers=ORIGIN).status_code == 200
            else:
                for key in server.editor_access.sessions:
                    server.editor_access.sessions[key] = 0
            with pytest.raises(WebSocketDisconnect) as error:
                ws.receive_json()
            assert error.value.code == 1008
        assert not server.sockets and not server.ws_task_map
        assert not server.task_ws_map


def test_anonymous_local_editor_is_compatible_but_cross_origin_is_denied(tmp_path, monkeypatch):
    from nndeploy.server.template import WorkflowTemplateManager
    monkeypatch.setattr(WorkflowTemplateManager, 'init_templates', lambda: tmp_path)
    server = Server(SimpleNamespace(resources=str(tmp_path), debug=True), queue.Queue(), queue.Queue(), queue.Queue())
    with TestClient(server.app) as client:
        assert client.get('/api/auth/session').json()['required'] is False
        assert client.get('/api/workflows').status_code == 200
        assert client.get('/api/resources', headers={'Origin': 'https://evil.example'}).status_code == 403
        assert client.post('/api/queue/flush', headers={'Origin': 'https://evil.example'}).status_code == 403
        with client.websocket_connect('/api/ws/progress') as ws:
            client.portal.call(server._broadcast, {'type': 'probe'})
            assert ws.receive_json() == {'type': 'probe'}


def test_explicit_cors_origins_and_api_only_has_no_editor_sessions(tmp_path):
    with pytest.raises(ValueError, match='Origins'):
        Server(resources=tmp_path, data_dir=tmp_path, cors_origins=['*'])
    server = Server(resources=tmp_path, data_dir=tmp_path, cors_origins=['https://editor.example'])
    with TestClient(server.app) as client:
        headers = {'Origin': 'https://editor.example', 'Access-Control-Request-Method': 'POST',
                   'Access-Control-Request-Headers': 'Authorization,Content-Type'}
        response = client.options('/v1/files', headers=headers)
        assert response.status_code == 200 and response.headers['access-control-allow-origin'] == 'https://editor.example'
        assert response.headers['access-control-allow-credentials'] == 'true'
        assert client.options('/v1/files', headers=dict(headers, Origin='https://evil.example')).status_code == 400
        assert client.get('/api/auth/session').status_code == 404


def test_editor_file_access_cannot_escape_resource_roots(editor, tmp_path):
    server, _ = editor
    outside = tmp_path.parent / (tmp_path.name + '-private.txt')
    outside.write_text('outside the editor')
    inside = tmp_path / 'images' / 'inside.txt'
    inside.write_text('inside the editor')
    symlink = tmp_path / 'images' / 'escape.txt'
    symlink.symlink_to(outside)
    try:
        with TestClient(server.app) as client:
            for endpoint in ('/api/preview', '/api/download'):
                for path in (outside, symlink):
                    assert client.get(endpoint, params={'file_path': str(path)}, headers=ADMIN).status_code == 403
                response = client.get(endpoint, params={'file_path': str(inside)}, headers=ADMIN)
                assert response.status_code == 200 and response.content == b'inside the editor'
                assert response.headers['x-content-type-options'] == 'nosniff'
                assert 'sandbox' in response.headers['content-security-policy']
            for endpoint, suffix in (('/api/workflow/upload', '.json'), ('/api/nodes/upload', '.py')):
                for name in ('../escape' + suffix, '/tmp/escape' + suffix, 'nested/escape' + suffix):
                    response = client.post(endpoint, files={'file': (name, b'{}')}, headers=ADMIN)
                    assert response.status_code == 400, response.text
            for name in ('../escape', '/tmp/escape', 'nested/escape', '..'):
                response = client.post('/api/workflow/save', json={'businessContent': {'name_': name}}, headers=ADMIN)
                assert response.status_code == 400, response.text
    finally:
        outside.unlink()


@pytest.mark.parametrize('message,code', [('not json', 1008), ('x' * 16385, 1009)])
def test_editor_websocket_rejects_invalid_or_oversized_messages(editor, message, code):
    server, _ = editor
    with TestClient(server.app) as client:
        with client.websocket_connect('/api/ws/progress', headers=ADMIN) as ws:
            ws.send_text(message)
            with pytest.raises(WebSocketDisconnect) as error:
                ws.receive_json()
            assert error.value.code == code
        assert not server.sockets


def test_independent_servers_do_not_share_or_overwrite_browser_sessions(editor, tmp_path):
    server, _ = editor
    other = Server(SimpleNamespace(resources=str(tmp_path / 'other'), debug=True), queue.Queue(), queue.Queue(), queue.Queue(),
                   admin_token='other-test')
    assert server.editor_access.cookie_name != other.editor_access.cookie_name
    with TestClient(server.app) as first, TestClient(other.app) as second:
        assert first.post('/api/auth/session', headers=dict(ADMIN, **ORIGIN)).status_code == 200
        second.cookies.update(first.cookies)
        assert second.get('/api/resources').status_code == 401
        assert second.post('/api/auth/session', headers={'Authorization': 'Bearer other-test', **ORIGIN}).status_code == 200
        first.cookies.update(second.cookies)
        assert first.get('/api/resources').status_code == second.get('/api/resources').status_code == 200
        assert first.delete('/api/auth/session', headers=ORIGIN).status_code == 200
        assert first.get('/api/resources').status_code == 401
        assert second.get('/api/resources').status_code == 200


def test_editor_access_disabled_without_management_token(tmp_path, credentials, monkeypatch):
    from nndeploy.server.template import WorkflowTemplateManager
    monkeypatch.setattr(WorkflowTemplateManager, 'init_templates', lambda: tmp_path)
    config, headers = credentials
    server = Server(SimpleNamespace(resources=str(tmp_path), debug=True), queue.Queue(), queue.Queue(), queue.Queue(), auth_config=config)
    with TestClient(server.app) as client:
        assert client.get('/api/auth/session').json() == {'required': True, 'authenticated': False, 'management_available': False}
        assert client.get('/api/resources', headers=headers['alice']).status_code == 403
        assert client.get('/api/resources').status_code == 401
        assert client.post('/api/auth/session', headers=headers['alice']).status_code == 403


def test_editor_websocket_binary_message_is_rejected(editor):
    server, _ = editor
    with TestClient(server.app) as client:
        with client.websocket_connect('/api/ws/progress', headers=ADMIN) as ws:
            ws.send_bytes(b'not a progress binding')
            with pytest.raises(WebSocketDisconnect) as error:
                ws.receive_json()
            assert error.value.code == 1003


def test_editor_serves_selected_local_build_without_downloading(tmp_path, monkeypatch):
    from nndeploy.server.template import WorkflowTemplateManager
    from nndeploy.server.frontend import FrontendManager
    from nndeploy.server.app import cli
    monkeypatch.setattr(WorkflowTemplateManager, 'init_templates', lambda: tmp_path)
    def forbidden_download(*args):
        pytest.fail('Explicit local build must not download a frontend release')
    monkeypatch.setattr(FrontendManager, 'init_frontend', forbidden_download)
    frontend = tmp_path / 'frontend' / 'dist'
    frontend.mkdir(parents=True)
    (frontend / 'index.html').write_text('<html><body>local editor build</body></html>')
    args = cli(['--resources', str(tmp_path / 'resources'), '--front-end-dir', str(frontend.parent)])
    server = Server(args, queue.Queue(), queue.Queue(), queue.Queue(), admin_token='editor-test')
    with TestClient(server.app) as client:
        assert 'local editor build' in client.get('/design').text
        assert client.get('/api/resources').status_code == 401
        assert client.post('/api/auth/session', headers=ADMIN).status_code == 200
