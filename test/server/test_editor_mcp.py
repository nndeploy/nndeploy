"""Editor MCP configuration survives updates and runtime restoration."""
import queue
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from nndeploy.server import Server
from test_server_api import text_graph
from test_mcp import rpc

ADMIN = {'Authorization': 'Bearer editor-mcp'}


def make_editor(tmp_path, monkeypatch, enabled=True):
    from nndeploy.server.template import WorkflowTemplateManager
    monkeypatch.setattr(WorkflowTemplateManager, 'init_templates', lambda: tmp_path)
    return Server(SimpleNamespace(resources=str(tmp_path), debug=True), queue.Queue(), queue.Queue(), queue.Queue(),
                  admin_token='editor-mcp', mcp_config={'enabled': enabled, 'allowed_hosts': ['testserver']})


def start(client, tool, identity='', name='echo', graph=None):
    return client.post('/api/workflow/server/start', headers=ADMIN, json={
        'id': identity, 'businessContent': graph or text_graph(), 'name': name, 'mcp': tool})


def tools(client):
    return {tool['name']: tool for tool in rpc(client, 'tools/list', token='editor-mcp')['tools']}


def test_editor_enable_update_disable_and_restore(tmp_path, monkeypatch):
    server = make_editor(tmp_path, monkeypatch)
    tool = {'name': 'echo_tool', 'description': 'Run the text workflow'}
    with TestClient(server.app) as client:
        assert client.get('/api/server/mcp/config').status_code == 401
        assert client.get('/api/server/mcp/config', headers=ADMIN).json() == {'enabled': True, 'path': '/mcp'}
        response = start(client, tool)
        assert response.status_code == 200, response.text
        record = response.json()
        identity = record['workflow_id']
        assert record['mcp'] == tool and record['mcp_path'] == '/mcp'
        assert 'echo_tool' in tools(client)
        loaded = client.get(f'/api/workflow/{identity}/server', headers=ADMIN).json()
        assert loaded['mcp'] == tool
        updated = {'name': 'renamed_tool', 'description': 'Updated description'}
        response = start(client, updated, identity)
        assert response.status_code == 200, response.text
        assert 'echo_tool' not in tools(client)
        assert tools(client)['renamed_tool']['description'] == 'Updated description'
    restored = make_editor(tmp_path, monkeypatch)
    with TestClient(restored.app) as client:
        assert 'renamed_tool' in tools(client)
        assert client.get(f'/api/workflow/{identity}/server', headers=ADMIN).json()['mcp'] == updated
        response = start(client, None, identity)
        assert response.status_code == 200, response.text
        assert response.json()['mcp'] is None
        assert 'renamed_tool' not in tools(client)
        assert client.get(f'/api/workflow/{identity}/server', headers=ADMIN).json()['mcp'] is None
    with TestClient(make_editor(tmp_path, monkeypatch).app) as client:
        assert 'renamed_tool' not in tools(client)


def test_validation_collision_and_failed_reload_preserve_export(tmp_path, monkeypatch):
    server = make_editor(tmp_path, monkeypatch)
    original = {'name': 'original', 'description': 'Original tool'}
    with TestClient(server.app) as client:
        first = start(client, original).json()
        identity = first['workflow_id']
        invalid = start(client, {'name': 'get_task', 'description': 'Reserved'}, identity)
        assert invalid.status_code == 422
        graph = text_graph()
        graph['name_'] = 'second'
        duplicate = start(client, original, name='second', graph=graph)
        assert duplicate.status_code == 409, duplicate.text
        graph = text_graph()
        graph['node_repository_'][0]['key_'] = 'missing.node.Type'
        failed = start(client, {'name': 'replacement', 'description': 'Failed replacement'}, identity, graph=graph)
        assert failed.status_code >= 400, failed.text
        assert 'original' in tools(client) and 'replacement' not in tools(client)
        assert client.get('/api/server/echo', headers=ADMIN).json()['mcp'] == original


def test_disabled_transport_rejects_export_but_allows_normal_start(tmp_path, monkeypatch):
    with TestClient(make_editor(tmp_path, monkeypatch, False).app) as client:
        assert client.get('/api/server/mcp/config', headers=ADMIN).json() == {'enabled': False, 'path': None}
        assert start(client, {'name': 'tool', 'description': 'Export'}).status_code == 409
        response = start(client, None)
        assert response.status_code == 200, response.text
        assert response.json()['mcp_path'] is None


def test_factory_discovery_keeps_export(tmp_path):
    import json
    import nndeploy.dag
    def factory():
        graph = nndeploy.dag.Graph('echo')
        graph.deserialize(json.dumps(text_graph()))
        return graph
    server = Server(data_dir=tmp_path, mcp_config={'enabled': True, 'allowed_hosts': ['testserver']})
    server.add(name='factory', graph_factory=factory, mcp={'name': 'factory_tool', 'description': 'Factory graph'})
    with TestClient(server.app) as client:
        assert 'factory_tool' in {tool['name'] for tool in rpc(client, 'tools/list')['tools']}
        assert server.entries['factory']['config'].mcp.name == 'factory_tool'
