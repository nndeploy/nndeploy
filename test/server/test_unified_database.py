"""All frontends and their workers use one selected SQL database."""
import asyncio
import importlib
import io
import json
import os
from types import SimpleNamespace
import uuid

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import inspect

from nndeploy.server import Server
from nndeploy.server.app import _editor_storage_options
from nndeploy.server.config import suggest_config
from nndeploy.server.db import DB
from nndeploy.server.task_queue import ExecutionStatus
from nndeploy.server.template import WorkflowTemplateManager
from test_server_api import image_graph, text_graph


URLS = [None] + json.loads(os.environ.get('NNDEPLOY_TEST_DATABASE_URLS', '[]'))


@pytest.fixture(params=URLS, ids=lambda url: url.split(':')[0] if url else 'sqlite')
def database_url(request):
    return request.param


def test_editor_api_and_process_worker_share_persistent_database(tmp_path, monkeypatch, database_url):
    root = tmp_path / 'resources'
    monkeypatch.setattr(WorkflowTemplateManager, 'init_templates', lambda: tmp_path)
    editor = Server(SimpleNamespace(resources=str(root), debug=True), database_url=database_url)
    selected_url = editor.db.url.render_as_string(hide_password=False)
    with TestClient(editor.app) as client:
        assert editor.storage.db is editor.db
        assert _editor_storage_options(editor)['database_url'] == selected_url
        graph_path = root / 'workflow' / 'saved.json'
        graph_path.write_text(json.dumps(text_graph()))
        workflow_id = editor.db.upsert_workflow_by_path(graph_path, 'Saved workflow')
        identity = client.post('/api/queue', json=text_graph()).json()['result']['task_id']
        index, _ = editor.queue.get(timeout=1)
        editor.queue.task_done(index, ExecutionStatus(True, 'Completed'), {'sink': {'String': ['saved']}}, {})

    workflow = image_graph()
    config = suggest_config(workflow)
    config['execution'] = {'isolation': 'process'}
    server = Server(resources=root, database_url=database_url, workflow=workflow, config=config)
    with TestClient(server.app) as client:
        assert server.db.url == editor.db.url
        assert server.db.get_workflow_name(workflow_id) == 'Saved workflow'
        worker = server.entries['image']['workers'][0]
        assert worker.pid != os.getpid()
        assert worker.storage_options['database_url'] == selected_url
        data = io.BytesIO()
        Image.new('RGB', (4, 6), (12, 23, 34)).save(data, format='PNG')
        uploaded = client.post('/v1/files', files={'file': ('image.png', data.getvalue(), 'image/png')})
        assert uploaded.status_code == 201, uploaded.text
        response = client.post('/v1/server/image/deploy', json={
            'inputs': {'decode': {'file_id': uploaded.json()['file_id']}}})
        assert response.status_code == 200, response.text
        result = response.json()
        file_id = result['outputs']['encode']['file_id']
        assert server.db.get_file(file_id) is not None
        assert server.db.get_task(result['task_id'])['state'] == 'succeeded'

    restored = Server(SimpleNamespace(resources=str(root), debug=True), database_url=database_url)
    with TestClient(restored.app) as client:
        history = client.get('/api/queue/task/' + identity)
        assert history.status_code == 200, history.text
        assert history.json()['result']['outputs'] == {'sink': 'saved'}
        assert restored.db.get_task(result['task_id'])['state'] == 'succeeded'
        content = client.get('/v1/files/' + file_id + '/content')
        assert content.status_code == 200, content.text
        with Image.open(io.BytesIO(content.content)) as output:
            assert output.size == (4, 6) and output.getpixel((0, 0)) == (12, 23, 34)
        assert {'workflows', 'server_tasks', 'server_files', 'server_editor_history'} <= set(inspect(restored.db.engine).get_table_names())
    expected = [] if database_url else [root / 'db' / 'nndeploy.db']
    assert sorted(root.rglob('*.db')) == expected


def test_data_directory_cannot_select_another_database(tmp_path, database_url):
    root = tmp_path / 'resources'
    task_id = uuid.uuid4().hex
    first = Server(resources=root, data_dir=tmp_path / 'first-files', database_url=database_url)
    try:
        first.db.create_task(task_id, 'shared', 1)
    finally:
        asyncio.run(first.close())
    second = Server(resources=root, data_dir=tmp_path / 'second-files', database_url=database_url)
    try:
        assert second.data_dir != first.data_dir
        assert second.db.url == first.db.url
        assert second.db.get_task(task_id)['name'] == 'shared'
        assert second.storage.db is second.db
    finally:
        asyncio.run(second.close())
    assert sorted(tmp_path.rglob('*.db')) == ([] if database_url else [root / 'db' / 'nndeploy.db'])


def test_default_resource_root_selects_canonical_database(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    server = Server()
    try:
        assert server.db.db_path == tmp_path / 'resources' / 'db' / 'nndeploy.db'
        assert server.data_dir == tmp_path / 'resources' / 'server'
    finally:
        asyncio.run(server.close())
    assert list(tmp_path.rglob('*.db')) == [server.db.db_path]


@pytest.mark.parametrize('editor', [False, True])
@pytest.mark.parametrize('url', ['mysql+pymysql://localhost/nndeploy', 'postgresql+psycopg://localhost/nndeploy'])
def test_database_failure_never_falls_back_to_sqlite(tmp_path, monkeypatch, editor, url):
    attempts = []

    class UnavailableDatabase:
        _validate_retention = staticmethod(DB._validate_retention)

        def __init__(self, location, **kwargs):
            attempts.append(location)

        def init_schema(self):
            raise RuntimeError('database unavailable')

    monkeypatch.setattr(importlib.import_module('nndeploy.server.server'), 'DB', UnavailableDatabase)
    args = SimpleNamespace(resources=str(tmp_path), debug=True) if editor else None
    with pytest.raises(RuntimeError, match='database unavailable'):
        Server(args, resources=tmp_path, database_url=url)
    assert attempts == [url]
    assert not list(tmp_path.rglob('*.db'))
