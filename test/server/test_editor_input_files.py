"""Editor Run uses the same managed model files as resident Graphs."""
import copy
import io
import json
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select, update

from nndeploy.server import Server
from nndeploy.server.app import _EditorRuntime
from nndeploy.server.db import editor_file_refs, editor_history, editor_tasks
from nndeploy.server.errors import ServerError
from nndeploy.server.task_queue import ExecutionStatus
from test_editor_history import editor, location
from test_editor_queue_recovery import wait
from test_model_resources import resource_graph, payload as model_payload
from test_db import db
from test_storage import s3_service


def workflow(reference, directory=False):
    return json.loads(resource_graph(reference, directory).serialize())


@pytest.mark.parametrize('outcome', ['succeeded', 'failed', 'cancelled'])
def test_editor_input_survives_queue_and_history_without_becoming_output(tmp_path, location, outcome):
    server = editor(tmp_path / 'editor', location)
    with TestClient(server.app) as client:
        file = server.storage.save(io.BytesIO(b'weights'), 'application/octet-stream')
        reference = {'file_id': file['file_id']}
        graph = workflow(reference)
        key = str(uuid.uuid4())
        response = client.post('/api/queue', json=graph, headers={'Idempotency-Key': key})
        assert response.status_code == 202, response.text
        with pytest.raises(ServerError) as held:
            server.storage.remove(file['file_id'])
        assert held.value.code == 'file_in_use'
        assert client.post('/api/queue', json=graph, headers={'Idempotency-Key': key}).status_code == 202
        with server.db.engine.connect() as connection:
            refs = connection.execute(select(editor_file_refs).where(
                editor_file_refs.c.scope == server.config_scope, editor_file_refs.c.task_id == key)).mappings().all()
        assert len(refs) == 1 and refs[0]['kind'] == 'input'
        idx, task = server.queue.get(timeout=1)
        assert task['graph_json'] == graph
        server.queue.task_done(idx, ExecutionStatus(outcome == 'succeeded', 'Finished',
                               'success' if outcome == 'succeeded' else outcome), {}, {})
        result = client.get('/api/queue/task/' + key).json()['result']
        assert result['state'] == outcome.upper() and result['output_files'] == []
        assert client.get(f'/api/queue/task/{key}/files/{file["file_id"]}').status_code == 404
        with pytest.raises(ServerError):
            server.storage.remove(file['file_id'])
        with server.db.engine.begin() as connection:
            connection.execute(update(editor_history).where(editor_history.c.scope == server.config_scope,
                editor_history.c.task_id == key).values(finished_at=0))
        assert server.queue.history_store.cleanup(60) == [key]
        # Expiring a task must not schedule deletion of a user-uploaded model.
        assert server.db.get_file(file['file_id'])['purge_after'] is None
        assert server.storage.cleanup()['deleted'] == 0
        server.storage.remove(file['file_id'])


def test_editor_rejects_missing_model_atomically_and_ignores_preview_metadata(tmp_path, location):
    server = editor(tmp_path / 'editor', location)
    with TestClient(server.app) as client:
        file = server.storage.save(io.BytesIO(b'weights'), 'application/octet-stream')
        graph = workflow({'file_id': file['file_id']})
        graph['node_repository_'][0]['artifacts'].append({'file_id': 'file_missing'})
        key = str(uuid.uuid4())
        response = client.post('/api/queue', json=graph, headers={'Idempotency-Key': key})
        assert response.status_code == 422, response.text
        assert client.get('/api/queue/task/' + key).status_code == 404
        with server.db.engine.connect() as connection:
            assert connection.execute(select(editor_tasks).where(
                editor_tasks.c.scope == server.config_scope, editor_tasks.c.task_id == key)).first() is None
        server.storage.remove(file['file_id'])
        graph['node_repository_'][0]['artifacts'] = []
        graph['node_repository_'][0]['frontend_params'] = {'preview': {'file_id': 'file_missing'}}
        assert client.post('/api/queue', json=graph).status_code == 202


@pytest.mark.parametrize('directory', [False, True])
@pytest.mark.parametrize('remote', [False, True])
def test_real_editor_materializes_model_before_graph_loading(tmp_path, location, directory, remote, request, monkeypatch):
    resources = tmp_path / 'resources'
    plugins = resources / 'plugin'
    plugins.mkdir(parents=True)
    (plugins / 'model_resource.py').write_text('import test_declared_resources\n')
    args = SimpleNamespace(resources=str(resources), log=str(tmp_path / 'editor.log'), debug=True)
    storage_config = None
    if remote:
        endpoint, environment, _ = request.getfixturevalue('s3_service')
        monkeypatch.setenv('NNDEPLOY_TEST_ACCESS', environment['MINIO_ROOT_USER'])
        monkeypatch.setenv('NNDEPLOY_TEST_SECRET', environment['MINIO_ROOT_PASSWORD'])
        storage_config = dict(backend='s3', bucket='nndeploy-tests', region='us-east-1', endpoint=endpoint,
            access_key_env='NNDEPLOY_TEST_ACCESS', secret_key_env='NNDEPLOY_TEST_SECRET')
    runtime = _EditorRuntime(args)
    server = runtime.start(lambda jobs, plugins, cancellations: Server(
        args, jobs, plugins, cancellations, database_url=location, storage_config=storage_config))
    with TestClient(server.app) as client:
        try:
            data, media = model_payload(directory)
            uploaded = client.post('/v1/files', files={'file': ('model', data, media)})
            assert uploaded.status_code == 201, uploaded.text
            reference = {'file_id': uploaded.json()['file_id']}
            if remote:
                server.storage.resolve(reference['file_id'])[0].unlink()
            graph = workflow(reference, directory)
            original = copy.deepcopy(graph)
            response = client.post('/api/queue', json=graph)
            assert response.status_code == 202, response.text
            identity = response.json()['result']['task_id']
            def completed():
                record = client.get('/api/queue/task/' + identity).json()['result']
                return record if record['state'] in ('SUCCEEDED', 'FAILED') else None
            record = wait(completed)
            assert record['state'] == 'SUCCEEDED', record
            assert record['outputs']['sink'] == ('tokens' if directory else 'weights')
            assert server.queue.history_store.get(identity)['task']['graph_json'] == original
            assert record['output_files'] == []
        finally:
            runtime.close()


@pytest.mark.parametrize('legacy,missing', [(False, False), (True, False), (True, True)])
def test_pending_editor_inputs_are_protected_after_restart(tmp_path, location, legacy, missing):
    path = tmp_path / 'editor'
    first = editor(path, location)
    with TestClient(first.app) as client:
        file = first.storage.save(io.BytesIO(b'weights'), 'application/octet-stream')
        graph = workflow({'file_id': file['file_id']})
        identity = client.post('/api/queue', json=graph).json()['result']['task_id']
        if legacy:
            # Old pending rows had no input references; recovery must establish
            # them before dispatch, or settle failure if the bytes are gone.
            with first.db.engine.begin() as connection:
                connection.execute(delete(editor_file_refs).where(editor_file_refs.c.scope == first.config_scope,
                    editor_file_refs.c.task_id == identity))
            if missing:
                first.storage.remove(file['file_id'])
    restored = editor(path, location)
    with TestClient(restored.app) as client:
        if missing:
            assert restored.queue.get(timeout=.01) is None
            result = client.get('/api/queue/task/' + identity).json()['result']
            assert result['state'] == 'FAILED' and result['output_files'] == []
        else:
            with pytest.raises(ServerError, match='referenced'):
                restored.storage.remove(file['file_id'])
            idx, task = restored.queue.get(timeout=1)
            assert task['id'] == identity and task['graph_json'] == graph
            restored.queue.task_done(idx, ExecutionStatus(True), {}, {})
            assert client.get('/api/queue/task/' + identity).json()['result']['output_files'] == []


def test_input_migration_preserves_existing_output_files(db, tmp_path, monkeypatch):
    from sqlalchemy import text
    from nndeploy.server.template import WorkflowTemplateManager
    from test_database_backup import empty_database
    from test_editor_output_files import _pending_output
    monkeypatch.setattr(WorkflowTemplateManager, 'init_templates', lambda: tmp_path)
    with empty_database(db, tmp_path, 'input-reference-migration') as database:
        url = database.url.render_as_string(hide_password=False)
        server = editor(tmp_path / 'editor', url)
        with TestClient(server.app) as client:
            idx, task, _, _ = _pending_output(client, server, tmp_path)
            server.storage.save(io.BytesIO(b'old output'), 'text/plain', owner='editor:' + task['_editor_scope'],
                editor_task=(task['_editor_scope'], task['id'], task['_editor_token'], ['output']))
            server.queue.task_done(idx, ExecutionStatus(True), {}, {})
            outputs = server.db.editor_output_files(task['_editor_scope'], task['id'])
        with database.engine.begin() as connection:
            connection.execute(text('ALTER TABLE server_editor_file_refs DROP COLUMN kind'))
            connection.execute(text("UPDATE alembic_version SET version_num='0032_editor_file_fields'"))
        database.init_schema()
        assert database.editor_output_files(task['_editor_scope'], task['id']) == outputs
        with TestClient(editor(tmp_path / 'editor', url).app) as client:
            assert client.get(outputs[0]['url']).content == b'old output'


def test_resource_references_follow_nested_node_fields_and_skip_outputs():
    from nndeploy.server.resources import workflow_file_refs
    ref = {'file_id': 'file_model'}
    graph = {'node_repository_': [{'name_': 'sub', 'node_repository_': [
        {'name_': 'model', 'io_type_': 'Model', 'param_': {'io_param_': ['weights'], 'weights': [[ref], ref]},
         'frontend_params': {'preview': {'file_id': 'file_ignore'}}},
        {'name_': 'inline', 'param_': {'model_value_': [ref], 'is_path_': False}},
        {'name_': 'out', 'node_type_': 'Output', 'io_type_': 'File', 'io_param_': ['path_'], 'path_': ref},
        {'name_': 'encode', 'key_': 'nndeploy.codec.PILImageEncodec', 'path_': ref}]}]}
    original = copy.deepcopy(graph)
    assert workflow_file_refs(graph) == [
        (('sub', 'model'), ('param_', 'weights', 0, 0), 'file_model'),
        (('sub', 'model'), ('param_', 'weights', 1), 'file_model')]
    assert graph == original


def test_real_editor_native_image_uses_managed_input_and_independent_output(tmp_path, location):
    from PIL import Image
    from test_native_media import image_graph
    resources = tmp_path / 'resources'
    resources.mkdir()
    args = SimpleNamespace(resources=str(resources), log=str(tmp_path / 'editor.log'), debug=True)
    runtime = _EditorRuntime(args)
    server = runtime.start(lambda jobs, plugins, cancellations: Server(
        args, jobs, plugins, cancellations, database_url=location))
    with TestClient(server.app) as client:
        try:
            data = io.BytesIO()
            Image.new('RGB', (8, 6), (12, 34, 56)).save(data, format='PNG')
            uploaded = client.post('/v1/files', files={'file': ('image.png', data.getvalue(), 'image/png')})
            assert uploaded.status_code == 201, uploaded.text
            reference = {'file_id': uploaded.json()['file_id']}
            graph = json.loads(image_graph().serialize())
            graph['node_repository_'][0]['path_'] = reference
            target = resources / 'output.png'
            graph['node_repository_'][1]['path_'] = str(target)
            response = client.post('/api/queue', json=graph)
            assert response.status_code == 202, response.text
            identity = response.json()['result']['task_id']
            def completed():
                record = client.get('/api/queue/task/' + identity).json()['result']
                return record if record['state'] in ('SUCCEEDED', 'FAILED') else None
            record = wait(completed)
            assert record['state'] == 'SUCCEEDED', record
            assert target.exists()  # Legacy output paths retain their original behavior.
            assert len(record['output_files']) == 1
            file = record['output_files'][0]
            assert file['file_id'] != reference['file_id'] and file['node_path'] == ['encode']
            target.unlink()
            content = client.get(file['url'])
            assert content.status_code == 200
            with Image.open(io.BytesIO(content.content)) as result:
                assert result.size == (8, 6) and result.convert('RGB').getpixel((0, 0)) == (12, 34, 56)
        finally:
            runtime.close()


def test_worker_rejects_corrupt_managed_input_and_preserves_local_paths(tmp_path, location):
    from nndeploy.server.app import _editor_storage_options
    from nndeploy.server.editor_inputs import prepare_worker_inputs
    server = editor(tmp_path / 'editor', location)
    with TestClient(server.app) as client:
        file = server.storage.save(io.BytesIO(b'weights'), 'application/octet-stream')
        graph = workflow({'file_id': file['file_id']})
        graph['node_repository_'][0]['artifacts'].append('relative/local-model.bin')
        key = client.post('/api/queue', json=graph).json()['result']['task_id']
        _, task = server.queue.get(timeout=1)
        original = copy.deepcopy(task)
        options = _editor_storage_options(server)
        prepared = prepare_worker_inputs(task, options, tmp_path)
        assert prepared['node_repository_'][0]['artifacts'][1] == 'relative/local-model.bin'
        assert task == original
        server.storage.resolve(file['file_id'])[0].write_bytes(b'corrupt')
        with pytest.raises(ServerError, match='integrity'):
            prepare_worker_inputs(task, options, tmp_path)
        with pytest.raises(ServerError, match='referenced'):
            server.storage.remove(file['file_id'])
        assert client.get('/api/queue/task/' + key).json()['result']['state'] == 'PENDING'
