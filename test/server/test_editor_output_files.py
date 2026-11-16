"""Editor output files belong to completed tasks, not mutable workflow paths."""
import io
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from nndeploy.server import Server
from nndeploy.server.app import _EditorRuntime
from test_editor_history import location
from test_editor_queue_recovery import wait
from test_server_api import image_graph
from test_db import db


def test_real_editor_output_survives_overwrite_and_restart(tmp_path, location):
    resources = tmp_path / 'resources'
    resources.mkdir()
    source, destination = resources / 'input.png', resources / 'output.png'
    args = SimpleNamespace(resources=str(resources), log=str(tmp_path / 'editor.log'), debug=True)
    runtime = _EditorRuntime(args)
    server = runtime.start(lambda jobs, plugins, cancellations: Server(
        args, jobs, plugins, cancellations, database_url=location))
    outputs = []
    graph = image_graph()
    graph['node_repository_'][0]['path_'] = str(source)
    graph['node_repository_'][1]['path_'] = str(destination)
    with TestClient(server.app) as client:
        try:
            for color in ((10, 20, 30), (70, 80, 90)):
                Image.new('RGB', (4, 5), color).save(source)
                response = client.post('/api/queue', json=graph)
                assert response.status_code == 202, response.text
                identity = response.json()['result']['task_id']
                def completed():
                    record = client.get('/api/queue/task/' + identity).json()['result']
                    return record if record['state'] in ('SUCCEEDED', 'FAILED') else None
                record = wait(completed)
                assert record['state'] == 'SUCCEEDED', record
                assert len(record['output_files']) == 1
                file = record['output_files'][0]
                assert file['node_path'] == ['encode']
                content = client.get(file['url'])
                assert content.status_code == 200, content.text
                assert Image.open(io.BytesIO(content.content)).getpixel((0, 0)) == color
                outputs.append((identity, file, content.content))
        finally:
            runtime.close()
    source.unlink()
    destination.unlink()
    assert outputs[0][1]['file_id'] != outputs[1][1]['file_id']
    with TestClient(Server(args, database_url=location).app) as client:
        for identity, file, content in outputs:
            result = client.get('/api/queue/task/' + identity).json()['result']
            assert result['output_files'] == [file]
            assert client.get(file['url']).content == content
            part = client.get(file['url'], headers={'Range': 'bytes=0-7'})
            assert part.status_code == 206 and part.content == content[:8]
        # A valid file cannot be requested through a different task.
        assert client.get(outputs[0][1]['url'].replace(outputs[0][0], outputs[1][0])).status_code == 404


def _pending_output(client, server, tmp_path):
    from test_server_api import text_graph
    identity = client.post('/api/queue', json=text_graph()).json()['result']['task_id']
    idx, payload = server.queue.get(timeout=1)
    server.queue.mark_dispatched(idx)
    source = tmp_path / 'result.txt'
    source.write_text('original output')
    graph = {'node_repository_': [{'name_': 'output', 'node_type_': 'Output',
             'io_type_': 'File', 'io_param_': ['path_'], 'path_': str(source)}]}
    return idx, payload, source, graph


def test_output_reference_protects_file_and_is_private_until_success(tmp_path, location):
    from nndeploy.server.editor_outputs import capture_files
    from nndeploy.server.errors import ServerError
    from nndeploy.server.task_queue import ExecutionStatus
    from test_editor_history import editor
    server = editor(tmp_path / 'editor', location)
    with TestClient(server.app) as client:
        idx, payload, source, graph = _pending_output(client, server, tmp_path)
        scope, token = payload['_editor_scope'], payload['_editor_token']
        files = capture_files(graph, server.storage, scope, payload['id'], token)
        file = files[0]
        assert client.get(file['url']).status_code == 404
        assert client.get('/v1/files/' + file['file_id'] + '/content').status_code == 404
        with pytest.raises(ServerError) as referenced:
            server.storage.remove(file['file_id'])
        assert referenced.value.status_code == 409
        server.queue.task_done(idx, ExecutionStatus(True, 'Completed'), {}, {})
        assert client.get(file['url']).content == b'original output'
        with pytest.raises(ServerError):
            server.storage.remove(file['file_id'])
    with TestClient(editor(tmp_path / 'other', location).app) as client:
        assert client.get(file['url']).status_code == 404


@pytest.mark.parametrize('failure', ['failed', 'cancelled'])
def test_failed_or_cancelled_output_is_detached_and_cleaned(tmp_path, location, failure):
    from nndeploy.server.editor_outputs import capture_files
    from nndeploy.server.task_queue import ExecutionStatus
    from test_editor_history import editor
    server = editor(tmp_path / 'editor', location)
    with TestClient(server.app) as client:
        idx, payload, source, graph = _pending_output(client, server, tmp_path)
        file = capture_files(graph, server.storage, payload['_editor_scope'], payload['id'], payload['_editor_token'])[0]
        server.queue.task_done(idx, ExecutionStatus(False, 'Stopped', failure), {}, {})
        record = client.get('/api/queue/task/' + payload['id']).json()['result']
        assert record['output_files'] == []
        assert client.get(file['url']).status_code == 404
        server.storage.cleanup()
        assert server.db.get_file(file['file_id']) is None


def test_lost_file_commit_receipt_preserves_original_bytes_without_recopy(tmp_path, location, monkeypatch):
    from nndeploy.server.editor_outputs import capture_files
    from nndeploy.server.task_queue import ExecutionStatus
    from test_editor_history import editor
    server = editor(tmp_path / 'editor', location)
    with TestClient(server.app) as client:
        idx, payload, source, graph = _pending_output(client, server, tmp_path)
        create = server.db.create_file
        def lost(*args, **kwargs):
            create(*args, **kwargs)
            raise OSError('Lost file commit receipt')
        monkeypatch.setattr(server.db, 'create_file', lost)
        with pytest.raises(OSError, match='Lost file commit'):
            capture_files(graph, server.storage, payload['_editor_scope'], payload['id'], payload['_editor_token'])
        source.write_text('overwritten after commit')
        monkeypatch.setattr(server.db, 'create_file', create)
        files = capture_files(graph, server.storage, payload['_editor_scope'], payload['id'], payload['_editor_token'])
        assert len(files) == 1
        server.queue.task_done(idx, ExecutionStatus(True, 'Completed'), {}, {})
        assert client.get(files[0]['url']).content == b'original output'


def test_cancelled_task_cannot_publish_new_files(tmp_path, location):
    from nndeploy.server.editor_outputs import capture_files
    from nndeploy.server.errors import ServerError
    from test_editor_history import editor
    server = editor(tmp_path / 'editor', location)
    with TestClient(server.app) as client:
        idx, payload, source, graph = _pending_output(client, server, tmp_path)
        assert client.post('/api/queue/cancel/' + payload['id']).status_code == 200
        with pytest.raises(ServerError) as rejected:
            capture_files(graph, server.storage, payload['_editor_scope'], payload['id'], payload['_editor_token'])
        assert rejected.value.code == 'editor_ownership_lost'
        assert server.db.editor_output_files(payload['_editor_scope'], payload['id']) == []
        assert not list(server.storage.root.glob('file_*'))


def test_output_discovery_handles_null_params_and_preserves_nested_paths():
    from nndeploy.server.editor_outputs import file_outputs
    graph = {'node_repository_': [
        {'name_': 'memory', 'node_type_': 'Output', 'io_type_': 'Image', 'param_': None, 'param': None},
        {'name_': 'nested', 'node_repository_': [
            {'name_': 'output', 'node_type_': 'Output', 'io_type_': 'File',
             'param_': {'io_param_': ['path'], 'path': '/tmp/example.txt'}}]}]}
    assert file_outputs(graph) == [(['nested', 'output'], ['param_', 'path'], '/tmp/example.txt')]


def test_multiple_file_fields_survive_overwrite_and_server_restart(tmp_path, location):
    from nndeploy.server.editor_outputs import capture_files
    from nndeploy.server.task_queue import ExecutionStatus
    from test_editor_history import editor
    server = editor(tmp_path / 'editor', location)
    with TestClient(server.app) as client:
        idx, payload, first, graph = _pending_output(client, server, tmp_path)
        second = tmp_path / 'second.txt'
        second.write_text('second output')
        node = graph['node_repository_'][0]
        node['io_param_'] = ['path_', 'report_']
        node['report_'] = str(second)
        nested = tmp_path / 'nested.txt'
        nested.write_text('nested output')
        graph['node_repository_'].append({'name_': 'group', 'node_repository_': [
            {'name_': 'output', 'node_type_': 'Output', 'io_type_': 'File',
             'param_': {'io_params_': ['report_'], 'report_': str(nested)}}]})
        files = capture_files(graph, server.storage, payload['_editor_scope'], payload['id'], payload['_editor_token'])
        assert len(files) == 3
        expected = {(('output',), ('path_',)): b'original output',
                    (('output',), ('report_',)): b'second output',
                    (('group', 'output'), ('param_', 'report_')): b'nested output'}
        for source in (first, second, nested):
            source.write_text('overwritten')
        assert capture_files(graph, server.storage, payload['_editor_scope'], payload['id'], payload['_editor_token']) == files
        server.queue.task_done(idx, ExecutionStatus(True, 'Completed'), {}, {})
    with TestClient(editor(tmp_path / 'editor', location).app) as client:
        record = client.get('/api/queue/task/' + payload['id']).json()['result']
        assert record['output_files'] == files
        for file in files:
            assert client.get(file['url']).content == expected[(tuple(file['node_path']), tuple(file['field_path']))]


def test_multi_file_lost_receipt_keeps_first_field_and_continues_remaining_fields(tmp_path, location, monkeypatch):
    from nndeploy.server.editor_outputs import capture_files
    from nndeploy.server.task_queue import ExecutionStatus
    from test_editor_history import editor
    server = editor(tmp_path / 'editor', location)
    with TestClient(server.app) as client:
        idx, payload, source, graph = _pending_output(client, server, tmp_path)
        second = tmp_path / 'report.txt'
        second.write_text('report')
        graph['node_repository_'][0].update(io_param_=['path_', 'report_'], report_=str(second))
        create = server.db.create_file
        def lost(*args, **kwargs):
            create(*args, **kwargs)
            raise OSError('Lost file commit receipt')
        monkeypatch.setattr(server.db, 'create_file', lost)
        with pytest.raises(OSError, match='Lost file commit'):
            capture_files(graph, server.storage, payload['_editor_scope'], payload['id'], payload['_editor_token'])
        source.write_text('overwritten after commit')
        monkeypatch.setattr(server.db, 'create_file', create)
        files = capture_files(graph, server.storage, payload['_editor_scope'], payload['id'], payload['_editor_token'])
        assert len(files) == 2
        server.queue.task_done(idx, ExecutionStatus(True, 'Completed'), {}, {})
        assert {tuple(file['field_path']): client.get(file['url']).content for file in files} == {
            ('path_',): b'original output', ('report_',): b'report'}


def test_old_output_reference_retry_keeps_bytes_and_download_urls(tmp_path, location):
    from nndeploy.server.editor_outputs import capture_files
    from nndeploy.server.task_queue import ExecutionStatus
    from test_editor_history import editor
    server = editor(tmp_path / 'editor', location)
    with TestClient(server.app) as client:
        idx, payload, source, graph = _pending_output(client, server, tmp_path)
        # A pre-upgrade Worker supplies the original four-element identity.
        server.storage.save(io.BytesIO(b'old bytes'), 'text/plain', owner='editor:' + payload['_editor_scope'],
                            editor_task=(payload['_editor_scope'], payload['id'], payload['_editor_token'], ['output']))
        old = server.db.editor_output_files(payload['_editor_scope'], payload['id'])[0]
        assert 'field_path' not in old
        source.write_text('do not recopy this')
        second = tmp_path / 'report.txt'
        second.write_text('new field')
        graph['node_repository_'][0].update(io_param_=['path_', 'report_'], report_=str(second))
        captured = capture_files(graph, server.storage, payload['_editor_scope'], payload['id'], payload['_editor_token'])
        assert len(captured) == 2 and old in captured
        server.queue.task_done(idx, ExecutionStatus(True, 'Completed'), {}, {})
        assert client.get(old['url']).content == b'old bytes'
    with TestClient(editor(tmp_path / 'editor', location).app) as client:
        assert client.get(old['url']).content == b'old bytes'
        assert len(client.get('/api/queue/task/' + payload['id']).json()['result']['output_files']) == 2


def test_file_field_migration_preserves_legacy_references(db, tmp_path, monkeypatch):
    from sqlalchemy import text
    from nndeploy.server.task_queue import ExecutionStatus
    from nndeploy.server.template import WorkflowTemplateManager
    from test_database_backup import empty_database
    from test_editor_history import editor
    monkeypatch.setattr(WorkflowTemplateManager, 'init_templates', lambda: tmp_path)
    with empty_database(db, tmp_path, 'legacy-outputs') as database:
        location = database.url.render_as_string(hide_password=False)
        server = editor(tmp_path / 'editor', location)
        with TestClient(server.app) as client:
            idx, payload, source, graph = _pending_output(client, server, tmp_path)
            server.storage.save(io.BytesIO(b'old bytes'), 'text/plain', owner='editor:' + payload['_editor_scope'],
                                editor_task=(payload['_editor_scope'], payload['id'], payload['_editor_token'], ['output']))
            old = server.db.editor_output_files(payload['_editor_scope'], payload['id'])
            server.queue.task_done(idx, ExecutionStatus(True, 'Completed'), {}, {})
        # No Server is using this dedicated database during the upgrade.
        with database.engine.begin() as connection:
            connection.execute(text('ALTER TABLE server_editor_file_refs DROP COLUMN field_path'))
            connection.execute(text("UPDATE alembic_version SET version_num='0031_workflow_timestamps'"))
        database.init_schema()
        assert database.editor_output_files(payload['_editor_scope'], payload['id']) == old
        with TestClient(editor(tmp_path / 'editor', location).app) as client:
            assert client.get('/api/queue/task/' + payload['id']).json()['result']['output_files'] == old
            assert client.get(old[0]['url']).content == b'old bytes'


@pytest.mark.parametrize('path_list', [False, True])
def test_real_editor_node_saves_both_image_fields_before_next_run(tmp_path, location, path_list):
    resources = tmp_path / 'resources'
    plugins = resources / 'plugin'
    plugins.mkdir(parents=True)
    (plugins / 'two_images.py').write_text('''
import json
import nndeploy.dag
from nndeploy.codec.pil import PILImageEncodec
from PIL import ImageOps
class TwoImages(PILImageEncodec):
    def deserialize(self, value):
        data = json.loads(value)
        self.report = data['reports_'][1] if 'reports_' in data else data['report_']
        return super().deserialize(value)
    def run(self):
        status = super().run()
        ImageOps.invert(self.get_input(0).get(self)).save(self.report)
        return status
class Creator(nndeploy.dag.NodeCreator):
    def create_node(self, name, inputs, outputs):
        self.node = TwoImages(name, inputs, outputs)
        self.node.set_key('test.editor.TwoImages')
        return self.node
creator = Creator()
nndeploy.dag.register_node('test.editor.TwoImages', creator)
''')
    source, first, second = (resources / name for name in ('input.png', 'image.png', 'inverse.png'))
    graph = image_graph()
    graph['node_repository_'][0]['path_'] = str(source)
    graph['node_repository_'][1].update(key_='test.editor.TwoImages', path_=str(first), report_=str(second),
                                       node_type_='Output', io_type_='Image', io_params_=['path_', 'report_'])
    if path_list:
        graph['node_repository_'][1].update(io_params_=['reports_'], reports_=[str(first), str(second)])
    args = SimpleNamespace(resources=str(resources), log=str(tmp_path / 'editor.log'), debug=True)
    runtime = _EditorRuntime(args)
    server = runtime.start(lambda jobs, plugins, cancellations: Server(args, jobs, plugins, cancellations, database_url=location))
    saved = []
    with TestClient(server.app) as client:
        try:
            for color in ((10, 20, 30), (70, 80, 90)):
                Image.new('RGB', (4, 5), color).save(source)
                response = client.post('/api/queue', json=graph)
                assert response.status_code == 202, response.text
                identity = response.json()['result']['task_id']
                def completed():
                    record = client.get('/api/queue/task/' + identity).json()['result']
                    return record if record['state'] in ('SUCCEEDED', 'FAILED') else None
                record = wait(completed)
                assert record['state'] == 'SUCCEEDED', record
                assert len(record['output_files']) == 2
                for file in record['output_files']:
                    expected = color if file['field_path'] in (['path_'], ['reports_', 0]) else tuple(255 - channel for channel in color)
                    assert Image.open(io.BytesIO(client.get(file['url']).content)).getpixel((0, 0)) == expected
                    saved.append((file, expected))
        finally:
            runtime.close()
    first.unlink()
    second.unlink()
    with TestClient(Server(args, database_url=location).app) as client:
        for file, expected in saved:
            assert Image.open(io.BytesIO(client.get(file['url']).content)).getpixel((0, 0)) == expected


def test_fifo_output_rejected_without_waiting_for_a_writer(tmp_path, location):
    import os
    from nndeploy.server.editor_outputs import capture_files
    from test_editor_history import editor
    server = editor(tmp_path / 'editor', location)
    with TestClient(server.app) as client:
        idx, payload, source, graph = _pending_output(client, server, tmp_path)
        source.unlink()
        os.mkfifo(source)
        with pytest.raises(ValueError, match='regular files'):
            capture_files(graph, server.storage, payload['_editor_scope'], payload['id'], payload['_editor_token'])


def test_multi_file_history_expiry_keeps_live_download_then_cleans_every_field(tmp_path, location):
    from nndeploy.server.editor_outputs import capture_files
    from nndeploy.server.task_queue import ExecutionStatus
    from test_editor_history_retention import age, make_server
    server = make_server(tmp_path / 'editor', location, 60)
    with TestClient(server.app) as client:
        idx, payload, source, graph = _pending_output(client, server, tmp_path)
        graph['node_repository_'][0].update(io_param_=['path_', 'copy_'], copy_=str(source))
        files = capture_files(graph, server.storage, payload['_editor_scope'], payload['id'], payload['_editor_token'])
        server.queue.task_done(idx, ExecutionStatus(True, 'Completed'), {}, {})
        with server.storage.open(files[0]['file_id']) as reader:
            age(server, payload['id'])
            assert client.post('/api/queue/history/cleanup').json()['result']['deleted_history'] == 1
            assert client.get('/api/queue/task/' + payload['id']).json()['result']['output_files'] == []
            for file in files:
                assert client.get(file['url']).status_code == 404
            server.storage.cleanup()
            assert server.db.get_file(files[1]['file_id']) is None
            assert server.db.get_file(files[0]['file_id']) is not None
            assert reader.read() == b'original output'
        server.storage.cleanup()
        assert server.db.get_file(files[0]['file_id']) is None


def test_file_discovery_deduplicates_declared_fields_and_limits_actual_file_count():
    from nndeploy.server.editor_outputs import file_outputs, MAX_FILES
    node = {'name_': 'output', 'node_type_': 'Output', 'io_type_': 'File',
            'io_param_': [], 'io_params_': ['path_', 'path_'], 'path_': '/tmp/output.txt',
            'param_': {'io_params_': ['path_'], 'path_': '/tmp/nested.txt'}}
    assert file_outputs({'node_repository_': [node]}) == [
        (['output'], ['path_'], '/tmp/output.txt'), (['output'], ['param_', 'path_'], '/tmp/nested.txt')]
    many = [dict(node, name_=str(index)) for index in range(MAX_FILES // 2 + 1)]
    with pytest.raises(ValueError, match='Too many output files'):
        file_outputs({'node_repository_': many})


def _slow_output_worker(*args, **kwargs):
    """Hold only file publication, after a real Graph has deinitialized."""
    import time
    from pathlib import Path
    from nndeploy.server import worker, editor_outputs
    root = Path(args[6])
    capture = editor_outputs.capture_worker_files
    interrupt = worker.GraphExecutor.interrupt_running
    def held(payload, options):
        (root / 'copy-started').touch()
        deadline = time.monotonic() + 15
        while not (root / 'release-copy').exists():
            if time.monotonic() > deadline:
                raise RuntimeError('Test copy was not released')
            time.sleep(.02)
        return capture(payload, options)
    def observed(executor):
        (root / 'cancel-observed').touch()
        return interrupt(executor)
    editor_outputs.capture_worker_files = held
    worker.GraphExecutor.interrupt_running = observed
    worker.run(*args, **kwargs)


def test_cancellation_is_observed_while_worker_is_saving_files(tmp_path, location, monkeypatch):
    from nndeploy.server import app
    resources = tmp_path / 'resources'
    resources.mkdir()
    source, destination = resources / 'input.png', resources / 'output.png'
    Image.new('RGB', (4, 5), (1, 2, 3)).save(source)
    args = SimpleNamespace(resources=str(resources), log=str(tmp_path / 'editor.log'), debug=True)
    monkeypatch.setattr(app, 'worker_run', _slow_output_worker)
    runtime = _EditorRuntime(args)
    server = runtime.start(lambda jobs, plugins, cancellations: Server(
        args, jobs, plugins, cancellations, database_url=location))
    graph = image_graph()
    graph['node_repository_'][0]['path_'] = str(source)
    graph['node_repository_'][1]['path_'] = str(destination)
    with TestClient(server.app) as client:
        try:
            identity = client.post('/api/queue', json=graph).json()['result']['task_id']
            wait(lambda: (resources / 'copy-started').exists())
            assert client.post('/api/queue/cancel/' + identity).status_code == 200
            wait(lambda: (resources / 'cancel-observed').exists())
            pending = client.get('/api/queue/task/' + identity).json()['result']
            assert pending['state'] not in ('CANCELLED', 'SUCCEEDED', 'FAILED')
            (resources / 'release-copy').touch()
            wait(lambda: client.get('/api/queue/task/' + identity).json()['result']['state'] == 'CANCELLED')
            assert client.get('/api/queue/task/' + identity).json()['result']['output_files'] == []
        finally:
            (resources / 'release-copy').touch()
            runtime.close()


@pytest.mark.parametrize('multiple', [False, True, 'list'])
def test_editor_output_reloads_from_s3_after_local_copy_is_removed(tmp_path, location, s3_service, monkeypatch, multiple):
    from nndeploy.server.editor_outputs import capture_files
    from nndeploy.server.task_queue import ExecutionStatus
    endpoint, environment, _ = s3_service
    monkeypatch.setenv('NNDEPLOY_TEST_ACCESS', environment['MINIO_ROOT_USER'])
    monkeypatch.setenv('NNDEPLOY_TEST_SECRET', environment['MINIO_ROOT_PASSWORD'])
    config = dict(backend='s3', bucket='nndeploy-tests', region='us-east-1', endpoint=endpoint,
                  access_key_env='NNDEPLOY_TEST_ACCESS', secret_key_env='NNDEPLOY_TEST_SECRET')
    args = SimpleNamespace(resources=str(tmp_path / 'editor'), debug=True)
    server = Server(args, database_url=location, storage_config=config)
    with TestClient(server.app) as client:
        idx, payload, source, graph = _pending_output(client, server, tmp_path)
        if multiple == 'list':
            graph['node_repository_'][0]['path_'] = [None, str(source), str(source)]
        elif multiple:
            graph['node_repository_'][0].update(io_param_=['path_', 'copy_'], copy_=str(source))
        files = capture_files(graph, server.storage, payload['_editor_scope'], payload['id'], payload['_editor_token'])
        assert len(files) == (2 if multiple else 1)
        if multiple == 'list':
            assert {tuple(file['field_path']) for file in files} == {('path_', 1), ('path_', 2)}
        server.queue.task_done(idx, ExecutionStatus(True, 'Completed'), {}, {})
        for file in files:
            server.storage.resolve(file['file_id'])[0].unlink()
        source.unlink()
    with TestClient(Server(args, database_url=location, storage_config=config).app) as client:
        for file in files:
            result = client.get(file['url'])
            assert result.status_code == 200, result.text
            assert result.content == b'original output'


# Reuse the real disposable MinIO fixture used by the storage integrations.
from test_storage import s3_service


@pytest.mark.parametrize('quota', [False, True])
def test_editor_output_journal_recovers_unknown_commit_without_deleting_file(tmp_path, location, monkeypatch, quota):
    from nndeploy.server.editor_outputs import capture_files
    from nndeploy.server.task_queue import ExecutionStatus
    server = Server(SimpleNamespace(resources=str(tmp_path / 'editor'), debug=True), database_url=location,
                    storage_config={'max_files_per_subject': 1} if quota else None)
    with TestClient(server.app) as client:
        idx, payload, source, graph = _pending_output(client, server, tmp_path)
        create, reconcile = server.db.create_file, server.db.reconcile_editor_file
        def lost(*args, **kwargs):
            create(*args, **kwargs)
            raise OSError('Lost commit receipt')
        def unavailable(*args, **kwargs):
            raise OSError('Database is unavailable')
        monkeypatch.setattr(server.db, 'create_file', lost)
        monkeypatch.setattr(server.db, 'reconcile_editor_file', unavailable)
        with pytest.raises(OSError, match='Lost commit'):
            capture_files(graph, server.storage, payload['_editor_scope'], payload['id'], payload['_editor_token'])
        assert len(list(server.storage.root.glob('.pending-*.json'))) == 1
        monkeypatch.setattr(server.db, 'create_file', create)
        monkeypatch.setattr(server.db, 'reconcile_editor_file', reconcile)
        assert server.storage.recover_pending() == 1
        assert not list(server.storage.root.glob('.pending-*.json'))
        source.unlink()
        files = server.db.editor_output_files(payload['_editor_scope'], payload['id'])
        assert len(files) == 1
        server.queue.task_done(idx, ExecutionStatus(True, 'Completed'), {}, {})
        assert client.get(files[0]['url']).content == b'original output'


@pytest.mark.parametrize('same_node', [False, True])
def test_editor_output_quota_failure_does_not_leave_partial_success(tmp_path, location, same_node):
    from nndeploy.server.editor_outputs import capture_files
    from nndeploy.server.errors import ServerError
    from nndeploy.server.task_queue import ExecutionStatus
    server = Server(SimpleNamespace(resources=str(tmp_path / 'editor'), debug=True), database_url=location,
                    storage_config={'max_files_per_subject': 1})
    with TestClient(server.app) as client:
        idx, payload, source, graph = _pending_output(client, server, tmp_path)
        if same_node:
            graph['node_repository_'][0].update(io_param_=['path_', 'copy_'], copy_=str(source))
        else:
            graph['node_repository_'].append(dict(graph['node_repository_'][0], name_='second'))
        with pytest.raises(ServerError) as full:
            capture_files(graph, server.storage, payload['_editor_scope'], payload['id'], payload['_editor_token'])
        assert full.value.code == 'subject_files_full'
        file = server.db.editor_output_files(payload['_editor_scope'], payload['id'])[0]
        assert client.get(file['url']).status_code == 404
        server.queue.task_done(idx, ExecutionStatus(False, 'Output files could not be saved'), {}, {})
        server.storage.cleanup()
        assert server.db.get_file(file['file_id']) is None
        # The failed task's quota reservation is released as well as its content.
        assert server.storage.save(io.BytesIO(b'next'), 'text/plain', owner='editor:' + payload['_editor_scope'])['size'] == 4
