"""Declared path lists retain their indices through normal task file storage."""
import copy
import io
import zipfile
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from nndeploy.server import Server
from nndeploy.server.editor_outputs import capture_files, file_outputs
from nndeploy.server.errors import ServerError
from nndeploy.server.task_queue import ExecutionStatus
from test_editor_history import editor, location
from test_editor_output_files import _pending_output


def test_nested_path_lists_preserve_indices_bytes_and_directory_after_restart(tmp_path, location):
    server = editor(tmp_path / 'editor', location)
    root = tmp_path / 'directory'
    root.mkdir()
    (root / 'report.txt').write_text('directory content')
    with TestClient(server.app) as client:
        idx, payload, source, graph = _pending_output(client, server, tmp_path)
        node = graph['node_repository_'][0]
        node.update(io_param_=['path_', 'path_'], path_=[None, str(source), ['', str(root), str(source)]])
        original = copy.deepcopy(graph)
        files = capture_files(graph, server.storage, payload['_editor_scope'], payload['id'], payload['_editor_token'])
        assert graph == original
        assert {tuple(file['field_path']) for file in files} == {('path_', 1), ('path_', 2, 1), ('path_', 2, 2)}
        assert len({file['file_id'] for file in files}) == 3
        assert all(client.get(file['url']).status_code == 404 for file in files)
        source.unlink()
        (root / 'report.txt').unlink()
        assert capture_files(graph, server.storage, payload['_editor_scope'], payload['id'], payload['_editor_token']) == files
        server.queue.task_done(idx, ExecutionStatus(True, 'Completed'), {}, {})
    with TestClient(editor(tmp_path / 'editor', location).app) as client:
        assert client.get('/api/queue/task/' + payload['id']).json()['result']['output_files'] == files
        for file in files:
            response = client.get(file['url'])
            assert response.status_code == 200
            if file['field_path'] == ['path_', 2, 1]:
                with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                    assert archive.read('report.txt') == b'directory content'
            else:
                assert response.content == b'original output'


def test_list_partial_lost_receipt_continues_without_recopying_committed_indices(tmp_path, location, monkeypatch):
    server = editor(tmp_path / 'editor', location)
    with TestClient(server.app) as client:
        idx, payload, source, graph = _pending_output(client, server, tmp_path)
        second = tmp_path / 'second.txt'
        second.write_text('second')
        graph['node_repository_'][0]['path_'] = [str(source), str(second), str(second)]
        create = server.db.create_file
        calls = 0
        def lost(*args, **kwargs):
            nonlocal calls
            result = create(*args, **kwargs)
            calls += 1
            if calls == 2:
                raise OSError('Lost second receipt')
            return result
        monkeypatch.setattr(server.db, 'create_file', lost)
        with pytest.raises(OSError, match='Lost second receipt'):
            capture_files(graph, server.storage, payload['_editor_scope'], payload['id'], payload['_editor_token'])
        source.unlink()
        second.write_text('changed')
        monkeypatch.setattr(server.db, 'create_file', create)
        files = capture_files(graph, server.storage, payload['_editor_scope'], payload['id'], payload['_editor_token'])
        server.queue.task_done(idx, ExecutionStatus(True, 'Completed'), {}, {})
        content = {tuple(file['field_path']): client.get(file['url']).content for file in files}
        assert content == {('path_', 0): b'original output', ('path_', 1): b'second', ('path_', 2): b'changed'}


@pytest.mark.parametrize('failure', ['quota', 'missing'])
def test_list_partial_failure_detaches_and_releases_all_files(tmp_path, location, failure):
    server = Server(SimpleNamespace(resources=str(tmp_path / 'editor'), debug=True), database_url=location,
                    storage_config={'max_files_per_subject': 1})
    with TestClient(server.app) as client:
        idx, payload, source, graph = _pending_output(client, server, tmp_path)
        root = tmp_path / 'directory'
        root.mkdir()
        (root / 'report.txt').write_text('content')
        graph['node_repository_'][0]['path_'] = [str(source), str(root if failure == 'quota' else root / 'missing')]
        with pytest.raises(ServerError if failure == 'quota' else FileNotFoundError):
            capture_files(graph, server.storage, payload['_editor_scope'], payload['id'], payload['_editor_token'])
        saved = server.db.editor_output_files(payload['_editor_scope'], payload['id'])
        assert len(saved) == 1
        server.queue.task_done(idx, ExecutionStatus(False, 'Output capture failed'), {}, {})
        server.storage.cleanup()
        assert server.db.editor_output_files(payload['_editor_scope'], payload['id']) == []
        assert server.db.get_file(saved[0]['file_id']) is None
        assert {path.name for path in server.storage.root.iterdir()} == {'.server-id'}
        assert server.storage.save(io.BytesIO(b'next'), 'text/plain', owner='editor:' + payload['_editor_scope'])['size'] == 4


@pytest.mark.parametrize('kind', ['type', 'count', 'depth', 'empty_count'])
def test_invalid_lists_are_rejected_before_any_file_is_published(tmp_path, location, kind):
    server = editor(tmp_path / 'editor', location)
    with TestClient(server.app) as client:
        idx, payload, source, graph = _pending_output(client, server, tmp_path)
        value = {'type': [str(source), {'bad': 'path'}], 'count': [str(source)] * 257,
                 'empty_count': [[None] * 256] * 5}.get(kind)
        if kind == 'depth':
            value = str(source)
            for _ in range(65):
                value = [value]
        graph['node_repository_'][0]['path_'] = value
        with pytest.raises(ValueError):
            capture_files(graph, server.storage, payload['_editor_scope'], payload['id'], payload['_editor_token'])
        assert server.db.editor_output_files(payload['_editor_scope'], payload['id']) == []
        assert {path.name for path in server.storage.root.iterdir()} == {'.server-id'}


def test_array_indices_do_not_collide_with_literal_field_names():
    workflow = {'node_repository_': [{'name_': 'output', 'node_type_': 'Output', 'io_type_': 'File',
                'io_params_': ['paths', 'paths.0', 'paths[0]'], 'paths': ['a'], 'paths.0': 'b', 'paths[0]': 'c'}]}
    assert file_outputs(workflow) == [(['output'], ['paths', 0], 'a'),
                                      (['output'], ['paths.0'], 'b'), (['output'], ['paths[0]'], 'c')]
