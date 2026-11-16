"""Directory outputs use ordinary managed files and editor task history."""
import io
import zipfile

import pytest
from fastapi.testclient import TestClient

from nndeploy.server.editor_outputs import capture_files
from nndeploy.server.task_queue import ExecutionStatus
from test_editor_history import editor, location
from test_editor_output_files import _pending_output
from test_storage import s3_service


@pytest.mark.parametrize('kind', ['Dir', 'File'])
def test_directory_result_keeps_tree_after_source_changes_and_restart(tmp_path, location, kind):
    root = tmp_path / 'result'
    (root / 'nested' / 'empty').mkdir(parents=True)
    (root / 'text.txt').write_text('original')
    (root / 'nested' / '中文.bin').write_bytes(b'\x00\xff\x01')
    server = editor(tmp_path / 'editor', location)
    with TestClient(server.app) as client:
        idx, payload, source, graph = _pending_output(client, server, tmp_path)
        graph['node_repository_'][0].update(io_type_=kind, path_=str(root))
        files = capture_files(graph, server.storage, payload['_editor_scope'], payload['id'], payload['_editor_token'])
        assert len(files) == 1 and files[0]['media_type'] == 'application/zip'
        file = files[0]
        assert file['field_path'] == ['path_']
        assert client.get(file['url']).status_code == 404
        (root / 'text.txt').write_text('overwritten')
        (root / 'nested' / '中文.bin').unlink()
        assert capture_files(graph, server.storage, payload['_editor_scope'], payload['id'], payload['_editor_token']) == files
        server.queue.task_done(idx, ExecutionStatus(True, 'Completed'), {}, {})
    with TestClient(editor(tmp_path / 'editor', location).app) as client:
        response = client.get(file['url'])
        assert response.status_code == 200 and response.headers['content-type'] == 'application/zip'
        assert '.zip' in response.headers['content-disposition'] and 'attachment' in response.headers['content-disposition']
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            assert archive.read('text.txt') == b'original'
            assert archive.read('nested/中文.bin') == b'\x00\xff\x01'
            assert archive.getinfo('nested/empty/').is_dir()
            assert set(archive.namelist()) == {'text.txt', 'nested/', 'nested/中文.bin', 'nested/empty/'}


@pytest.mark.parametrize('kind', ['symlink', 'fifo', 'bytes', 'overhead', 'entries', 'name'])
def test_invalid_directory_leaves_no_published_or_temporary_archive(tmp_path, location, monkeypatch, kind):
    import os
    import nndeploy.server.storage.archive as archive
    root = tmp_path / 'result'
    root.mkdir()
    (root / 'first.txt').write_text('first')
    server = editor(tmp_path / 'editor', location)
    if kind == 'symlink':
        (root / 'link').symlink_to(tmp_path / 'outside')
        (tmp_path / 'outside').write_text('outside contents')
    elif kind == 'fifo':
        os.mkfifo(root / 'fifo')
    elif kind == 'bytes':
        server.storage.max_bytes = 100
        (root / 'large').write_bytes(b'x' * 101)
    elif kind == 'overhead':
        server.storage.max_bytes = 100
    elif kind == 'entries':
        monkeypatch.setattr(archive, 'MAX_ENTRIES', 1)
        (root / 'second').write_text('second')
    elif kind == 'name':
        (root / 'back\\slash').write_text('bad path')
    with TestClient(server.app) as client:
        idx, payload, _, graph = _pending_output(client, server, tmp_path)
        graph['node_repository_'][0].update(io_type_='Dir', path_=str(root))
        with pytest.raises((ValueError, OSError)):
            capture_files(graph, server.storage, payload['_editor_scope'], payload['id'], payload['_editor_token'])
        assert server.db.editor_output_files(payload['_editor_scope'], payload['id']) == []
        assert {path.name for path in server.storage.root.iterdir()} == {'.server-id'}


def test_directory_write_during_capture_is_rejected_without_publishing(tmp_path, location, monkeypatch):
    root = tmp_path / 'result'
    root.mkdir()
    source = root / 'value.txt'
    source.write_bytes(b'a' * 65536)
    original = zipfile.ZipFile.open
    def changing(archive, name, mode='r', *args, **kwargs):
        result = original(archive, name, mode, *args, **kwargs)
        if mode == 'w' and name == 'value.txt':
            source.write_bytes(b'b' * 10)
        return result
    monkeypatch.setattr(zipfile.ZipFile, 'open', changing)
    server = editor(tmp_path / 'editor', location)
    with TestClient(server.app) as client:
        idx, payload, _, graph = _pending_output(client, server, tmp_path)
        graph['node_repository_'][0].update(io_type_='Dir', path_=str(root))
        with pytest.raises(ValueError, match='changed'):
            capture_files(graph, server.storage, payload['_editor_scope'], payload['id'], payload['_editor_token'])
        assert server.db.editor_output_files(payload['_editor_scope'], payload['id']) == []


def test_directory_lost_commit_receipt_never_recaptures_changed_tree(tmp_path, location, monkeypatch):
    root = tmp_path / 'result'
    root.mkdir()
    source = root / 'value.txt'
    source.write_text('original')
    server = editor(tmp_path / 'editor', location)
    with TestClient(server.app) as client:
        idx, payload, _, graph = _pending_output(client, server, tmp_path)
        graph['node_repository_'][0].update(io_type_='Dir', path_=str(root))
        create = server.db.create_file
        def lost(*args, **kwargs):
            create(*args, **kwargs)
            raise OSError('Lost archive receipt')
        monkeypatch.setattr(server.db, 'create_file', lost)
        with pytest.raises(OSError, match='Lost archive receipt'):
            capture_files(graph, server.storage, payload['_editor_scope'], payload['id'], payload['_editor_token'])
        source.write_text('changed')
        monkeypatch.setattr(server.db, 'create_file', create)
        files = capture_files(graph, server.storage, payload['_editor_scope'], payload['id'], payload['_editor_token'])
        assert len(files) == 1
        server.queue.task_done(idx, ExecutionStatus(True, 'Completed'), {}, {})
        with zipfile.ZipFile(io.BytesIO(client.get(files[0]['url']).content)) as archive:
            assert archive.read('value.txt') == b'original'


def test_real_editor_graph_directory_is_captured_before_next_request(tmp_path, location):
    from types import SimpleNamespace
    from PIL import Image
    from nndeploy.server import Server
    from nndeploy.server.app import _EditorRuntime
    from test_editor_queue_recovery import wait
    from test_server_api import image_graph
    resources = tmp_path / 'resources'
    plugins = resources / 'plugin'
    plugins.mkdir(parents=True)
    (plugins / 'directory_output.py').write_text('''
import nndeploy.dag
import nndeploy.base
from pathlib import Path
from nndeploy.codec.pil import PILImageEncodec
class DirectoryOutput(PILImageEncodec):
    def run(self):
        root = Path(self.path_)
        (root / 'empty').mkdir(parents=True, exist_ok=True)
        image = self.get_input(0).get(self)
        image.save(root / 'image.png')
        (root / 'report.txt').write_text(str(image.getpixel((0, 0))))
        return nndeploy.base.Status.ok()
class Creator(nndeploy.dag.NodeCreator):
    def create_node(self, name, inputs, outputs):
        self.node = DirectoryOutput(name, inputs, outputs)
        self.node.set_key('test.editor.DirectoryOutput')
        return self.node
creator = Creator()
nndeploy.dag.register_node('test.editor.DirectoryOutput', creator)
''')
    source = resources / 'input.png'
    output = resources / 'output'
    graph = image_graph()
    graph['node_repository_'][0]['path_'] = str(source)
    graph['node_repository_'][1].update(key_='test.editor.DirectoryOutput', path_=str(output),
                                       node_type_='Output', io_type_='Dir', io_params_=['path_'])
    args = SimpleNamespace(resources=str(resources), log=str(tmp_path / 'editor.log'), debug=True)
    runtime = _EditorRuntime(args)
    server = runtime.start(lambda jobs, plugins, cancellations: Server(args, jobs, plugins, cancellations, database_url=location))
    saved = []
    with TestClient(server.app) as client:
        try:
            for color in ((10, 20, 30), (70, 80, 90)):
                Image.new('RGB', (4, 5), color).save(source)
                result = client.post('/api/queue', json=graph)
                assert result.status_code == 202, result.text
                identity = result.json()['result']['task_id']
                def completed():
                    record = client.get('/api/queue/task/' + identity).json()['result']
                    return record if record['state'] in ('SUCCEEDED', 'FAILED') else None
                record = wait(completed)
                assert record['state'] == 'SUCCEEDED', record
                assert len(record['output_files']) == 1
                saved.append((record['output_files'][0], color))
        finally:
            runtime.close()
    (output / 'image.png').unlink()
    (output / 'report.txt').unlink()
    with TestClient(Server(args, database_url=location).app) as client:
        for file, color in saved:
            assert file['media_type'] == 'application/zip'
            with zipfile.ZipFile(io.BytesIO(client.get(file['url']).content)) as archive:
                assert Image.open(io.BytesIO(archive.read('image.png'))).getpixel((0, 0)) == color
                assert archive.read('report.txt').decode() == str(color)
                assert archive.getinfo('empty/').is_dir()


def test_directory_archive_restored_from_minio_and_removed_with_history(tmp_path, location, s3_service, monkeypatch):
    from types import SimpleNamespace
    from nndeploy.server import Server
    from test_editor_history_retention import age
    endpoint, environment, _ = s3_service
    monkeypatch.setenv('NNDEPLOY_DIR_ACCESS', environment['MINIO_ROOT_USER'])
    monkeypatch.setenv('NNDEPLOY_DIR_SECRET', environment['MINIO_ROOT_PASSWORD'])
    config = dict(backend='s3', bucket='nndeploy-tests', region='us-east-1', endpoint=endpoint,
                  access_key_env='NNDEPLOY_DIR_ACCESS', secret_key_env='NNDEPLOY_DIR_SECRET')
    args = SimpleNamespace(resources=str(tmp_path / 'editor'), debug=True)
    root = tmp_path / 'directory'
    root.mkdir()
    (root / 'value.txt').write_text('stored remotely')
    server = Server(args, database_url=location, storage_config=config, editor_history_retention_seconds=60)
    with TestClient(server.app) as client:
        idx, payload, _, graph = _pending_output(client, server, tmp_path)
        graph['node_repository_'][0].update(io_type_='Dir', path_=str(root))
        file = capture_files(graph, server.storage, payload['_editor_scope'], payload['id'], payload['_editor_token'])[0]
        server.queue.task_done(idx, ExecutionStatus(True, 'Completed'), {}, {})
        server.storage.resolve(file['file_id'])[0].unlink()
    restored = Server(args, database_url=location, storage_config=config, editor_history_retention_seconds=60)
    with TestClient(restored.app) as client:
        result = client.get(file['url'])
        assert result.status_code == 200, result.text
        with zipfile.ZipFile(io.BytesIO(result.content)) as archive:
            assert archive.read('value.txt') == b'stored remotely'
        with restored.storage.open(file['file_id']) as reader:
            age(restored, payload['id'])
            assert client.post('/api/queue/history/cleanup').json()['result']['deleted_history'] == 1
            restored.storage.cleanup()
            assert restored.db.get_file(file['file_id']) is not None
            assert reader.read(2) == b'PK'
        restored.storage.cleanup()
        assert restored.db.get_file(file['file_id']) is None
        assert client.get(file['url']).status_code == 404
