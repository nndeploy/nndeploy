"""Managed model files use the existing upload, Graph and reference lifecycle."""
import asyncio
import copy
import functools
import io
import json
import os
from pathlib import Path
import zipfile

from fastapi.testclient import TestClient
import pytest
import nndeploy.dag

from nndeploy.server import Server
from nndeploy.server.errors import ServerError
from nndeploy.server.resources import prepare_workflow
from nndeploy.server.config import ServerConfig
from test_db import db
from test_declared_resources import graph_with_resources, settings, assets
from test_storage import s3_service
from test_redis_queue import backend, redis_service


def resource_graph(reference, directory=False):
    graph = graph_with_resources([reference])
    if directory:
        graph.get_node('model').set_io_type(nndeploy.dag.IOType.Dir)
    return graph


def payload(directory):
    if not directory:
        return b'weights', 'application/octet-stream'
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('vocab.txt', 'tokens')
        archive.writestr('nested/config.json', '{"enabled":true}')
    return stream.getvalue(), 'application/zip'


@pytest.mark.parametrize('source,isolation', [('workflow', 'thread'), ('workflow', 'process'),
                                              ('factory', 'thread'), ('factory', 'process'), ('graph', 'thread')])
@pytest.mark.parametrize('directory', [False, True])
def test_uploaded_model_resources_load_and_keep_existing_file_references(db, tmp_path, source, isolation, directory):
    options = dict(data_dir=tmp_path / 'server', database_url=db.url.render_as_string(hide_password=False),
                   admin_token='m' * 32)
    server = Server(**options)
    plugin = tmp_path / 'server' / 'plugin'
    plugin.mkdir()
    (plugin / 'resource_node.py').write_text('import test_declared_resources\n')
    headers = {'Authorization': 'Bearer ' + 'm' * 32}
    expected = {'text': 'tokens' if directory else 'weights'}
    with TestClient(server.app) as client:
        data, media = payload(directory)
        uploaded = client.post('/v1/files', files={'file': ('model', data, media)})
        assert uploaded.status_code == 201, uploaded.text
        reference = {'file_id': uploaded.json()['file_id']}
        graph = resource_graph(reference, directory)
        workflow = json.loads(graph.serialize())
        original = copy.deepcopy(workflow)
        arguments = ({'workflow': workflow} if source == 'workflow' else {'graph': graph} if source == 'graph' else
                     {'graph_factory': functools.partial(resource_graph, reference, directory)})
        server.add(**arguments, config=settings(isolation))
        started = client.post('/api/server/assets/start', headers=headers)
        assert started.status_code == 200, started.text
        result = client.post('/v1/server/assets/deploy', json={})
        assert result.status_code == 200, result.text
        assert result.json()['outputs'] == expected
        assert workflow == original
        assert client.delete('/v1/files/' + reference['file_id'], headers=headers).status_code == 409
        assert client.post('/api/server/assets/stop', headers=headers).status_code == 200
        if source != 'graph':
            assert client.delete('/v1/files/' + reference['file_id'], headers=headers).status_code == 409
        assert client.post('/api/server/assets/start', headers=headers).status_code == 200
        assert client.post('/v1/server/assets/deploy', json={}).json()['outputs'] == expected
    if source != 'graph':
        restored = Server(**options, **({'graph_factory': arguments['graph_factory'], 'config': settings(isolation)}
                                      if source == 'factory' else {}))
        with TestClient(restored.app) as client:
            result = client.post('/v1/server/assets/deploy', json={})
            assert result.status_code == 200, result.text
            assert result.json()['outputs'] == expected
            assert client.delete('/v1/files/' + reference['file_id'], headers=headers).status_code == 409


@pytest.mark.parametrize('directory', [False, True])
def test_json_restore_rejects_modified_managed_resource_content(tmp_path, directory):
    weights, vocab = assets(tmp_path / 'original')
    workflow = json.loads(graph_with_resources([str(vocab if directory else weights)]).serialize())
    async def scenario():
        first = Server(workflow=workflow, config=settings(), data_dir=tmp_path / 'server')
        await first.start()
        path = Path(first.entries['assets']['source']['workflow']['node_repository_'][0]['artifacts'][0])
        await first.close()
        (path / 'vocab.txt' if directory else path).write_text('corrupt')
        second = Server(data_dir=tmp_path / 'server')
        try:
            with pytest.raises(ValueError, match='integrity'):
                await second.start()
        finally:
            await second.close()
    asyncio.run(scenario())


def test_duplicate_asset_copy_rejects_corrupted_existing_destination(tmp_path):
    weights, _ = assets(tmp_path / 'original')
    original = {'io_type_': 'Model', 'io_params_': ['weights'], 'weights': str(weights)}
    config = ServerConfig()
    prepared = prepare_workflow(original, config, tmp_path / 'server', tmp_path)
    Path(prepared['weights']).write_text('corrupt')
    with pytest.raises(ValueError, match='integrity'):
        prepare_workflow(original, config, tmp_path / 'server', tmp_path)


def test_model_file_cannot_be_deleted_during_native_initialization(tmp_path, monkeypatch):
    from test_declared_resources import ResourceNode
    server = Server(data_dir=tmp_path / 'server')
    record = server.storage.save(io.BytesIO(b'weights'), 'application/octet-stream')
    server.add(graph_factory=functools.partial(resource_graph, {'file_id': record['file_id']}), config=settings())
    initialize = ResourceNode.init
    checked = []
    def init(node):
        with pytest.raises(ServerError) as failure:
            server.storage.remove(record['file_id'])
        assert failure.value.code == 'file_in_use'
        checked.append(True)
        return initialize(node)
    monkeypatch.setattr(ResourceNode, 'init', init)
    async def scenario():
        try:
            await server.start()
            assert checked
            assert (await server.deploy('assets', {}))['outputs'] == {'text': 'weights'}
        finally:
            await server.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('directory', [False, True])
def test_object_storage_model_resources_download_before_graph_initialization(db, s3_service, tmp_path, monkeypatch, directory):
    endpoint, environment, _ = s3_service
    monkeypatch.setenv('NNDEPLOY_TEST_ACCESS', environment['MINIO_ROOT_USER'])
    monkeypatch.setenv('NNDEPLOY_TEST_SECRET', environment['MINIO_ROOT_PASSWORD'])
    storage_config = dict(backend='s3', bucket='nndeploy-tests', region='us-east-1', endpoint=endpoint,
                          access_key_env='NNDEPLOY_TEST_ACCESS', secret_key_env='NNDEPLOY_TEST_SECRET')
    options = dict(data_dir=tmp_path / 'server', database_url=db.url.render_as_string(hide_password=False),
                   storage_config=storage_config)
    server = Server(**options)
    plugin = tmp_path / 'server' / 'plugin'
    plugin.mkdir()
    (plugin / 'resource_node.py').write_text('import test_declared_resources\n')
    data, media = payload(directory)
    record = server.storage.save(io.BytesIO(data), media)
    cached, _ = server.storage.resolve(record['file_id'])
    cached.unlink()
    workflow = json.loads(resource_graph({'file_id': record['file_id']}, directory).serialize())
    server.add(workflow=workflow, config=settings('process'))
    async def scenario():
        await server.start()
        try:
            assert cached.exists()
            assert (await server.deploy('assets', {}))['outputs'] == {'text': 'tokens' if directory else 'weights'}
        finally:
            await server.close()
        cached.unlink()
        restored = Server(**options)
        await restored.start()
        try:
            assert cached.exists()
            assert (await restored.deploy('assets', {}))['outputs'] == {'text': 'tokens' if directory else 'weights'}
        finally:
            await restored.close()
    asyncio.run(scenario())


def test_corrupt_model_upload_cache_is_rejected_and_failed_start_releases_reference(tmp_path):
    server = Server(data_dir=tmp_path / 'server')
    record = server.storage.save(io.BytesIO(b'weights'), 'application/octet-stream')
    path, _ = server.storage.resolve(record['file_id'])
    path.write_bytes(b'corrupt')
    server.add(graph_factory=functools.partial(resource_graph, {'file_id': record['file_id']}), config=settings())
    async def scenario():
        try:
            with pytest.raises(ServerError) as failure:
                await server.start()
            assert failure.value.code == 'content_mismatch'
            assert server.db.get_server_config(server.config_scope, 'assets') is None
            server.storage.remove(record['file_id'])
        finally:
            await server.close()
        assert not list((tmp_path / 'server' / 'assets').glob('.materialize-*'))
    asyncio.run(scenario())


def test_directory_model_resource_rejects_an_ordinary_uploaded_file(tmp_path):
    server = Server(data_dir=tmp_path / 'server')
    record = server.storage.save(io.BytesIO(b'weights'), 'application/octet-stream')
    server.add(graph_factory=functools.partial(resource_graph, {'file_id': record['file_id']}, True), config=settings())
    async def scenario():
        try:
            with pytest.raises(ValueError, match='ZIP'):
                await server.start()
            server.storage.remove(record['file_id'])
        finally:
            await server.close()
    asyncio.run(scenario())


def test_native_codec_deserializes_a_managed_file_reference_before_loading(tmp_path):
    from PIL import Image
    from test_native_media import image_graph
    server = Server(data_dir=tmp_path / 'server')
    data = io.BytesIO()
    Image.new('RGB', (8, 6), (12, 34, 56)).save(data, format='PNG')
    record = server.storage.save(io.BytesIO(data.getvalue()), 'image/png')
    workflow = json.loads(image_graph().serialize())
    workflow['node_repository_'][0]['path_'] = {'file_id': record['file_id']}
    server.add(workflow=workflow, config={'name': 'image', 'outputs': {'image': {'type': 'image', 'node': ['encode']}}})
    with TestClient(server.app) as client:
        response = client.post('/v1/server/image/deploy', json={})
        assert response.status_code == 200, response.text
        output = client.get('/v1/files/' + response.json()['outputs']['image']['file_id'] + '/content')
        with Image.open(io.BytesIO(output.content)) as image:
            assert image.size == (8, 6) and image.convert('RGB').getpixel((0, 0)) == (12, 34, 56)


def test_legacy_managed_model_path_is_checked_even_without_io_declarations(tmp_path):
    weights, _ = assets(tmp_path / 'original')
    workflow = {'model_path_': str(weights)}
    prepared = prepare_workflow(workflow, ServerConfig(), tmp_path / 'server', tmp_path)
    Path(prepared['model_path_']).unlink()
    with pytest.raises(ValueError, match='no longer exists'):
        prepare_workflow(prepared, ServerConfig(), tmp_path / 'server', tmp_path)


def test_asset_publication_rejects_a_redirected_checksum_directory_before_writing(tmp_path):
    from nndeploy.server.resources import _digest
    weights, _ = assets(tmp_path / 'original')
    outside = tmp_path / 'outside'
    outside.mkdir()
    root = tmp_path / 'server' / 'assets'
    root.mkdir(parents=True)
    (root / _digest(weights)).symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        prepare_workflow({'model_path_': str(weights)}, ServerConfig(), tmp_path / 'server', tmp_path)
    assert not list(outside.iterdir())


def test_failed_model_file_reload_keeps_old_file_and_releases_candidate_reference(db, tmp_path, monkeypatch):
    from test_declared_resources import ResourceNode
    options = dict(data_dir=tmp_path / 'server', database_url=db.url.render_as_string(hide_password=False))
    server = Server(**options)
    old = server.storage.save(io.BytesIO(b'weights'), 'application/octet-stream')
    candidate = server.storage.save(io.BytesIO(b'invalid'), 'application/octet-stream')
    workflow = json.loads(resource_graph({'file_id': old['file_id']}).serialize())
    changed = json.loads(resource_graph({'file_id': candidate['file_id']}).serialize())
    server.add(workflow=workflow, config=settings())
    initialize = ResourceNode.init
    def init(node):
        if Path(node.artifacts[0]).read_bytes() == b'invalid':
            raise ValueError('Invalid model')
        return initialize(node)
    monkeypatch.setattr(ResourceNode, 'init', init)
    async def scenario():
        await server.start()
        try:
            saved = server.db.get_server_config(server.config_scope, 'assets')
            with pytest.raises(ValueError, match='Invalid model'):
                await server.reload_entry('assets', workflow=changed)
            assert server.db.get_server_config(server.config_scope, 'assets') == saved
            assert (await server.deploy('assets', {}))['outputs'] == {'text': 'weights'}
            server.storage.remove(candidate['file_id'])
            with pytest.raises(ServerError) as failure:
                server.storage.remove(old['file_id'])
            assert failure.value.code == 'file_in_use'
        finally:
            await server.close()
        restored = Server(**options)
        await restored.start()
        try:
            assert (await restored.deploy('assets', {}))['outputs'] == {'text': 'weights'}
        finally:
            await restored.close()
    asyncio.run(scenario())


def shared_model_factory():
    return resource_graph({'file_id': os.environ['NNDEPLOY_TEST_MODEL_FILE_ID']},
                          os.environ['NNDEPLOY_TEST_MODEL_DIRECTORY'] == '1')


@pytest.mark.parametrize('source', ['workflow', 'factory'])
@pytest.mark.parametrize('directory', [False, True])
def test_shared_model_file_references_survive_one_server_exit_and_restart(backend, tmp_path, monkeypatch, source, directory):
    database, redis_config = backend
    options = dict(data_dir=tmp_path / 'server', database_url=database.url.render_as_string(hide_password=False),
                   redis_config=redis_config)
    first = Server(**options)
    plugin = tmp_path / 'server' / 'plugin'
    plugin.mkdir()
    (plugin / 'resource_node.py').write_text('import test_declared_resources\n')
    data, media = payload(directory)
    record = first.storage.save(io.BytesIO(data), media)
    monkeypatch.setenv('NNDEPLOY_TEST_MODEL_FILE_ID', record['file_id'])
    monkeypatch.setenv('NNDEPLOY_TEST_MODEL_DIRECTORY', '1' if directory else '0')
    arguments = ({'workflow': json.loads(resource_graph({'file_id': record['file_id']}, directory).serialize())}
                 if source == 'workflow' else {'graph_factory': shared_model_factory})
    first.add(**arguments, config=settings('process'))
    second = Server(**options, **arguments, config=settings('process'))
    expected = {'text': 'tokens' if directory else 'weights'}
    async def scenario():
        try:
            await first.start()
            await second.start()
            assert first.entries['assets']['workers'][0].pid != second.entries['assets']['workers'][0].pid
            assert (await first.deploy('assets', {}))['outputs'] == expected
            await first.close()
            assert (await second.deploy('assets', {}))['outputs'] == expected
            with pytest.raises(ServerError) as failure:
                second.storage.remove(record['file_id'])
            assert failure.value.code == 'file_in_use'
        finally:
            await first.close()
            await second.close()
        restored = Server(**options, **({'graph_factory': shared_model_factory, 'config': settings('process')}
                                       if source == 'factory' else {}))
        await restored.start()
        try:
            assert (await restored.deploy('assets', {}))['outputs'] == expected
        finally:
            await restored.close()
    asyncio.run(scenario())


def test_model_directory_uses_configured_file_capacity_above_the_legacy_archive_limit(tmp_path):
    # Functional size boundary: model directories must not keep the editor's
    # fixed 64 MiB archive cap after storage capacity was explicitly increased.
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('vocab.txt', 'tokens')
        with archive.open('weights.bin', 'w') as output:
            chunk = b'\x00' * 1024**2
            for _ in range(65):
                output.write(chunk)
    stream.seek(0)
    server = Server(data_dir=tmp_path / 'server', storage_config={'max_bytes': 128 * 1024**2})
    async def scenario():
        try:
            record = server.storage.save(stream, 'application/zip')
            server.add(graph_factory=functools.partial(resource_graph, {'file_id': record['file_id']}, True), config=settings())
            await server.start()
            node = server.entries['assets']['workers'][0].executor.runner.graph.get_node('model')
            directory = Path(node.artifacts[0])
            assert (directory / 'weights.bin').stat().st_size == 65 * 1024**2
            assert (await server.deploy('assets', {}))['outputs'] == {'text': 'tokens'}
        finally:
            await server.close()
    asyncio.run(scenario())
