"""Existing Graph I/O declarations identify immutable initialization assets."""
import copy
import functools
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
import nndeploy.base
import nndeploy.dag

from nndeploy.server import Server
from nndeploy.server.config import ServerConfig
from nndeploy.server.resources import prepare_workflow
from test_db import db
from test_redis_queue import backend, redis_service


class ResourceNode(nndeploy.dag.Node):
    def __init__(self, name, inputs, outputs):
        super().__init__(name, inputs, outputs)
        self.set_key('test.server.ResourceNode')
        self.set_io_type(nndeploy.dag.IOType.Model)
        self.add_io_param('artifacts')
        self.set_output_type(str)
        self.artifacts = []

    def init(self):
        assert all(Path(path).exists() for path in self.artifacts)
        return nndeploy.base.Status.ok()

    def run(self):
        values = []
        for value in self.artifacts:
            path = Path(value)
            values.append((path / 'vocab.txt' if path.is_dir() else path).read_text())
        self.get_output(0).set('|'.join(values))
        return nndeploy.base.Status.ok()

    def serialize(self):
        value = json.loads(super().serialize())
        value['artifacts'] = self.artifacts
        return json.dumps(value)

    def deserialize(self, value):
        self.artifacts = json.loads(value)['artifacts']
        return super().deserialize(value)


class ResourceCreator(nndeploy.dag.NodeCreator):
    def create_node(self, name, inputs, outputs):
        self.node = ResourceNode(name, inputs, outputs)
        return self.node


creator = ResourceCreator()
nndeploy.dag.register_node('test.server.ResourceNode', creator)


def graph_with_resources(paths, node_type=ResourceNode):
    from nndeploy.basic import ConsolePrintNode
    edge = nndeploy.dag.Edge('text')
    graph = nndeploy.dag.Graph('assets')
    model = node_type('model', [], [edge])
    if hasattr(model, 'artifacts'):
        model.artifacts = list(paths)
    else:
        model.set_artifacts(list(paths))
    graph.add_node(model)
    sink = ConsolePrintNode('sink', [edge], [])
    graph.add_node(sink)
    return graph


def assets(path):
    path.mkdir()
    weights = path / 'weights.bin'
    weights.write_text('weights')
    vocab = path / 'vocab'
    vocab.mkdir()
    (vocab / 'vocab.txt').write_text('tokens')
    return weights, vocab


def settings(isolation='thread'):
    return {'name': 'assets', 'outputs': {'text': {'type': 'string', 'edge': 'text'}},
            'execution': {'isolation': isolation}}


@pytest.mark.parametrize('isolation', ['thread', 'process'])
def test_json_declared_resources_survive_source_and_draft_deletion(db, tmp_path, isolation):
    weights, vocab = assets(tmp_path / 'draft')
    workflow = json.loads(graph_with_resources([str(weights), str(vocab)]).serialize())
    path = tmp_path / 'draft' / 'model.json'
    path.write_text(json.dumps(workflow))
    sidecar = path.with_suffix('.server.json')
    sidecar.write_text(json.dumps(settings(isolation)))
    directory = tmp_path / 'server'
    plugin = directory / 'plugin'
    plugin.mkdir(parents=True)
    (plugin / 'resource_node.py').write_text('import test_declared_resources\n')
    url = db.url.render_as_string(hide_password=False)
    first = Server(workflow=path, resources=directory, data_dir=directory, database_url=url)
    with TestClient(first.app) as client:
        runtime = first.entries['assets']['source']['workflow']['node_repository_'][0]['artifacts']
        assert all(Path(value).is_relative_to(directory / 'assets') for value in runtime)
        assert json.loads(path.read_text()) == workflow
        assert json.loads(sidecar.read_text()) == settings(isolation)
        weights.unlink()
        (vocab / 'vocab.txt').unlink()
        vocab.rmdir()
        path.unlink()
        sidecar.unlink()
        response = client.post('/v1/server/assets/deploy', json={})
        assert response.status_code == 200, response.text
        assert response.json()['outputs'] == {'text': 'weights|tokens'}
    with TestClient(Server(resources=directory, data_dir=directory, database_url=url).app) as client:
        response = client.post('/v1/server/assets/deploy', json={})
        assert response.status_code == 200, response.text
        assert response.json()['outputs'] == {'text': 'weights|tokens'}


@pytest.mark.parametrize('source,isolation', [('graph', 'thread'), ('factory', 'thread'), ('factory', 'process')])
def test_code_graph_keeps_objects_and_declared_assets_after_source_deletion(tmp_path, source, isolation):
    weights, vocab = assets(tmp_path / 'original')
    graph = graph_with_resources([str(weights), str(vocab)])
    original_node = graph.get_node('model')
    arguments = {'graph': graph} if source == 'graph' else {
        'graph_factory': functools.partial(graph_with_resources, [str(weights), str(vocab)])}
    server = Server(**arguments, config=settings(isolation), resources=tmp_path / 'server', data_dir=tmp_path / 'server')
    with TestClient(server.app) as client:
        if isolation == 'thread':
            actual = server.entries['assets']['workers'][0].executor.runner.graph
            if source == 'graph':
                assert actual is graph and actual.get_node('model') is original_node
            assert all(Path(value).is_relative_to(tmp_path / 'server' / 'assets') for value in actual.get_node('model').artifacts)
        weights.unlink()
        (vocab / 'vocab.txt').unlink()
        vocab.rmdir()
        response = client.post('/v1/server/assets/deploy', json={})
        assert response.status_code == 200, response.text
        assert response.json()['outputs'] == {'text': 'weights|tokens'}


def test_nested_param_declarations_copy_only_inputs_and_leave_bound_and_output_paths(tmp_path):
    weights, vocab = assets(tmp_path / 'files')
    model = {'name_': 'model', 'io_type_': 'Model', 'param_': {'io_params_': ['custom_weights'], 'custom_weights': [str(weights), str(vocab)]}}
    output = {'name_': 'output', 'node_type_': 'Output', 'io_type_': 'Model', 'io_params_': ['artifact'], 'artifact': str(weights), 'path_': str(weights)}
    bound = {'name_': 'bound', 'io_type_': 'Binary', 'io_params_': ['artifact'], 'artifact': str(weights), 'path_': str(weights)}
    workflow = {'node_repository_': [{'name_': 'group', 'node_repository_': [model, output, bound]}]}
    before = copy.deepcopy(workflow)
    config = ServerConfig.model_validate({'inputs': {'file': {'type': 'file', 'node': ['group', 'bound']}}})
    prepared = prepare_workflow(workflow, config, tmp_path / 'server', tmp_path)
    nodes = prepared['node_repository_'][0]['node_repository_']
    assert nodes[1:] == [output, bound] and workflow == before
    saved = nodes[0]['param_']['custom_weights']
    assert Path(saved[0]).read_text() == 'weights' and (Path(saved[1]) / 'vocab.txt').read_text() == 'tokens'
    assert saved != model['param_']['custom_weights']


@pytest.mark.parametrize('value', ['missing.bin', 'https://example.invalid/model', 123, {'unexpected': 'path'}])
def test_declared_resources_fail_before_native_initialization_on_invalid_paths(tmp_path, value):
    workflow = {'node_repository_': [{'name_': 'model', 'io_type_': 'Model', 'io_params_': ['artifact'], 'artifact': value}]}
    with pytest.raises(ValueError, match='resource'):
        prepare_workflow(workflow, ServerConfig(), tmp_path / 'server', tmp_path)


def test_in_memory_model_value_is_not_mistaken_for_a_path(tmp_path):
    (tmp_path / 'model-bytes').write_text('unrelated file')
    workflow = {'node_repository_': [{'name_': 'model', 'param_': {'is_path_': False, 'model_value_': ['model-bytes']}}]}
    assert prepare_workflow(workflow, ServerConfig(), tmp_path / 'server', tmp_path) == workflow


def test_declared_relative_path_does_not_fall_back_to_an_unrelated_working_directory(tmp_path, monkeypatch):
    draft, cwd = tmp_path / 'draft', tmp_path / 'cwd'
    draft.mkdir()
    cwd.mkdir()
    (cwd / 'weights.bin').write_text('wrong weights')
    monkeypatch.chdir(cwd)
    workflow = {'io_type_': 'Model', 'io_params_': ['artifact'], 'artifact': 'weights.bin'}
    with pytest.raises(ValueError, match='does not exist'):
        prepare_workflow(workflow, ServerConfig(), tmp_path / 'server', draft)


def test_declared_directory_cannot_copy_symbolic_links_outside_the_resource(tmp_path):
    weights, vocab = assets(tmp_path / 'files')
    (vocab / 'outside').symlink_to(weights)
    workflow = {'io_type_': 'Dir', 'io_params_': ['artifact'], 'artifact': str(vocab)}
    with pytest.raises(ValueError, match='symbolic links'):
        prepare_workflow(workflow, ServerConfig(), tmp_path / 'server', tmp_path)
    assert not list((tmp_path / 'server' / 'assets').iterdir())


def test_failed_resource_reload_keeps_the_running_and_saved_graph(db, tmp_path):
    import asyncio
    weights, vocab = assets(tmp_path / 'files')
    workflow = json.loads(graph_with_resources([str(weights), str(vocab)]).serialize())
    directory = tmp_path / 'server'
    server = Server(workflow=workflow, config=settings(), resources=directory, data_dir=directory,
                    database_url=db.url.render_as_string(hide_password=False))
    async def scenario():
        await server.start()
        try:
            original = server.entries['assets']['workers'][0].executor.runner.graph
            changed = copy.deepcopy(workflow)
            changed['node_repository_'][0]['artifacts'] = ['missing-weights']
            with pytest.raises(ValueError, match='does not exist'):
                await server.reload_entry('assets', workflow=changed)
            assert server.entries['assets']['workers'][0].executor.runner.graph is original
            weights.unlink()
            (vocab / 'vocab.txt').unlink()
            vocab.rmdir()
            assert (await server.deploy('assets', {}))['outputs'] == {'text': 'weights|tokens'}
        finally:
            await server.close()
        restored = Server(resources=directory, data_dir=directory, database_url=db.url.render_as_string(hide_password=False))
        await restored.start()
        try:
            assert (await restored.deploy('assets', {}))['outputs'] == {'text': 'weights|tokens'}
        finally:
            await restored.close()
    asyncio.run(scenario())


def test_code_graph_resource_preparation_does_not_require_json_export(tmp_path, monkeypatch):
    weights, vocab = assets(tmp_path / 'files')
    graph = graph_with_resources([str(weights), str(vocab)])
    def unavailable():
        raise AssertionError('Code Graph must not require JSON serialization')
    monkeypatch.setattr(graph, 'serialize', unavailable)
    monkeypatch.setattr(graph.get_node('model'), 'serialize', unavailable)
    server = Server(graph=graph, config=settings(), resources=tmp_path / 'server', data_dir=tmp_path / 'server')
    with TestClient(server.app) as client:
        weights.unlink()
        (vocab / 'vocab.txt').unlink()
        vocab.rmdir()
        result = client.post('/v1/server/assets/deploy', json={})
        assert result.status_code == 200, result.text
        assert result.json()['outputs'] == {'text': 'weights|tokens'}


@pytest.mark.skipif(not hasattr(os, 'mkfifo'), reason='Requires POSIX FIFO')
@pytest.mark.parametrize('directory', [False, True])
def test_resource_files_and_directories_reject_special_files(tmp_path, directory):
    root = tmp_path / 'files'
    root.mkdir()
    fifo = root / 'pipe'
    os.mkfifo(fifo)
    workflow = {'io_type_': 'Dir' if directory else 'Binary', 'io_params_': ['artifact'],
                'artifact': str(root if directory else fifo)}
    with pytest.raises(ValueError, match='regular files'):
        prepare_workflow(workflow, ServerConfig(), tmp_path / 'server', tmp_path)


@pytest.mark.parametrize('isolation', ['thread', 'process'])
def test_factory_resources_survive_stop_start_and_registered_server_restart(db, tmp_path, isolation):
    import asyncio
    weights, vocab = assets(tmp_path / 'original')
    factory = functools.partial(graph_with_resources, [str(weights), str(vocab)])
    config = settings(isolation)
    config['execution']['workers'] = 2
    options = dict(graph_factory=factory, config=config, resources=tmp_path / 'server', data_dir=tmp_path / 'server',
                   database_url=db.url.render_as_string(hide_password=False))
    async def scenario():
        first = Server(**options)
        await first.start()
        try:
            weights.unlink()
            (vocab / 'vocab.txt').unlink()
            vocab.rmdir()
            await first.stop_entry('assets')
            await first.start_entry('assets')
            assert (await first.deploy('assets', {}))['outputs'] == {'text': 'weights|tokens'}
        finally:
            await first.close()
        # Persisted metadata must never instantiate Python code by itself.
        without_code = Server(resources=tmp_path / 'server', data_dir=tmp_path / 'server', database_url=options['database_url'])
        await without_code.start()
        try:
            assert 'assets' not in without_code.entries
        finally:
            await without_code.close()
        restored = Server(**options)
        await restored.start()
        try:
            assert (await restored.deploy('assets', {}))['outputs'] == {'text': 'weights|tokens'}
        finally:
            await restored.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('directory', [False, True])
def test_factory_rebuild_rejects_modified_saved_resources(tmp_path, directory):
    import asyncio
    weights, vocab = assets(tmp_path / 'original')
    factory = functools.partial(graph_with_resources, [str(weights), str(vocab)])
    async def scenario():
        server = Server(graph_factory=factory, config=settings(), resources=tmp_path / 'server', data_dir=tmp_path / 'server')
        await server.start()
        try:
            node = server.entries['assets']['workers'][0].executor.runner.graph.get_node('model')
            saved = Path(node.artifacts[1]) / 'vocab.txt' if directory else Path(node.artifacts[0])
            await server.stop_entry('assets')
            saved.write_text('unexpected content')
            with pytest.raises(ValueError, match='integrity'):
                await server.start_entry('assets')
            assert server.describe('assets')['status'] == 'failed'
        finally:
            await server.close()
    asyncio.run(scenario())


def test_new_factory_registration_adopts_existing_sources_and_retains_missing_sources(db, tmp_path):
    import asyncio
    weights, vocab = assets(tmp_path / 'original')
    options = dict(graph_factory=functools.partial(graph_with_resources, [str(weights), str(vocab)]),
                   config=settings(), resources=tmp_path / 'server', data_dir=tmp_path / 'server', database_url=db.url.render_as_string(hide_password=False))
    async def scenario():
        first = Server(**options)
        await first.start()
        await first.close()
        weights.write_text('updated')
        (vocab / 'vocab.txt').unlink()
        vocab.rmdir()
        second = Server(**options)
        await second.start()
        try:
            assert (await second.deploy('assets', {}))['outputs'] == {'text': 'updated|tokens'}
        finally:
            await second.close()
        weights.unlink()
        third = Server(**options)
        await third.start()
        try:
            assert (await third.deploy('assets', {}))['outputs'] == {'text': 'updated|tokens'}
        finally:
            await third.close()
    asyncio.run(scenario())


def test_failed_factory_reload_preserves_saved_resources(db, tmp_path):
    import asyncio
    weights, vocab = assets(tmp_path / 'original')
    options = dict(graph_factory=functools.partial(graph_with_resources, [str(weights), str(vocab)]),
                   config=settings(), resources=tmp_path / 'server', data_dir=tmp_path / 'server', database_url=db.url.render_as_string(hide_password=False))
    async def scenario():
        server = Server(**options)
        await server.start()
        try:
            saved = server.db.get_server_config(server.config_scope, 'assets')
            weights.unlink()
            (vocab / 'vocab.txt').unlink()
            vocab.rmdir()
            invalid = settings()
            invalid['outputs']['text']['edge'] = 'missing-edge'
            with pytest.raises(ValueError):
                await server.reload_entry('assets', config=invalid)
            assert server.db.get_server_config(server.config_scope, 'assets') == saved
            assert (await server.deploy('assets', {}))['outputs'] == {'text': 'weights|tokens'}
        finally:
            await server.close()
        restored = Server(**options)
        await restored.start()
        try:
            assert (await restored.deploy('assets', {}))['outputs'] == {'text': 'weights|tokens'}
        finally:
            await restored.close()
    asyncio.run(scenario())


def test_factory_resource_commit_receipt_loss_is_reconciled(db, tmp_path, monkeypatch):
    import asyncio
    weights, vocab = assets(tmp_path / 'original')
    options = dict(graph_factory=functools.partial(graph_with_resources, [str(weights), str(vocab)]),
                   config=settings(), resources=tmp_path / 'server', data_dir=tmp_path / 'server', database_url=db.url.render_as_string(hide_password=False))
    async def scenario():
        server = Server(**options)
        save = server.db.save_server_config
        lost = []
        def lose_receipt(scope, name, value, **kwargs):
            result = save(scope, name, value, **kwargs)
            if value.get('kind') == 'graph_factory' and not lost:
                lost.append(result)
                raise OSError('Committed but receipt lost')
            return result
        monkeypatch.setattr(server.db, 'save_server_config', lose_receipt)
        await server.start()
        try:
            assert lost and server.entries['assets']['saved_token'] == lost[0]
            weights.unlink()
            (vocab / 'vocab.txt').unlink()
            vocab.rmdir()
        finally:
            await server.close()
        restored = Server(**options)
        await restored.start()
        try:
            assert (await restored.deploy('assets', {}))['outputs'] == {'text': 'weights|tokens'}
        finally:
            await restored.close()
    asyncio.run(scenario())


class GetterResources(ResourceNode):
    def __init__(self, name, inputs, outputs):
        super().__init__(name, inputs, outputs)
        self._artifacts = self.artifacts
        del self.artifacts

    def get_artifacts(self):
        return self._artifacts

    def set_artifacts(self, value):
        self._artifacts[:] = value

    def init(self):
        assert all(Path(path).exists() for path in self._artifacts)
        return nndeploy.base.Status.ok()

    def run(self):
        self.get_output(0).set(Path(self._artifacts[0]).read_text())
        return nndeploy.base.Status.ok()


def test_factory_getter_setter_preserves_original_resource_declarations(tmp_path):
    import asyncio
    weights, _ = assets(tmp_path / 'original')
    async def scenario():
        factory = functools.partial(graph_with_resources, [str(weights)], GetterResources)
        server = Server(graph_factory=factory, config=settings(), resources=tmp_path / 'server', data_dir=tmp_path / 'server')
        await server.start()
        try:
            weights.unlink()
            await server.stop_entry('assets')
            await server.start_entry('assets')
            assert (await server.deploy('assets', {}))['outputs'] == {'text': 'weights'}
        finally:
            await server.close()
    asyncio.run(scenario())


def environment_resource_graph():
    root = Path(os.environ['NNDEPLOY_TEST_RESOURCE_ROOT'])
    return graph_with_resources([str(root / 'weights.bin'), str(root / 'vocab')])


def test_shared_factory_resources_restore_in_the_registered_workspace(backend, tmp_path, monkeypatch):
    import asyncio
    database, redis_config = backend
    weights, vocab = assets(tmp_path / 'original')
    monkeypatch.setenv('NNDEPLOY_TEST_RESOURCE_ROOT', str(tmp_path / 'original'))
    options = dict(graph_factory=environment_resource_graph, config=settings('process'),
                   resources=tmp_path / 'server', data_dir=tmp_path / 'server', redis_config=redis_config,
                   database_url=database.url.render_as_string(hide_password=False))
    async def scenario():
        server = Server(**options)
        await server.start()
        try:
            weights.unlink()
            (vocab / 'vocab.txt').unlink()
            vocab.rmdir()
            assert (await server.deploy('assets', {}))['outputs'] == {'text': 'weights|tokens'}
        finally:
            await server.close()
        restored = Server(**options)
        await restored.start()
        try:
            assert (await restored.deploy('assets', {}))['outputs'] == {'text': 'weights|tokens'}
        finally:
            await restored.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('absolute', [False, True])
def test_factory_cache_does_not_reinterpret_relative_paths_in_another_directory(tmp_path, monkeypatch, absolute):
    import asyncio
    first_dir, second_dir = tmp_path / 'first', tmp_path / 'second'
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / 'weights.bin').write_text('first directory')
    factory = functools.partial(graph_with_resources, [str(first_dir / 'weights.bin') if absolute else 'weights.bin'])
    async def scenario():
        monkeypatch.chdir(first_dir)
        first = Server(graph_factory=factory, config=settings(), resources=tmp_path / 'server', data_dir=tmp_path / 'server')
        await first.start()
        await first.close()
        (first_dir / 'weights.bin').unlink()
        monkeypatch.chdir(second_dir)
        second = Server(graph_factory=factory, config=settings(), resources=tmp_path / 'server', data_dir=tmp_path / 'server')
        try:
            if absolute:
                await second.start()
                assert (await second.deploy('assets', {}))['outputs'] == {'text': 'first directory'}
            else:
                with pytest.raises(ValueError, match='does not exist'):
                    await second.start()
        finally:
            await second.close()
    asyncio.run(scenario())


def test_factory_process_recovery_keeps_the_loaded_resource_version(tmp_path):
    import asyncio
    weights, vocab = assets(tmp_path / 'original')
    factory = functools.partial(graph_with_resources, [str(weights), str(vocab)])
    async def scenario():
        server = Server(graph_factory=factory, config=settings('process'), resources=tmp_path / 'server', data_dir=tmp_path / 'server')
        await server.start()
        try:
            worker = server.entries['assets']['workers'][0]
            old_pid = worker.pid
            # A replacement Worker must not silently change the loaded model.
            weights.write_text('changed')
            (vocab / 'vocab.txt').unlink()
            vocab.rmdir()
            worker.process.kill()
            await asyncio.to_thread(worker.process.join)
            assert (await server.deploy('assets', {}))['outputs'] == {'text': 'weights|tokens'}
            assert worker.pid != old_pid
        finally:
            await server.close()
    asyncio.run(scenario())
