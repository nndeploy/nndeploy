"""Initialization samples exercise actual execution before an entry is ready."""
import asyncio
import copy
import io
import json
import os
from pathlib import Path
import threading
import time

import httpx
import pytest
from PIL import Image
from pydantic import BaseModel, ValidationError
from sqlalchemy import func, select

from nndeploy.server import Server
from nndeploy.server.config import ServerConfig, suggest_config
from nndeploy.server.db import files, tasks, task_events
from nndeploy.server.errors import ServerError
from test_server_api import text_graph, image_graph


def counts(server):
    with server.db.engine.connect() as connection:
        return [connection.execute(select(func.count()).select_from(table)).scalar_one()
                for table in (tasks, files, task_events)]


def text_config(**execution):
    return {'inputs': {'text': {'type': 'string', 'node': ['source'], 'required': False, 'default': 'default'}},
            'outputs': {'text': {'type': 'string', 'edge': 'text'}},
            'execution': {'warmup': [{'inputs': {'text': 'warm'}}], **execution}}


@pytest.mark.parametrize('samples', [[{}] * 17, [{'inputs': {'huge': 'x' * 1024**2}}], [{'inputs': {'value': float('nan')}}]])
def test_warmup_configuration_is_bounded(samples):
    with pytest.raises(ValidationError):
        ServerConfig.model_validate({'execution': {'warmup': samples}})


@pytest.mark.parametrize('isolation', ['thread', 'process'])
def test_real_text_graph_warmup_resets_defaults_and_creates_no_task_records(tmp_path, isolation):
    marker = tmp_path / 'executed.txt'
    workflow = text_graph()
    workflow['node_repository_'][1]['frontend_params'].update(save_to_file=True, file_path=str(marker))
    server = Server(workflow=workflow, config=text_config(isolation=isolation), resources=tmp_path, data_dir=tmp_path)
    async def scenario():
        await server.start()
        try:
            assert marker.read_text().splitlines() == ['warm']
            assert counts(server) == [0, 0, 0]
            assert (await server.deploy('default', {}))['outputs'] == {'text': 'default'}
            assert (await server.deploy('default', {'text': 'actual'}))['outputs'] == {'text': 'actual'}
        finally:
            await server.close()
        restored = Server(resources=tmp_path, data_dir=tmp_path)
        await restored.start()
        try:
            assert marker.read_text().splitlines() == ['warm', 'default', 'actual', 'warm']
            assert counts(restored)[0] == 2
            assert (await restored.deploy('default', {}))['outputs']['text'] == 'default'
        finally:
            await restored.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('source', ['graph', 'graph_factory'])
def test_code_graph_warms_without_exporting_or_recreating_it(tmp_path, source):
    from nndeploy.dag import Graph
    marker = tmp_path / 'executed.txt'
    workflow = text_graph()
    workflow['node_repository_'][1]['frontend_params'].update(save_to_file=True, file_path=str(marker))
    created = []
    def factory():
        graph = Graph('echo')
        graph.deserialize(json.dumps(workflow))
        created.append(graph)
        return graph
    server = Server(**{source: factory if source == 'graph_factory' else factory()}, config=text_config(), resources=tmp_path, data_dir=tmp_path)
    async def scenario():
        await server.start()
        try:
            assert marker.read_text().splitlines() == ['warm']
            assert counts(server) == [0, 0, 0] and len(created) == 1
            assert server.entries['default']['workers'][0].executor.runner.graph is created[0]
            assert (await server.deploy('default', {}))['outputs'] == {'text': 'default'}
            assert len(created) == 1
        finally:
            await server.close()
        assert not created[0].get_initialized()
    asyncio.run(scenario())


@pytest.mark.parametrize('isolation', ['thread', 'process'])
def test_real_image_warmup_is_ephemeral_and_retains_input_for_restart(tmp_path, isolation):
    server = Server(resources=tmp_path, data_dir=tmp_path)
    picture = io.BytesIO()
    Image.new('RGB', (7, 9), (20, 30, 40)).save(picture, format='PNG')
    picture.seek(0)
    identity = server.storage.save(picture)['file_id']
    config = suggest_config(image_graph())
    config['execution'] = {'isolation': isolation, 'warmup': [{'inputs': {'decode': {'file_id': identity}}}]}
    server.add(workflow=image_graph(), config=config)
    async def scenario():
        await server.start()
        try:
            assert counts(server) == [0, 1, 0]
            assert not list((tmp_path / 'tmp').rglob('request-*'))
            result = await server.deploy('image', {'decode': {'file_id': identity}})
            with server.storage.open(result['outputs']['encode']['file_id']) as stream:
                with Image.open(stream) as actual:
                    assert actual.size == (7, 9) and actual.getpixel((0, 0)) == (20, 30, 40)
        finally:
            await server.close()
        with pytest.raises(ServerError) as protected:
            server.storage.remove(identity)
        assert protected.value.code == 'file_in_use'
        restored = Server(resources=tmp_path, data_dir=tmp_path)
        await restored.start()
        try:
            assert counts(restored)[:2] == [1, 2]
        finally:
            await restored.close()
    asyncio.run(scenario())


def test_start_is_not_ready_until_every_worker_has_warmed_on_its_owner_thread(tmp_path):
    entered, release = threading.Event(), threading.Event()
    handlers = []
    class Handler:
        def __init__(self):
            self.owner = threading.get_ident()
            self.calls = []
            handlers.append(self)
        def deploy(self, values):
            assert threading.get_ident() == self.owner
            self.calls.append(copy.deepcopy(values))
            entered.set()
            assert release.wait(5)
            return values
    config = {'inputs': {'text': {'type': 'string'}},
              'execution': {'workers': 2, 'warmup': [{'inputs': {'text': 'warm'}}]}}
    server = Server(handler_factory=Handler, config=config, resources=tmp_path, data_dir=tmp_path)
    async def scenario():
        starting = asyncio.create_task(server.start())
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            assert server.describe('default')['status'] == 'starting'
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                assert (await client.get('/health/ready')).status_code == 503
                assert (await client.post('/v1/server/default/deploy', json={'inputs': {'text': 'real'}})).status_code == 503
            release.set()
            await starting
            assert len(handlers) == 2
            assert all(handler.calls == [{'text': 'warm'}] for handler in handlers)
            assert handlers[0].owner != handlers[1].owner
            assert counts(server) == [0, 0, 0]
        finally:
            release.set()
            await starting
            await server.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('scheduling', ['serial', 'concurrent', 'engine'])
@pytest.mark.parametrize('streaming', [False, True])
def test_code_warmup_uses_typed_inputs_stream_limits_and_output_validation(tmp_path, scheduling, streaming):
    seen, owners = [], []
    class Input(BaseModel):
        text: str
    class Output(BaseModel):
        text: str
    def record(values):
        seen.append(copy.deepcopy(values))
        owners.append(threading.get_ident())
        return values
    def sync(values):
        return record(values)
    def sync_stream(values):
        yield record(values)
    async def asynchronous(values):
        await asyncio.sleep(0)
        return record(values)
    async def async_stream(values):
        yield record(values)
    target = (async_stream if streaming else asynchronous) if scheduling == 'engine' else (sync_stream if streaming else sync)
    config = {'execution': {'scheduling': scheduling, 'warmup': [{'inputs': {'text': 'warm'}}]},
              'streaming': {'enabled': streaming}}
    server = Server(deploy=target, config=config, input_model=Input, output_model=Output, resources=tmp_path, data_dir=tmp_path)
    async def scenario():
        await server.start()
        try:
            assert seen == [{'text': 'warm'}] and owners[0] != threading.get_ident()
            assert counts(server) == [0, 0, 0]
            assert (await server.deploy('default', {'text': 'real'}))['outputs'] == {'text': 'real'}
            assert seen == [{'text': 'warm'}, {'text': 'real'}]
        finally:
            await server.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('failure', ['input', 'output', 'stream'])
def test_invalid_warmup_fails_start_and_releases_handler(tmp_path, failure):
    closed = []
    class Handler:
        def deploy(self, values):
            if failure == 'stream':
                return iter_values(values)
            return {'text': 12} if failure == 'output' else values
        def deinit(self):
            closed.append(True)
    def iter_values(values):
        yield values
        yield values
    config = {'inputs': {'text': {'type': 'string'}}, 'outputs': {'text': {'type': 'string'}},
              'execution': {'warmup': [{'inputs': {'text': 123 if failure == 'input' else 'warm'}}]},
              'streaming': {'enabled': failure == 'stream', 'max_events': 1}}
    server = Server(handler_factory=Handler, config=config, resources=tmp_path, data_dir=tmp_path)
    async def scenario():
        try:
            with pytest.raises((ValueError, ServerError)):
                await server.start()
            assert closed == [True]
            assert server.describe('default')['status'] == 'failed'
            assert not server.entries['default']['workers']
            assert counts(server) == [0, 0, 0]
        finally:
            await server.close()
    asyncio.run(scenario())


def test_failed_reload_warmup_preserves_live_graph_and_saved_configuration(tmp_path):
    config = text_config()
    server = Server(workflow=text_graph(), config=config, resources=tmp_path, data_dir=tmp_path)
    async def scenario():
        await server.start()
        old = server.entries['default']['workers'][0]
        saved = server.db.get_server_config(server.config_scope, 'default')
        invalid = copy.deepcopy(config)
        invalid['execution']['warmup'] = [{'inputs': {'unknown': 'broken'}}]
        try:
            with pytest.raises(ValueError, match='Unknown'):
                await server.reload_entry('default', config=invalid)
            assert server.entries['default']['workers'][0] is old
            assert (await server.deploy('default', {}))['outputs'] == {'text': 'default'}
            assert server.db.get_server_config(server.config_scope, 'default') == saved
        finally:
            await server.close()
    asyncio.run(scenario())


def test_reload_from_untyped_code_uses_new_schema_for_warmup_and_requests(tmp_path):
    seen = []
    def deploy(values):
        seen.append(copy.deepcopy(values))
        return values
    server = Server(deploy=deploy, resources=tmp_path, data_dir=tmp_path)
    config = {'inputs': {'text': {'type': 'string'}}, 'execution': {'warmup': [{'inputs': {'text': 123}}]}}
    async def scenario():
        await server.start()
        try:
            with pytest.raises(ValueError, match='Expected string'):
                await server.reload_entry('default', config=config)
            assert not seen
            assert (await server.deploy('default', {'old': True}))['outputs'] == {'old': True}
            config['execution']['warmup'][0]['inputs']['text'] = 'warm'
            await server.reload_entry('default', config=config)
            assert seen[-1] == {'text': 'warm'}
            with pytest.raises(ServerError) as invalid:
                await server.deploy('default', {'text': 123})
            assert invalid.value.code == 'invalid_request'
            assert (await server.deploy('default', {'text': 'real'}))['outputs'] == {'text': 'real'}
        finally:
            await server.close()
    asyncio.run(scenario())


def test_dynamic_batch_warmup_uses_the_actual_batch_method(tmp_path):
    calls = []
    class Handler:
        def batch_key(self, values):
            return 'compatible'
        def deploy_batch(self, items):
            calls.append(copy.deepcopy(items))
            return [item['inputs'] for item in items]
    config = {'inputs': {'text': {'type': 'string'}},
              'execution': {'batching': {'enabled': True}, 'warmup': [{'inputs': {'text': 'warm'}}]}}
    server = Server(handler_factory=Handler, config=config, resources=tmp_path, data_dir=tmp_path)
    async def scenario():
        await server.start()
        try:
            assert len(calls) == 1 and calls[0][0]['inputs'] == {'text': 'warm'}
            assert counts(server) == [0, 0, 0]
            assert (await server.deploy('default', {'text': 'real'}))['outputs'] == {'text': 'real'}
            assert len(calls) == 2
        finally:
            await server.close()
    asyncio.run(scenario())


def blocked_process(values):
    Path(values['marker']).write_text(str(os.getpid()))
    time.sleep(30)
    return values


def test_process_startup_deadline_includes_warmup_and_confirms_exit(tmp_path):
    marker = tmp_path / 'pid'
    config = {'inputs': {'marker': {'type': 'string'}},
              'execution': {'isolation': 'process', 'startup_timeout_seconds': 2.0,
                            'warmup': [{'inputs': {'marker': str(marker)}}]}}
    server = Server(deploy=blocked_process, config=config, resources=tmp_path, data_dir=tmp_path)
    async def scenario():
        try:
            with pytest.raises(ServerError) as timeout:
                await server.start()
            assert timeout.value.code == 'startup_timeout'
            assert marker.exists(), 'The deadline must expire during actual warmup, not before child imports'
            with pytest.raises(ProcessLookupError):
                os.kill(int(marker.read_text()), 0)
            assert not server.entries['default']['workers']
            assert counts(server) == [0, 0, 0]
        finally:
            await server.close()
    asyncio.run(scenario())


def test_thread_warmup_deadline_waits_for_actual_computation_before_releasing(tmp_path):
    entered, release = threading.Event(), threading.Event()
    closed = []
    class Handler:
        def deploy(self, values):
            entered.set()
            assert release.wait(5)
            return values
        def deinit(self):
            closed.append(True)
    server = Server(handler_factory=Handler, config={'execution': {
        'startup_timeout_seconds': .1, 'warmup': [{}]}}, resources=tmp_path, data_dir=tmp_path)
    async def scenario():
        starting = asyncio.create_task(server.start())
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            await asyncio.sleep(.15)
            assert not starting.done() and not closed
            release.set()
            with pytest.raises(ServerError) as timeout:
                await starting
            assert timeout.value.code == 'startup_timeout'
            assert closed == [True]
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())


def test_engine_warmup_timeout_waits_for_async_cleanup_and_never_becomes_ready(tmp_path):
    cleaning, release = threading.Event(), threading.Event()
    closed = []
    class Handler:
        async def deploy(self, values):
            try:
                await asyncio.sleep(30)
            finally:
                cleaning.set()
                while not release.is_set():
                    await asyncio.sleep(.01)
        async def deinit(self):
            closed.append(True)
    server = Server(handler_factory=Handler, config={'execution': {
        'scheduling': 'engine', 'startup_timeout_seconds': .1, 'warmup': [{}]}}, resources=tmp_path, data_dir=tmp_path)
    async def scenario():
        starting = asyncio.create_task(server.start())
        try:
            assert await asyncio.to_thread(cleaning.wait, 5)
            assert not starting.done() and not closed
            assert server.describe('default')['status'] == 'starting'
            release.set()
            with pytest.raises(ServerError) as timeout:
                await starting
            assert timeout.value.code == 'startup_timeout'
            assert closed == [True] and counts(server) == [0, 0, 0]
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())
