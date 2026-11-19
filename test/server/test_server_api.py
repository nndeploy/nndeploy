"""Server's real public interface: native Graphs and controlled concurrency."""

import asyncio
import copy
import io
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import BaseModel

from nndeploy.server import Server
from nndeploy.server.errors import ServerError


def text_graph():
    return dict(key_='nndeploy.dag.Graph', name_='echo', parallel_type_='kParallelTypeSequential', node_repository_=[
        dict(key_='nndeploy.basic.InputStr', name_='source', str_='', inputs_=[], outputs_=[dict(name_='text', type_='str')]),
        dict(key_='nndeploy.basic.ConsolePrintNode', name_='sink', inputs_=[dict(name_='text', type_='str')], outputs_=[],
             frontend_params=dict(echo_to_stdout=False, print_to_logger=False, save_to_file=False)),
    ])


def image_graph():
    return dict(key_='nndeploy.dag.Graph', name_='image', parallel_type_='kParallelTypeSequential', node_repository_=[
        dict(key_='nndeploy.codec.PILImageDecodec', name_='decode', path_='', color_mode_='RGB', inputs_=[],
             outputs_=[dict(name_='image', type_='PIL.Image.Image')]),
        dict(key_='nndeploy.codec.PILImageEncodec', name_='encode', path_='unused.png', inputs_=[dict(name_='image', type_='PIL.Image.Image')], outputs_=[]),
    ])


def test_api_only_does_not_import_editor_or_modelscope(tmp_path):
    code = '''import sys
import nndeploy
loaded_before_server = set(sys.modules)
from nndeploy.server import Server
server = Server(resources=sys.argv[1], data_dir=sys.argv[1])
assert 'nndeploy.server.frontend' not in sys.modules
assert 'nndeploy.server.template' not in sys.modules
assert 'modelscope' not in sys.modules
# Optional PyTorch remote-cache support can import Redis during nndeploy import.
assert 'redis' not in sys.modules or 'redis' in loaded_before_server
assert not any(route.path == '/api/nodes/upload' for route in server.app.routes)
'''
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_native_graph_reused_parameters_reset_and_records_survive_restart(tmp_path):
    config = dict(inputs={'text': dict(node=['source'], type='string', required=False, default='default')},
                  outputs={'text': dict(edge='text', type='string')})
    server = Server(workflow=text_graph(), config=config, name='echo', resources=tmp_path, data_dir=tmp_path)
    with TestClient(server.app) as client:
        assert client.get('/health/ready').status_code == 200
        first = client.post('/v1/server/echo/deploy', json={'inputs': {'text': 'first'}})
        assert first.status_code == 200, first.text
        graph = server.entries['echo']['workers'][0].executor.runner.graph
        assert client.post('/v1/server/echo/deploy', json={}).json()['outputs'] == {'text': 'default'}
        assert server.entries['echo']['workers'][0].executor.runner.graph is graph
        assert client.post('/v1/server/echo/deploy', json={'inputs': {'unknown': 1}}).status_code == 422
    with TestClient(Server(resources=tmp_path, data_dir=tmp_path).app) as client:
        record = client.get('/v1/tasks/' + first.json()['task_id']).json()
        assert record['state'] == 'succeeded' and record['outputs'] == {'text': 'first'}


def test_native_image_file_roundtrip_and_independent_outputs(tmp_path):
    with TestClient(Server(workflow=image_graph(), resources=tmp_path, data_dir=tmp_path).app) as client:
        picture = io.BytesIO()
        Image.new('RGB', (5, 7), (20, 40, 60)).save(picture, format='PNG')
        uploaded = client.post('/v1/files', files={'file': ('input.png', picture.getvalue(), 'image/png')})
        assert uploaded.status_code == 201, uploaded.text
        references = []
        for _ in range(2):
            result = client.post('/v1/server/image/deploy', json={'inputs': {'decode': {'file_id': uploaded.json()['file_id']}}})
            assert result.status_code == 200, result.text
            reference = result.json()['outputs']['encode']['file_id']
            references.append(reference)
            content = client.get(f'/v1/files/{reference}/content')
            output = Image.open(io.BytesIO(content.content))
            assert output.size == (5, 7) and output.getpixel((0, 0)) == (20, 40, 60)
        assert references[0] != references[1]
        assert client.post('/v1/files', files={'file': ('bad.png', b'not an image', 'image/png')}).status_code == 422


def test_q01_q02_slow_entry_does_not_block_other_entry(tmp_path):
    entered, release = threading.Event(), threading.Event()
    def blocked(values):
        entered.set()
        assert release.wait(10)
        return values
    server = Server(resources=tmp_path, data_dir=tmp_path)
    server.add(name='slow', deploy=blocked, config={'inputs': {'value': {'type': 'integer'}}, 'execution': {'max_queue_size': 0}})
    server.add(name='fast', deploy=lambda values: values)
    with TestClient(server.app) as client, ThreadPoolExecutor() as pool:
        running = pool.submit(client.post, '/v1/server/slow/deploy', json={'inputs': {'value': 1}})
        try:
            assert entered.wait(5)
            full = client.post('/v1/server/slow/deploy', json={'inputs': {'value': 2}})
            assert full.status_code == 429
            fast = client.post('/v1/server/fast/deploy', json={'inputs': {'value': 3}})
            assert fast.status_code == 200 and fast.json()['outputs'] == {'value': 3}
            assert not running.done()
        finally:
            release.set()
        assert running.result().status_code == 200


def test_worker_thread_owns_initialization_execution_and_close(tmp_path):
    calls = []
    class Handler:
        def init(self): calls.append(('init', threading.get_ident()))
        def deploy(self, inputs):
            calls.append(('deploy', threading.get_ident()))
            return inputs
        def deinit(self): calls.append(('deinit', threading.get_ident()))
    with TestClient(Server(handler_factory=Handler, resources=tmp_path, data_dir=tmp_path).app) as client:
        for _ in range(2):
            assert client.post('/v1/server/default/deploy', json={'inputs': {'text': 'test'}}).status_code == 200
    assert [name for name, _ in calls] == ['init', 'deploy', 'deploy', 'deinit']
    assert len({identity for _, identity in calls}) == 1


def test_q04_two_workers_can_execute_same_entry_in_parallel(tmp_path):
    import nndeploy.base
    import nndeploy.dag
    barrier = threading.Barrier(2)
    instances = []
    class Copy(nndeploy.dag.Node):
        def __init__(self, name, inputs, outputs):
            super().__init__(name, inputs, outputs)
            self.set_input_type(str)
            self.set_output_type(str)
        def run(self):
            value = self.get_input(0).get(self)
            barrier.wait(timeout=5)
            self.get_output(0).set(value)
            return nndeploy.base.Status.ok()
    def factory():
        source, target = nndeploy.dag.Edge('input'), nndeploy.dag.Edge('output')
        graph = nndeploy.dag.Graph('parallel', [source], [target])
        node = Copy('copy', [source], [target])
        graph.add_node(node)
        instances.append((graph, node))
        return graph
    config = {'inputs': {'value': {'type': 'string', 'edge': 'input'}},
              'outputs': {'value': {'type': 'string', 'edge': 'output'}}}
    with TestClient(Server(graph_factory=factory, config=config, workers=2, resources=tmp_path, data_dir=tmp_path).app) as client, ThreadPoolExecutor() as pool:
        futures = [pool.submit(client.post, '/v1/server/default/deploy', json={'inputs': {'value': str(i)}}) for i in range(2)]
        assert [f.result().json()['outputs']['value'] for f in futures] == ['0', '1']
        assert len(instances) == 2 and instances[0][0] is not instances[1][0] and instances[0][1] is not instances[1][1]


def test_http_wait_cancellation_keeps_slot_and_native_work(tmp_path):
    entered, release = threading.Event(), threading.Event()
    def blocked(values):
        entered.set()
        assert release.wait(10)
        return values
    async def run():
        server = Server(deploy=blocked, resources=tmp_path, data_dir=tmp_path, config={'execution': {'max_queue_size': 0}})
        await server.start()
        waiter = asyncio.create_task(server.deploy('default', {}))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            with pytest.raises(ServerError) as error:
                await server.deploy('default', {})
            assert error.value.status_code == 429
            assert server.inflight == 1
        finally:
            release.set()
            await server.close()
        assert server.inflight == 0
    asyncio.run(run())


def test_failed_admission_never_executes(tmp_path, monkeypatch):
    calls = []
    server = Server(deploy=lambda inputs: calls.append(inputs), resources=tmp_path, data_dir=tmp_path)
    with TestClient(server.app) as client:
        def fail(*args): raise OSError('database unavailable')
        monkeypatch.setattr(server.db, 'create_task', fail)
        response = client.post('/v1/server/default/deploy', json={})
        assert response.status_code == 503
        assert not calls and server.inflight == 0


def test_native_failure_rebuilds_graph(tmp_path, monkeypatch):
    from nndeploy.basic.input import InputStr
    original = InputStr.run
    def run(node):
        if node.str_ == 'fail':
            raise ValueError('broken native execution')
        return original(node)
    monkeypatch.setattr(InputStr, 'run', run)
    server = Server(workflow=text_graph(), resources=tmp_path, data_dir=tmp_path)
    with TestClient(server.app) as client:
        original_graph = server.entries['echo']['workers'][0].executor.runner.graph
        failed = client.post('/v1/server/echo/deploy', json={'inputs': {'source': 'fail'}})
        assert failed.status_code == 500
        result = client.post('/v1/server/echo/deploy', json={'inputs': {'source': 'recovered'}})
        assert result.status_code == 200 and result.json()['outputs'] == {'sink': 'recovered'}
        assert server.entries['echo']['workers'][0].executor.runner.graph is not original_graph


def test_graph_factory_discovers_bindings(tmp_path):
    from nndeploy.dag import Graph
    def factory():
        graph = Graph('echo')
        graph.deserialize(json.dumps(text_graph()))
        return graph
    with TestClient(Server(graph_factory=factory, name='factory', resources=tmp_path, data_dir=tmp_path).app) as client:
        result = client.post('/v1/server/factory/deploy', json={'inputs': {'source': 'factory'}})
        assert result.status_code == 200, result.text
        assert result.json()['outputs'] == {'sink': 'factory'}


def test_async_function_and_typed_input(tmp_path):
    class Input(BaseModel):
        value: int
    class Output(BaseModel):
        value: int
    async def deploy(values):
        await asyncio.sleep(0)
        return {'value': values['value'] + 1}
    with TestClient(Server(deploy=deploy, input_model=Input, output_model=Output, resources=tmp_path, data_dir=tmp_path).app) as client:
        assert client.post('/v1/server/default/deploy', json={'inputs': {'value': '1'}}).status_code == 422
        response = client.post('/v1/server/default/deploy', json={'inputs': {'value': 1}})
        assert response.json()['outputs'] == {'value': 2}


def test_management_auth_and_stop_restart(tmp_path):
    server = Server(workflow=text_graph(), resources=tmp_path, data_dir=tmp_path, admin_token='test-token')
    headers = {'Authorization': 'Bearer test-token'}
    with TestClient(server.app) as client:
        assert client.post('/api/server/echo/stop').status_code == 401
        assert client.post('/api/server/echo/stop', headers=headers).json()['status'] == 'stopped'
        assert client.post('/v1/server/echo/deploy', json={'inputs': {'source': 'stopped'}}).status_code == 503
        assert client.post('/api/server/echo/start', headers=headers).json()['status'] == 'ready'
        assert client.post('/v1/server/echo/deploy', json={'inputs': {'source': 'again'}}).json()['outputs'] == {'sink': 'again'}


def test_cli_api_mode_and_local_default():
    from nndeploy.server.app import cli
    args = cli(['--api-only', '--workflow', 'example.json'])
    assert args.host == '127.0.0.1'
    assert cli([]).host == '0.0.0.0'
    with pytest.raises(SystemExit):
        cli(['--workflow', 'example.json'])
    with pytest.raises(SystemExit):
        cli(['--api-only'])


def test_q07_reload_failure_keeps_existing_graph_and_other_entry(tmp_path):
    async def scenario():
        server = Server(workflow=text_graph(), resources=tmp_path, data_dir=tmp_path)
        server.add(name='other', deploy=lambda values: values)
        await server.start()
        original = server.entries['echo']['workers'][0].executor.runner.graph
        broken = text_graph()
        broken['node_repository_'][0]['key_'] = 'missing.Node'
        try:
            with pytest.raises(Exception):
                await server.reload_entry('echo', workflow=broken)
            assert server.entries['echo']['status'] == 'ready'
            assert server.entries['echo']['workers'][0].executor.runner.graph is original
            assert (await server.deploy('echo', {'source': 'still ready'}))['outputs']['sink'] == 'still ready'
            assert (await server.deploy('other', {'ok': True}))['outputs'] == {'ok': True}
        finally:
            await server.close()
    asyncio.run(scenario())


def test_reload_accepts_new_requests_while_old_worker_drains(tmp_path):
    entered, release = threading.Event(), threading.Event()
    generations = []
    class Handler:
        def __init__(self):
            self.generation = len(generations)
            generations.append(self)
            self.closed = False
        def deploy(self, values):
            if values.get('block'):
                entered.set()
                assert release.wait(10)
            return {'generation': self.generation}
        def deinit(self):
            self.closed = True
    async def scenario():
        server = Server(handler_factory=Handler, resources=tmp_path, data_dir=tmp_path)
        await server.start()
        try:
            _, pending = await server.submit('default', {'block': True})
            assert await asyncio.to_thread(entered.wait, 5)
            reload = asyncio.create_task(server.reload_entry('default'))
            for _ in range(500):
                if len(generations) == 2 and server.entries['default']['workers'][0].executor.handler is generations[1]:
                    break
                await asyncio.sleep(.01)
            assert (await server.deploy('default', {}))['outputs']['generation'] == 1
            assert not reload.done() and not generations[0].closed
            release.set()
            assert (await pending)['outputs']['generation'] == 0
            await reload
            assert generations[0].closed
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())


def test_stop_during_start_is_rejected_without_losing_worker(tmp_path):
    entered, release = threading.Event(), threading.Event()
    class Handler:
        def init(self):
            entered.set()
            assert release.wait(10)
        def deploy(self, values):
            return values
    async def scenario():
        server = Server(handler_factory=Handler, resources=tmp_path, data_dir=tmp_path)
        starting = asyncio.create_task(server.start())
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            with pytest.raises(ServerError) as rejected:
                await server.stop_entry('default')
            assert rejected.value.code == 'entry_busy'
            release.set()
            await starting
            assert (await server.deploy('default', {'ok': True}))['outputs']['ok']
        finally:
            release.set()
            await starting
            await server.close()
    asyncio.run(scenario())


def test_declared_code_output_is_checked(tmp_path):
    config = {'outputs': {'count': {'type': 'integer'}}}
    with TestClient(Server(deploy=lambda values: {'count': 'wrong'}, config=config, resources=tmp_path, data_dir=tmp_path).app) as client:
        result = client.post('/v1/server/default/deploy', json={})
        assert result.status_code == 500
        task = client.get('/v1/tasks/' + result.json()['error']['task_id']).json()
        assert task['state'] == 'failed' and task['outputs'] is None


def test_editor_save_start_reload_stop_uses_existing_workflow(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import queue
    from nndeploy.server.template import WorkflowTemplateManager
    monkeypatch.setattr(WorkflowTemplateManager, 'init_templates', lambda: tmp_path)
    server = Server(SimpleNamespace(resources=str(tmp_path), debug=True), queue.Queue(), queue.Queue(), queue.Queue(),
                    admin_token='test-token')
    headers = {'Authorization': 'Bearer test-token'}
    with TestClient(server.app) as client:
        body = {'id': '', 'businessContent': text_graph()}
        assert client.post('/api/workflow/server/start', json=body).status_code == 401
        started = client.post('/api/workflow/server/start', json=body, headers=headers)
        assert started.status_code == 200, started.text
        info = started.json()
        identity = info['workflow_id']
        assert client.post(info['deploy_path'], json={'inputs': {'source': 'editor'}}).json()['outputs']['sink'] == 'editor'
        workflows = client.get('/api/workflows', headers=headers).json()['result']
        assert len(workflows) == 1 and workflows[0]['id'] == identity
        assert (server.workflow_dir / 'echo.server.json').is_file()
        body['id'] = identity
        updated = client.post('/api/workflow/server/start', json=body, headers=headers)
        assert updated.status_code == 200, updated.text
        assert updated.json()['workflow_id'] == identity
        assert client.post(f"/api/server/{info['name']}/stop", headers=headers).status_code == 200
        assert client.post(info['deploy_path'], json={'inputs': {'source': 'closed'}}).status_code == 503
        assert server.loop is not None


def test_pending_cancel_frees_queue_and_does_not_execute(tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = []
    def deploy(values):
        calls.append(values)
        entered.set()
        assert release.wait(10)
        return values
    async def scenario():
        server = Server(deploy=deploy, resources=tmp_path, data_dir=tmp_path)
        await server.start()
        try:
            _, first = await server.submit('default', {'first': True})
            assert await asyncio.to_thread(entered.wait, 5)
            identity, second = await server.submit('default', {'second': True})
            assert (await server.cancel_task(identity))['state'] == 'cancelled'
            assert (await server.cancel_task(identity))['state'] == 'cancelled'
            with pytest.raises(ServerError) as error:
                await second
            assert error.value.code == 'task_cancelled'
            assert server.inflight == 1 and calls == [{'first': True}]
            release.set()
            await first
        finally:
            release.set()
            await server.close()
    asyncio.run(scenario())


def test_idempotent_http_requests_reuse_result_and_reject_changed_input(tmp_path):
    calls = []
    def deploy(values):
        calls.append(values)
        return values
    with TestClient(Server(deploy=deploy, resources=tmp_path, data_dir=tmp_path).app) as client:
        headers = {'Idempotency-Key': 'one-call'}
        first = client.post('/v1/server/default/deploy', json={'inputs': {'value': 1}}, headers=headers)
        second = client.post('/v1/server/default/deploy', json={'inputs': {'value': 1}}, headers=headers)
        assert first.status_code == second.status_code == 200
        assert first.json() == second.json() and len(calls) == 1
        assert client.post('/v1/server/default/deploy', json={'inputs': {'value': 2}}, headers=headers).status_code == 409


def test_concurrent_idempotency_only_executes_once(tmp_path):
    calls = []
    def deploy(values):
        calls.append(values)
        return values
    async def scenario():
        server = Server(deploy=deploy, resources=tmp_path, data_dir=tmp_path)
        await server.start()
        try:
            submissions = await asyncio.gather(*(server.submit('default', {'value': 1}, idempotency_key='same') for _ in range(10)))
            assert len({identity for identity, future in submissions}) == 1
            await asyncio.gather(*(future for identity, future in submissions))
            assert len(calls) == 1 and server.inflight == 0
        finally:
            await server.close()
    asyncio.run(scenario())


def test_bad_entry_does_not_prevent_other_entry_startup(tmp_path):
    broken = text_graph()
    broken['node_repository_'][0]['key_'] = 'unknown.Node'
    server = Server(name='broken', workflow=broken, resources=tmp_path, data_dir=tmp_path)
    server.add(name='working', deploy=lambda values: values)
    with TestClient(server.app) as client:
        assert client.get('/health/ready').status_code == 200
        assert client.get('/v1/server/broken/schema').json()['status'] == 'failed'
        response = client.post('/v1/server/working/deploy', json={'inputs': {'ready': True}})
        assert response.status_code == 200 and response.json()['outputs']['ready']


def test_editor_without_token_uses_workflow_name_and_remembers_rename(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import queue
    from nndeploy.server.template import WorkflowTemplateManager
    monkeypatch.setattr(WorkflowTemplateManager, 'init_templates', lambda: tmp_path)
    server = Server(SimpleNamespace(resources=str(tmp_path), debug=True), queue.Queue(), queue.Queue(), queue.Queue())
    with TestClient(server.app) as client:
        access = client.get('/api/auth/session').json()
        assert access['required'] is False and access['management_available'] is True
        body = {'id': '', 'businessContent': text_graph()}
        first = client.post('/api/workflow/server/start', json=body)
        assert first.status_code == 200, first.text
        assert first.json()['name'] == 'echo'
        body['id'] = first.json()['workflow_id']
        body['name'] = 'MyEcho'
        renamed = client.post('/api/workflow/server/start', json=body)
        assert renamed.status_code == 200, renamed.text
        assert renamed.json()['name'] == 'MyEcho'
        assert client.get('/api/server/echo').json()['status'] == 'stopped'
        assert client.get('/api/workflow/' + body['id'] + '/server').json()['name'] == 'MyEcho'
        del body['name']
        assert client.post('/api/workflow/server/start', json=body).json()['name'] == 'MyEcho'
        assert client.post(renamed.json()['deploy_path'], json={'inputs': {'source': 'works'}}).json()['outputs']['sink'] == 'works'
        other = client.post('/api/workflow/server/start', json={'id': '', 'businessContent': text_graph(), 'name': 'MyEcho'})
        assert other.status_code == 409, other.text
        assert client.post('/api/server/MyEcho/stop').status_code == 200
        broken = text_graph()
        broken['name_'] = 'RetryEcho'
        broken['node_repository_'][1]['key_'] = 'missing.Node'
        failed = client.post('/api/workflow/server/start', json={'businessContent': broken, 'name': 'RetryName'})
        assert failed.status_code == 500, failed.text
        identity = next(item['id'] for item in client.get('/api/workflows').json()['result'] if item['name_'] == 'RetryEcho')
        repaired = text_graph()
        repaired['name_'] = 'RetryEcho'
        retry = client.post('/api/workflow/server/start', json={'id': identity, 'businessContent': repaired})
        assert retry.status_code == 200 and retry.json()['name'] == 'RetryName', retry.text


@pytest.mark.parametrize('form', ['path', 'relative', 'string', 'legacy'])
def test_deploy_accepts_local_image_without_upload(tmp_path, monkeypatch, form):
    path = tmp_path / 'input.jpg'
    Image.new('RGB', (5, 7), (20, 40, 60)).save(path)
    original = path.read_bytes()
    monkeypatch.chdir(tmp_path)
    reference = {'path': str(path)}
    if form == 'relative': reference = {'path': 'input.jpg'}
    if form == 'string': reference = str(path)
    if form == 'legacy': reference = {'file_id': str(path)}
    server = Server(workflow=image_graph(), name='image', resources=tmp_path / 'server', data_dir=tmp_path / 'server')
    with TestClient(server.app) as client:
        body = {'inputs': {'decode': reference}}
        response = client.post('/v1/server/image/deploy', json=body, headers={'Idempotency-Key': 'local-image'})
        assert response.status_code == 200, response.text
        repeat = client.post('/v1/server/image/deploy', json=body, headers={'Idempotency-Key': 'local-image'})
        assert repeat.status_code == 200 and repeat.json() == response.json()
        file = response.json()['outputs']['encode']['file_id']
        image = Image.open(io.BytesIO(client.get(f'/v1/files/{file}/content').content))
        assert image.size == (5, 7)
        for bad in [str(tmp_path / 'missing.jpg'), str(tmp_path)]:
            result = client.post('/v1/server/image/deploy', json={'inputs': {'decode': {'path': bad}}})
            assert result.status_code == 422 and result.json()['error']['code'] == 'invalid_file'
    assert path.read_bytes() == original


def test_numeric_local_file_is_decoded_without_upload(tmp_path):
    import numpy as np
    path = tmp_path / 'values.npy'
    np.save(path, np.array([1, 2, 3], dtype=np.float32))
    config = {'inputs': {'values': {'type': 'array', 'dtype': 'float32', 'shape': [3], 'media_type': 'application/x-npy'}},
              'outputs': {'sum': {'type': 'number'}}}
    server = Server(name='sum', deploy=lambda inputs: {'sum': float(inputs['values'].sum())}, config=config,
                    resources=tmp_path / 'server', data_dir=tmp_path / 'server')
    with TestClient(server.app) as client:
        result = client.post('/v1/server/sum/deploy', json={'inputs': {'values': {'path': str(path)}}})
        assert result.status_code == 200, result.text
        assert result.json()['outputs']['sum'] == 6


def test_node_parameters_override_input_output_and_restore_defaults(tmp_path):
    source = tmp_path / 'color.png'
    Image.new('RGB', (5, 7), (20, 40, 60)).save(source)
    output = tmp_path / 'custom.png'
    server = Server(workflow=image_graph(), name='image', resources=tmp_path / 'server', data_dir=tmp_path / 'server')
    with TestClient(server.app) as client:
        body = {'parameters': {'decode': {'path_': str(source), 'color_mode_': 'L'},
                               'encode': {'path_': str(output)}}}
        changed = client.post('/v1/server/image/deploy', json=body)
        assert changed.status_code == 200, changed.text
        assert Image.open(output).mode == 'L'
        normal = client.post('/v1/server/image/deploy', json={'inputs': {'decode': {'path': str(source)}}})
        assert normal.status_code == 200, normal.text
        reference = normal.json()['outputs']['encode']['file_id']
        image = Image.open(io.BytesIO(client.get('/v1/files/' + reference + '/content').content))
        assert image.mode == 'RGB'
        assert server.entries['image']['workers'][0].executor.runner.graph.get_node('decode').color_mode_ == 'RGB'
        bad = client.post('/v1/server/image/deploy', json={'inputs': {'decode': {'path': str(source)}},
                           'parameters': {'decode': {'key_': 'another.Node'}}})
        assert bad.status_code == 422 and 'Unknown node parameter' in bad.text


@pytest.mark.parametrize('factory', [False, True])
def test_nested_intermediate_parameters_reinitialize_code_graph_and_restore(tmp_path, factory):
    import nndeploy.base as base
    from nndeploy.dag import Edge, Graph, Node

    class Prefix(Node):
        def __init__(self, name, inputs, outputs):
            super().__init__(name, inputs, outputs)
            self.set_input_type(str)
            self.set_output_type(str)
            self.prefix = 'default:'
        def init(self):
            self.cached = self.prefix
            return base.Status.ok()
        def run(self):
            return self.get_output(0).set(self.cached + self.get_input(0).get(self))
        def serialize(self):
            data = json.loads(super().serialize())
            data['param_'] = {'prefix': self.prefix}
            return json.dumps(data)
        def deserialize(self, value):
            self.prefix = json.loads(value)['param_']['prefix']
            return super().deserialize(value)

    def make():
        incoming, outgoing = Edge('text'), Edge('result')
        graph = Graph('nested', [incoming], [outgoing])
        inner = Graph('inner', [incoming], [outgoing])
        inner.set_input_type(str)
        inner.set_output_type(str)
        inner.add_node(Prefix('prefix', [incoming], [outgoing]))
        graph.add_node(inner)
        return graph

    options = {'graph_factory': make} if factory else {'graph': make()}
    server = Server(name='nested', resources=tmp_path, data_dir=tmp_path, **options)
    with TestClient(server.app) as client:
        for overrides, expected in [({'inner_prefix': {'param_': {'prefix': 'changed:'}}}, 'changed:value'),
                                    ({}, 'default:value'),
                                    ({'inner_prefix': {'param_': {'prefix': 'again:'}}}, 'again:value')]:
            result = client.post('/v1/server/nested/deploy', json={'inputs': {'text': 'value'}, 'parameters': overrides})
            assert result.status_code == 200, result.text
            assert result.json()['outputs']['result'] == expected


def test_node_body_exposure_and_legacy_requests_cannot_bypass_it(tmp_path):
    from nndeploy.server.config import suggest_config
    source = tmp_path / 'source.png'
    Image.new('RGB', (5, 7), (20, 40, 60)).save(source)
    graph = image_graph()
    graph['node_repository_'][0]['path_'] = str(source)
    config = suggest_config(graph)
    config['parameters']['decode']['exposed'] = [['color_mode_']]
    config['parameters']['encode']['exposed'] = []
    server = Server(workflow=graph, config=config, name='image', resources=tmp_path / 'server', data_dir=tmp_path / 'server')
    with TestClient(server.app) as client:
        response = client.post('/v1/server/image/deploy', json={'decode': {'color_mode_': 'L'}})
        assert response.status_code == 200, response.text
        reference = response.json()['outputs']['encode']['file_id']
        assert Image.open(io.BytesIO(client.get('/v1/files/' + reference + '/content').content)).mode == 'L'
        for body in [{'decode': {'path_': str(source)}}, {'decode': {'path': str(source)}},
                     {'inputs': {'decode': {'path': str(source)}}},
                     {'parameters': {'encode': {'path_': str(tmp_path / 'output.png')}}}]:
            denied = client.post('/v1/server/image/deploy', json=body)
            assert denied.status_code == 422, denied.text
        normal = client.post('/v1/server/image/deploy', json={})
        assert normal.status_code == 200, normal.text
        reference = normal.json()['outputs']['encode']['file_id']
        assert Image.open(io.BytesIO(client.get('/v1/files/' + reference + '/content').content)).mode == 'RGB'
        batch = client.post('/v1/server/image/deploy/batch', json={'items': [{'decode': {'color_mode_': 'L'}}]})
        assert batch.status_code == 200 and batch.json()['items'][0]['accepted'], batch.text
        schema = server.describe('image')['schema']
        assert schema['inputs'] == {} and 'encode' not in schema['parameters']
        assert schema['parameters']['decode']['default'] == {'color_mode_': 'RGB'}


def test_editor_selects_and_persists_public_fields(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import queue
    from nndeploy.server.template import WorkflowTemplateManager
    monkeypatch.setattr(WorkflowTemplateManager, 'init_templates', lambda: tmp_path)
    server = Server(SimpleNamespace(resources=str(tmp_path), debug=True), queue.Queue(), queue.Queue(), queue.Queue())
    body = {'businessContent': text_graph()}
    with TestClient(server.app) as client:
        preview = client.post('/api/workflow/server/parameters', json=body)
        assert preview.status_code == 200, preview.text
        selected = {name: field['exposed'] for name, field in preview.json().items()}
        body['exposed_parameters'] = selected
        response = client.post('/api/workflow/server/start', json=body)
        assert response.status_code == 200, response.text
        body['id'] = response.json()['workflow_id']
        saved = client.post('/api/workflow/server/parameters', json=body)
        assert {name: field['exposed'] for name, field in saved.json().items()} == selected
        name = response.json()['name']
        result = client.post(f'/v1/server/{name}/deploy', json={'source': {'str_': 'flat'}})
        assert result.status_code == 200, result.text
