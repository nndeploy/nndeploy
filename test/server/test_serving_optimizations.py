"""Behavioral coverage for demand-driven results and safe parameter updates."""
import asyncio
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from nndeploy.server import Server
from nndeploy.server.config import ServerConfig
from nndeploy.server.entry import new_entry, replacement
from nndeploy.server.io import GraphIO
from nndeploy.server.metrics import Metrics
from nndeploy.server.stages import collection
from nndeploy.base import Status
from test_redis_queue import backend, redis_service
from test_metrics import scrape, value
from test_server_api import text_graph


class Parameters:
    def __init__(self):
        self.values = {'threshold': .5, 'model': 'a'}
        self.fail = False

    def serialize(self):
        return json.dumps(self.values)

    def deserialize(self, value):
        self.values = json.loads(value)
        return Status.ok()

    def get_runtime_parameter_paths(self):
        return [('threshold',)]

    def set_runtime_parameters(self, patch):
        self.values.update(patch)
        if self.fail and patch['threshold'] != .5:
            raise RuntimeError('partial setter failure')


class ParameterGraph:
    def __init__(self):
        self.node = Parameters()
        self.initializations = self.releases = 0
        self.edge = SimpleNamespace(get_name=lambda: 'output')

    def get_node(self, name):
        return self.node

    def get_all_output(self):
        return [self.edge]

    def get_edge(self, name):
        return self.edge

    def init(self):
        self.initializations += 1
        return Status.ok()

    def deinit(self):
        self.releases += 1
        return Status.ok()


def parameter_binding(**options):
    graph = ParameterGraph()
    config = ServerConfig(parameters={'node': dict(type='json', node=['node'], required=False,
                         default=copy.deepcopy(graph.node.values), **options)},
                         outputs={'output': {'type': 'string', 'edge': 'output'}})
    return graph, GraphIO(graph, config)


@pytest.mark.parametrize('failure', ['none', 'body', 'setter'])
def test_runtime_parameters_restore_without_reinitializing(failure):
    graph, binding = parameter_binding()
    graph.node.fail = failure == 'setter'
    def request():
        with binding.parameters(graph, {'parameters': {'node': {'threshold': .8}}}, {}):
            assert graph.node.values == {'threshold': .8, 'model': 'a'}
            if failure == 'body':
                raise RuntimeError('algorithm failure')
    if failure == 'none':
        request()
    else:
        with pytest.raises(RuntimeError):
            request()
    assert graph.node.values == {'threshold': .5, 'model': 'a'}
    assert graph.initializations == graph.releases == 0


@pytest.mark.parametrize('options,patch', [({}, {'model': 'b', 'threshold': .8}),
                                          ({'runtime': []}, {'threshold': .8})])
def test_initialization_and_explicit_opt_out_preserve_rebuild(options, patch):
    graph, binding = parameter_binding(**options)
    with binding.parameters(graph, {'parameters': {'node': patch}}, {}):
        assert graph.initializations == 1
        assert all(graph.node.values[k] == v for k, v in patch.items())
    assert graph.node.values == {'threshold': .5, 'model': 'a'}
    assert graph.initializations == graph.releases == 2


def test_config_cannot_promote_unsupported_field_to_runtime():
    with pytest.raises(ValueError, match='not supported'):
        parameter_binding(runtime=[['model']])
    with pytest.raises(ValueError, match='existing'):
        parameter_binding(runtime=[['missing']])


def test_parameter_stages_report_rebuilds_separately():
    graph, binding = parameter_binding()
    metrics = Metrics()
    executor = SimpleNamespace(storage=SimpleNamespace(metrics=metrics), config=ServerConfig(name='parameters'))
    with collection(executor):
        with binding.parameters(graph, {'parameters': {'node': {'threshold': .9}}}, {}):
            pass
    with collection(executor):
        with binding.parameters(graph, {'parameters': {'node': {'model': 'b'}}}, {}):
            pass
    samples = [s for metric in metrics.registry.collect() for s in metric.samples]
    assert value(samples, 'nndeploy_execution_stage_seconds_count', entry='parameters', stage='parameter_update', outcome='success') == 1
    assert value(samples, 'nndeploy_execution_stage_seconds_count', entry='parameters', stage='parameter_reinitialize', outcome='success') == 1


def test_replacement_does_not_inherit_runtime_ownership():
    original = new_entry(ServerConfig(name='model'), {'workflow': {}, 'untyped': False}, untyped=False)
    original.update(queue=object(), control_guard=object(), device_guard=object(), file_holder='old',
                    maintenance=object(), consumer_failed=True, local_stop=True, release_pending=True,
                    saved_token='persisted', shared_key='shared', workflow_id='workflow')
    original['workers'].append(object())
    original['futures']['task'] = object()
    original['admitted'].add('task')
    candidate = replacement(original)
    for key in ('queue', 'control_guard', 'device_guard', 'file_holder', 'maintenance', 'consumer_failed', 'local_stop', 'release_pending'):
        assert key not in candidate
    assert candidate['workers'] == [] and candidate['futures'] == {} and candidate['admitted'] == set()
    assert candidate['source'] is not original['source']
    assert candidate['saved_token'] == 'persisted' and candidate['shared_key'] == 'shared'
    assert candidate['workflow_id'] == 'workflow'
    assert original['workers'] and original['futures']


def process_echo(inputs):
    time.sleep(.01)
    if inputs.get('fail'):
        raise ValueError('test error')
    return inputs


@pytest.mark.parametrize('isolation', ['thread', 'process'])
def test_stages_survive_worker_boundary_and_include_failures(tmp_path, isolation):
    server = Server(name='stages', deploy=process_echo, resources=tmp_path, data_dir=tmp_path,
                    config={'execution': {'isolation': isolation}, 'inputs': {
                        'text': {'type': 'string', 'required': False, 'default': ''},
                        'fail': {'type': 'boolean', 'required': False, 'default': False}}}, admin_token='metrics-admin-secret')
    with TestClient(server.app) as client:
        assert client.post('/v1/server/stages/deploy', json={'inputs': {'text': 'hello'}}).status_code == 200
        assert client.post('/v1/server/stages/deploy', json={'inputs': {'fail': True}}).status_code == 500
        samples = scrape(client)
        for stage in ('input_decode', 'compute', 'output_encode', 'output_validate'):
            assert value(samples, 'nndeploy_execution_stage_seconds_count', entry='stages', stage=stage, outcome='success') >= 1
        assert value(samples, 'nndeploy_execution_stage_seconds_count', entry='stages', stage='compute', outcome='error') == 1


@pytest.mark.parametrize('isolation', ['thread', 'process'])
def test_real_graph_runtime_parameter_path_and_metrics(tmp_path, isolation):
    from nndeploy.server.config import suggest_config
    settings = suggest_config(text_graph())
    settings['execution'] = {'isolation': isolation}
    server = Server(workflow=text_graph(), config=settings, name='text', resources=tmp_path, data_dir=tmp_path,
                    admin_token='metrics-admin-secret')
    with TestClient(server.app) as client:
        for text in ('changed', 'restored'):
            result = client.post('/v1/server/text/deploy', json={'source': {'str_': text}})
            assert result.status_code == 200, result.text
            assert text in str(result.json()['outputs'])
        assert client.get('/v1/server/text/schema').json()['parameter_modes']['source']['runtime'] == [['str_']]
        samples = scrape(client)
        assert value(samples, 'nndeploy_execution_stage_seconds_count', entry='text', stage='parameter_update', outcome='success') == 2
        assert not any(s.name == 'nndeploy_execution_stage_seconds_count' and s.labels['stage'] == 'parameter_reinitialize' for s in samples)


def delayed_file(inputs):
    while not Path(inputs['release']).exists():
        time.sleep(.01)
    return {'ok': True}


async def until(predicate):
    for _ in range(300):
        if predicate():
            return
        await asyncio.sleep(.01)
    raise AssertionError('condition did not become true')


def test_async_shared_admission_has_no_waiter_and_readers_detach(backend, tmp_path):
    db, redis_config = backend
    server = Server(name='slow', deploy=delayed_file, resources=tmp_path / 'server', data_dir=tmp_path / 'server',
                    database_url=db.url.render_as_string(hide_password=False), redis_config=redis_config)
    release = tmp_path / 'release'
    async def scenario():
        await server.start()
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
                response = await client.post('/v1/server/slow/deploy', json={'inputs': {'release': str(release)}, 'mode': 'async'}, headers={'Idempotency-Key': 'same'})
                assert response.status_code == 202, response.text
                identity = response.json()['task_id']
                assert not server.waiters and not server.task_futures and server.inflight == 0
                first, a = await server.submit('slow', {'release': str(release)}, idempotency_key='same')
                second, b = await server.submit('slow', {'release': str(release)}, idempotency_key='same')
                assert first == second == identity and len(server.waiters) == 1
                a.cancel()
                await asyncio.sleep(.02)
                assert len(server.waiters) == 1
                b.cancel()
                await until(lambda: not server.waiters)
                assert not server.task_futures and server.inflight == 0
                assert not db.get_task(identity)['cancel_requested']
                _, reader = await server.submit('slow', {'release': str(release)}, idempotency_key='same')
                release.touch()
                result = await asyncio.wait_for(reader, 6)
                assert result['outputs'] == {'ok': True}
                await until(lambda: not server.waiters)
        finally:
            release.touch()
            await server.close()
    asyncio.run(scenario())


def test_shared_notifications_wake_without_frequent_database_polling(backend, tmp_path, monkeypatch):
    db, config = backend
    server = Server(name='slow', deploy=delayed_file, resources=tmp_path, data_dir=tmp_path,
                    database_url=db.url.render_as_string(hide_password=False), redis_config=config)
    release = tmp_path / 'release'
    async def scenario():
        await server.start()
        reads = []
        try:
            identity, receipt = await server.submit('slow', {'release': str(release)}, observe=False)
            # Read with a second Server DB connection to isolate reader queries
            # from cancellation probes and execution ownership maintenance.
            observer = Server(resources=tmp_path / 'observer', data_dir=tmp_path / 'observer',
                              database_url=db.url.render_as_string(hide_password=False), redis_config=config)
            original_observer = observer.db.get_task
            def observed(identity):
                reads.append(time.monotonic())
                return original_observer(identity)
            monkeypatch.setattr(observer.db, 'get_task', observed)
            try:
                _, future = observer._record_future(dict(task_id=identity, state='pending', queue_key='shared'))
                await asyncio.sleep(1.8)
                assert len(reads) <= 3
                release.touch()
                assert (await asyncio.wait_for(future, 2))['outputs'] == {'ok': True}
            finally:
                await observer.close()
        finally:
            release.touch()
            await server.close()
    asyncio.run(scenario())


def test_missing_result_signal_falls_back_to_database(backend, tmp_path, monkeypatch):
    db, config = backend
    server = Server(name='slow', deploy=delayed_file, resources=tmp_path, data_dir=tmp_path,
                    database_url=db.url.render_as_string(hide_password=False), redis_config=config)
    release = tmp_path / 'release'
    async def scenario():
        await server.start()
        try:
            monkeypatch.setattr(server.event_notifications, 'notify', lambda *args, **kwargs: None)
            identity, reader = await server.submit('slow', {'release': str(release)})
            await asyncio.sleep(.2)
            release.touch()
            assert (await asyncio.wait_for(reader, 6))['outputs'] == {'ok': True}
            assert db.get_task(identity)['state'] == 'succeeded'
        finally:
            release.touch()
            await server.close()
    asyncio.run(scenario())


def test_disconnect_during_shared_admission_does_not_leave_reader(backend, tmp_path, monkeypatch):
    import threading
    db, config = backend
    server = Server(name='slow', deploy=delayed_file, resources=tmp_path, data_dir=tmp_path,
                    database_url=db.url.render_as_string(hide_password=False), redis_config=config)
    release = tmp_path / 'release'
    entered, commit = threading.Event(), threading.Event()
    original = server.db.create_shared_task
    def delayed(*args, **kwargs):
        entered.set()
        assert commit.wait(5)
        return original(*args, **kwargs)
    async def scenario():
        await server.start()
        monkeypatch.setattr(server.db, 'create_shared_task', delayed)
        try:
            submitting = asyncio.create_task(server.submit('slow', {'release': str(release)}, idempotency_key='disconnect'))
            await until(entered.is_set)
            submitting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await submitting
            commit.set()
            release.touch()
            await until(lambda: server.inflight == 0)
            assert not server.waiters and not server.task_futures
            identity, receipt = await server.submit('slow', {'release': str(release)}, idempotency_key='disconnect', observe=False)
            await until(lambda: db.get_task(identity)['state'] == 'succeeded')
        finally:
            commit.set()
            release.touch()
            await server.close()
    asyncio.run(scenario())


def test_parameter_modes_only_report_exposed_paths():
    graph, binding = parameter_binding(exposed=[['threshold']])
    assert binding.parameter_modes() == {'node': {'runtime': [['threshold']], 'initialization': []}}
    graph, binding = parameter_binding(exposed=[])
    assert binding.parameter_modes() == {}


def test_initial_deinit_failure_is_not_retried():
    graph, binding = parameter_binding(runtime=[])
    calls = []
    def fail():
        calls.append(True)
        raise RuntimeError('release not confirmed')
    graph.deinit = fail
    with pytest.raises(RuntimeError, match='release not confirmed'):
        with binding.parameters(graph, {'parameters': {'node': {'threshold': .8}}}, {}):
            pytest.fail('must not execute')
    assert len(calls) == 1 and graph.initializations == 0


def test_closing_before_shared_reader_starts_settles_observer():
    from nndeploy.server.results import SharedResults
    from nndeploy.server.errors import ServerError
    async def scenario():
        server = SimpleNamespace(task_futures={}, waiters={}, redis_config={}, shutting_down=True)
        results = SharedResults(server)
        _, future = results.record(dict(task_id='task_shutdown', state='pending', queue_key='shared'))
        await results.close()
        with pytest.raises(ServerError) as failure:
            await future
        assert failure.value.code == 'server_stopping'
        assert not server.task_futures and not server.waiters
    asyncio.run(scenario())
