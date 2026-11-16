"""CPU compute settings belong to an isolated Worker, never the API process."""
import asyncio
import copy
import functools
import os

import cv2
import pytest
from pydantic import ValidationError

from nndeploy.server import Server
from nndeploy.server.config import ExecutionConfig, suggest_config
from nndeploy.server.errors import ServerError


def cpu_report(values):
    if values.get('crash'):
        os._exit(17)
    import numpy as np
    from threadpoolctl import threadpool_info
    return {'pid': os.getpid(), 'opencv': cv2.getNumThreads(),
            'environment': os.environ.get('OMP_NUM_THREADS'),
            'pools': [pool['num_threads'] for pool in threadpool_info()],
            'result': float((np.ones((8, 8)) @ np.ones((8, 8)))[0, 0])}


class CpuHandler:
    def __init__(self):
        self.created = cv2.getNumThreads()

    def init(self):
        self.initialized = cv2.getNumThreads()

    def deploy(self, values):
        return dict(cpu_report(values), created=self.created, initialized=self.initialized)


class RefusingCpuHandler(CpuHandler):
    def __init__(self):
        super().__init__()
        if os.environ.get('OMP_NUM_THREADS') == '3':
            # Model a native library which ignores both requested and serial
            # limits; failed runtime configuration must not publish Ready.
            cv2.setNumThreads = lambda count: None
            cv2.getNumThreads = lambda: 99


def cpu_graph(initial=8):
    import nndeploy.base
    from nndeploy.dag import Edge, Graph, Node
    from nndeploy.inference import InferenceParam
    class Report(Node):
        def __init__(self, output):
            super().__init__('report', [], [output])
            self.set_output_type(str)
            self.parameter = InferenceParam(nndeploy.base.InferenceType.Default)
            self.parameter.set_num_thread(initial)

        def get_param(self):
            return self.parameter

        def init(self):
            self.initial = self.parameter.get_num_thread()
            return nndeploy.base.Status.ok()

        def run(self):
            self.get_output(0).set(f'{self.initial}:{self.parameter.get_num_thread()}')
            return nndeploy.base.Status.ok()
    target = Edge('report')
    graph = Graph('cpu', [], [target])
    graph.add_node(Report(target))
    return graph


def nested_cpu_graph():
    from nndeploy.dag import Graph
    inner = cpu_graph()
    outer = Graph('nested', [], inner.get_all_output())
    outer.add_node(inner)
    return outer


def test_cpu_setting_requires_isolation_and_preserves_default_fingerprints():
    assert 'cpu_threads' not in ExecutionConfig().model_dump()
    for value in (0, -1, True, 1.5, 65537):
        with pytest.raises(ValidationError):
            ExecutionConfig(isolation='process', cpu_threads=value)
    with pytest.raises(ValidationError, match='process'):
        ExecutionConfig(cpu_threads=1)
    assert ExecutionConfig(isolation='process', cpu_threads=2).cpu_threads == 2


def test_cpu_limits_apply_before_factory_and_survive_worker_restart_without_changing_parent(tmp_path):
    parent_threads, parent_environment = cv2.getNumThreads(), os.environ.get('OMP_NUM_THREADS')
    config = {'execution': {'isolation': 'process', 'cpu_threads': 1, 'warmup': [{'inputs': {}}]},
              'inputs': {'crash': {'type': 'boolean', 'required': False, 'default': False}}}
    server = Server(name='one', handler_factory=CpuHandler, config=config, data_dir=tmp_path)
    server.add(name='two', deploy=cpu_report, config={'execution': {'isolation': 'process', 'cpu_threads': 2}})
    async def scenario():
        try:
            await server.start()
            one, two = await asyncio.gather(server.deploy('one', {}), server.deploy('two', {}))
            one, two = one['outputs'], two['outputs']
            assert one['opencv'] == one['created'] == one['initialized'] == 1, one
            assert 1 <= two['opencv'] <= 2 and one['pid'] != two['pid']
            for expected, report in ((1, one), (2, two)):
                assert report['environment'] == str(expected) and report['result'] == 8
                assert all(threads <= expected for threads in report['pools'])
            assert cv2.getNumThreads() == parent_threads and os.environ.get('OMP_NUM_THREADS') == parent_environment
            with pytest.raises(ServerError) as failure:
                await server.deploy('one', {'crash': True})
            assert failure.value.code == 'worker_crashed'
            restarted = (await server.deploy('one', {}))['outputs']
            assert restarted['pid'] != one['pid'] and restarted['opencv'] == restarted['created'] == 1
            assert (await server.deploy('two', {}))['outputs']['environment'] == '2'
        finally:
            await server.close()
        assert cv2.getNumThreads() == parent_threads and os.environ.get('OMP_NUM_THREADS') == parent_environment
    asyncio.run(scenario())


@pytest.mark.parametrize('count', [1, 2])
@pytest.mark.parametrize('factory', [cpu_graph, nested_cpu_graph, functools.partial(cpu_graph, 1)], ids=['graph', 'nested', 'smaller'])
def test_native_graph_parameters_are_configured_before_init_and_reused(tmp_path, count, factory):
    config = {'outputs': {'report': {'type': 'string', 'edge': 'report'}},
              'execution': {'isolation': 'process', 'cpu_threads': count, 'warmup': [{'inputs': {}}]}}
    async def scenario():
        server = Server(graph_factory=factory, config=config, data_dir=tmp_path)
        try:
            await server.start()
            pid = server.entries['default']['workers'][0].pid
            expected = 1 if isinstance(factory, functools.partial) else count
            for _ in range(2):
                assert (await server.deploy('default', {}))['outputs']['report'] == f'{expected}:{expected}'
                assert server.entries['default']['workers'][0].pid == pid
        finally:
            await server.close()
    asyncio.run(scenario())


def test_workflow_json_cpu_settings_do_not_modify_workflow_and_survive_restart(tmp_path):
    from test_server_api import text_graph
    workflow = text_graph()
    original = copy.deepcopy(workflow)
    async def scenario():
        config = suggest_config(workflow)
        config['execution'] = {'isolation': 'process', 'cpu_threads': 1}
        first = Server(name='text', workflow=workflow, config=config, data_dir=tmp_path)
        try:
            await first.start()
            assert (await first.deploy('text', {'source': 'first'}))['outputs']['sink'] == 'first'
            assert workflow == original
        finally:
            await first.close()
        restored = Server(data_dir=tmp_path)
        try:
            await restored.start()
            assert restored.describe('text')['cpu_threads'] == 1
            assert (await restored.deploy('text', {'source': 'second'}))['outputs']['sink'] == 'second'
        finally:
            await restored.close()
    asyncio.run(scenario())


def test_cpu_reload_changes_only_candidate_and_failed_configuration_preserves_old_worker(tmp_path):
    config = {'execution': {'isolation': 'process', 'cpu_threads': 1}}
    server = Server(handler_factory=RefusingCpuHandler, config=config, data_dir=tmp_path)
    async def scenario():
        try:
            await server.start()
            first = (await server.deploy('default', {}))['outputs']
            updated = copy.deepcopy(config)
            updated['execution']['cpu_threads'] = 2
            await server.reload_entry('default', config=updated)
            second = (await server.deploy('default', {}))['outputs']
            assert second['environment'] == '2' and second['pid'] != first['pid']
            with pytest.raises(ProcessLookupError):
                os.kill(first['pid'], 0)
            updated['execution']['cpu_threads'] = 3
            with pytest.raises(ServerError):
                await server.reload_entry('default', config=updated)
            assert server.describe('default')['cpu_threads'] == 2
            still_ready = (await server.deploy('default', {}))['outputs']
            assert still_ready['pid'] == second['pid'] and still_ready['environment'] == '2'
        finally:
            await server.close()
    asyncio.run(scenario())
