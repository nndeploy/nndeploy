"""Editor CPU settings reuse the isolated Server controls and native Graphs."""
import copy
import json
import os
from types import SimpleNamespace

import cv2
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from nndeploy.server import Server
from nndeploy.server.app import _EditorRuntime
from test_cpu_threads import cpu_report
from test_device_execution import limits
from test_editor_devices import saved
from test_editor_history import location
from test_editor_queue_recovery import wait, live
from test_server_api import image_graph


def workflow(tmp_path, kind='flat'):
    resources = tmp_path / 'resources'
    plugins = resources / 'plugin'
    plugins.mkdir(parents=True)
    (plugins / 'cpu_output.py').write_text('''
import json
import os
from pathlib import Path
import cv2
import numpy as np
from threadpoolctl import threadpool_info
import nndeploy.base
import nndeploy.dag
from nndeploy.codec.pil import PILImageEncodec
from nndeploy.inference import InferenceParam
module_threads = os.environ.get('OMP_NUM_THREADS')
class CpuOutput(PILImageEncodec):
    def __init__(self, name, inputs, outputs):
        super().__init__(name, inputs, outputs)
        self.created_threads = os.environ.get('OMP_NUM_THREADS')
        self.parameter = InferenceParam(nndeploy.base.InferenceType.Default)
    def deserialize(self, value):
        data = json.loads(value)
        self.parameter.set_num_thread(data['initial_threads_'])
        self.marker = data['marker_']
        if data.get('refuse_'):
            cv2.setNumThreads = lambda count: None
            cv2.getNumThreads = lambda: 99
        return super().deserialize(value)
    def get_param(self):
        return self.parameter
    def init(self):
        Path(self.marker).touch()
        self.initial_threads = self.parameter.get_num_thread()
        self.initial_cv = cv2.getNumThreads()
        # A native library may initialize its process-global pool in init().
        cv2.setNumThreads(8)
        return nndeploy.base.Status.ok()
    def run(self):
        value = float((np.ones((8, 8)) @ np.ones((8, 8)))[0, 0])
        Path(self.path_).write_text(json.dumps(dict(pid=os.getpid(), module_threads=module_threads,
            created_threads=self.created_threads, initial_threads=self.initial_threads,
            native_threads=self.parameter.get_num_thread(), initial_cv=self.initial_cv,
            opencv=cv2.getNumThreads(), environment=os.environ.get('OMP_NUM_THREADS'), result=value,
            pools=[pool['num_threads'] for pool in threadpool_info()])))
        return nndeploy.base.Status.ok()
class Creator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        self.nodes = []
    def create_node(self, name, inputs, outputs):
        node = CpuOutput(name, inputs, outputs)
        node.set_key('test.editor.CpuOutput')
        self.nodes.append(node)
        return node
creator = Creator()
nndeploy.dag.register_node('test.editor.CpuOutput', creator)
''')
    source = resources / 'input.png'
    Image.new('RGB', (4, 5), (10, 20, 30)).save(source)
    graph = image_graph()
    graph['node_repository_'][0]['path_'] = str(source)
    node = graph['node_repository_'][1]
    node.update(key_='test.editor.CpuOutput', path_=str(resources / 'report.json'), node_type_='Output',
                io_type_='File', io_params_=['path_'], initial_threads_=1 if kind == 'smaller' else 8,
                marker_=str(resources / 'initialized'))
    if kind == 'nested':
        # Connect the outer decoder to the subgraph's public input. A whole
        # file-to-file graph with no outer edges is an unused isolated node.
        decoder = graph['node_repository_'].pop(0)
        graph['inputs_'] = copy.deepcopy(decoder['outputs_'])
        graph = dict(key_='nndeploy.dag.Graph', name_='outer', parallel_type_='kParallelTypeSequential',
                     node_repository_=[decoder, graph])
    return SimpleNamespace(resources=str(resources), log=str(tmp_path / 'editor.log'), debug=True), graph, node


def completed(client, identity):
    def read():
        result = client.get('/api/queue/task/' + identity).json()['result']
        return result if result['state'] in ('SUCCEEDED', 'FAILED', 'CANCELLED') else None
    return wait(read)


@pytest.mark.parametrize('kind', ['flat', 'nested', 'smaller'])
@pytest.mark.parametrize('budgeted', [False, True])
def test_editor_cpu_limits_apply_at_each_stage_and_do_not_change_other_processes(tmp_path, location, kind, budgeted):
    parent = (cv2.getNumThreads(), os.environ.get('OMP_NUM_THREADS'))
    args, graph, node = workflow(tmp_path, kind)
    original = copy.deepcopy(graph)
    runtime = _EditorRuntime(args)
    server = runtime.start(lambda jobs, plugins, cancellations: Server(args, jobs, plugins, cancellations,
                            database_url=location, device_limits=limits() if budgeted else None))
    server.add(name='other', deploy=cpu_report, config={'execution': {'isolation': 'process', 'cpu_threads': 3,
                    'devices': ['cpu'] if budgeted else []}})
    with TestClient(server.app) as client:
        try:
            other = client.post('/v1/server/other/deploy', json={}).json()['outputs']
            identity, path = saved(client, server, graph)
            previous = None
            for count in (1, 2, None):
                execution = {'isolation': 'process', 'devices': ['cpu'] if budgeted else []}
                if count is not None:
                    execution['cpu_threads'] = count
                path.write_text(json.dumps({'execution': execution}))
                response = client.post('/api/queue', params={'workflow_id': identity}, json=graph)
                assert response.status_code == 202, response.text
                result = completed(client, response.json()['result']['task_id'])
                assert result['state'] == 'SUCCEEDED', result
                assert len(result['output_files']) == 1
                report = client.get(result['output_files'][0]['url']).json()
                expected = 1 if kind == 'smaller' else count or 8
                assert report['native_threads'] == report['initial_threads'] == expected
                environment = str(count) if count is not None else parent[1]
                assert report['module_threads'] == report['created_threads'] == report['environment'] == environment
                if count is not None:
                    assert 1 <= report['initial_cv'] <= count and 1 <= report['opencv'] <= count
                    assert all(threads <= count for threads in report['pools'])
                assert report['result'] == 8
                assert report['pid'] != previous and report['pid'] != other['pid']
                if count is not None or budgeted:
                    wait(lambda: not live(report['pid']))
                previous = report['pid']
                unchanged = client.post('/v1/server/other/deploy', json={}).json()['outputs']
                assert unchanged['pid'] == other['pid'] and unchanged['environment'] == '3'
                assert (cv2.getNumThreads(), os.environ.get('OMP_NUM_THREADS')) == parent
                assert graph == original
        finally:
            runtime.close()


@pytest.mark.parametrize('budgeted', [False, True])
def test_unapplied_cpu_setting_fails_before_graph_init_and_replacement_recovers(tmp_path, location, budgeted):
    parent = (cv2.getNumThreads(), os.environ.get('OMP_NUM_THREADS'))
    args, graph, node = workflow(tmp_path)
    runtime = _EditorRuntime(args)
    server = runtime.start(lambda jobs, plugins, cancellations: Server(args, jobs, plugins, cancellations,
                            database_url=location, device_limits=limits() if budgeted else None))
    with TestClient(server.app) as client:
        try:
            identity, path = saved(client, server, graph)
            path.write_text(json.dumps({'execution': {'isolation': 'process', 'cpu_threads': 1,
                                                     'devices': ['cpu'] if budgeted else []}}))
            node['refuse_'] = True
            response = client.post('/api/queue', params={'workflow_id': identity}, json=graph)
            assert response.status_code == 202, response.text
            result = completed(client, response.json()['result']['task_id'])
            assert result['state'] == 'FAILED' and result['output_files'] == []
            assert 'CPU thread limit' in str(result['status'])
            from pathlib import Path
            assert not Path(node['marker_']).exists()
            node.pop('refuse_')
            response = client.post('/api/queue', params={'workflow_id': identity}, json=graph)
            result = completed(client, response.json()['result']['task_id'])
            assert result['state'] == 'SUCCEEDED', result
            report = client.get(result['output_files'][0]['url']).json()
            assert report['opencv'] == report['initial_threads'] == 1
            assert (cv2.getNumThreads(), os.environ.get('OMP_NUM_THREADS')) == parent
        finally:
            runtime.close()
