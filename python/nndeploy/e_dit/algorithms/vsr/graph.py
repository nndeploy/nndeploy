"""Native DAG node for the VSR streaming operation (no sampling loop)."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from nndeploy.base import Status, StatusCode
from nndeploy.dag import Graph, Node, Edge, NodeCreator, IOType
from nndeploy.e_dit.params import DiTParams
from .resources import VSRResources


class VSRNode(Node):
    def __init__(self, name, inputs, outputs, params=None):
        super().__init__(name, inputs, outputs)
        self.set_key('nndeploy.e_dit.WanVSR')
        self.set_input_type(str)
        self.set_input_type(dict)
        self.set_output_type(bytes)
        self.set_io_type(IOType.Model)
        self.add_io_param('model_path')
        self.params = params or DiTParams(model='wan_vsr')
        self.model_path = self.params.model_path
        self.resources = None

    def get_param(self):
        return self.params

    def set_param(self, param):
        if not isinstance(param, DiTParams) or param.model != 'wan_vsr':
            raise TypeError('VSR requires DiTParams(model="wan_vsr")')
        if self.resources is not None:
            raise RuntimeError('Deinitialize before changing VSR parameters')
        param.build_config()
        self.params, self.model_path = param, param.model_path
        return Status.ok()

    def init(self):
        if self.resources is None:
            params = self.params.to_dict()
            params['model_path'] = self.model_path
            resources = VSRResources(DiTParams(**params).build_config())
            try:
                resources.open()
            except BaseException:
                resources.close()
                raise
            self.resources = resources
        return Status.ok()

    def deinit(self):
        if self.resources is not None:
            self.resources.close()
            self.resources = None
        return Status.ok()

    def infer_file(self, video, output, **sampling):
        if self.resources is None:
            raise RuntimeError('Initialize VSR before inference')
        return self.resources.restore(video, output, sampling)

    def run(self):
        video = self.get_input(0).get(self)
        sampling = self.get_input(1).get(self)
        if sampling is None:
            sampling = {}
        if not isinstance(sampling, dict):
            raise ValueError('sampling must be an object')
        with TemporaryDirectory(prefix='nndeploy-vsr-') as directory:
            path = self.infer_file(video, Path(directory) / 'result.mp4', **sampling)
            self.get_output(0).set(Path(path).read_bytes())
        return Status.ok()

    def serialize(self):
        value = json.loads(super().serialize())
        value.update(params=self.params.to_dict(), model_path=self.model_path)
        return json.dumps(value)

    def deserialize(self, value):
        data = json.loads(value)
        self.set_param(DiTParams(**data['params']))
        self.model_path = data.get('model_path', self.params.model_path)
        return super().deserialize(value)


class _Creator(NodeCreator):
    def create_node(self, name, inputs, outputs):
        self.node = VSRNode(name, inputs, outputs)
        return self.node


_creator = _Creator()


class VSRGraph(Graph):
    def __init__(self, params, name='wan_vsr'):
        params.build_config()
        inputs, outputs = [Edge('video'), Edge('sampling')], [Edge('result')]
        super().__init__(name, inputs, outputs)
        self.set_input_type(str)
        self.set_input_type(dict)
        self.set_output_type(bytes)
        self.set_loop_count(1)
        self.model = VSRNode('wan_vsr', inputs, outputs, params)

    @property
    def resources(self):
        return self.model.resources

    def infer(self, video, **sampling):
        self.get_input(0).set(str(video))
        self.get_input(1).set(sampling)
        status = self.run()
        if status != StatusCode.Ok:
            raise RuntimeError(f'VSR inference failed: {status}')
        return self.get_output(0).get_graph_output()

    def infer_file(self, video, output, **sampling):
        """Stream directly to MP4 without materializing the result as bytes."""
        return self.model.infer_file(video, output, **sampling)


def server_config(*, isolation='process'):
    return {
        'inputs': {
            'video': {'type': 'file', 'media_type': 'video/mp4', 'edge': 'video'},
            'sampling': {'type': 'json', 'edge': 'sampling', 'required': False, 'default': {}},
        },
        'outputs': {'video': {'type': 'file', 'media_type': 'video/mp4', 'edge': 'result'}},
        'streaming': {'enabled': True, 'max_duration_seconds': 3600},
        'execution': {'isolation': isolation, 'startup_timeout_seconds': 1800,
                      'shutdown_timeout_seconds': 180,
                      'execution_timeout_seconds': 3600 if isolation == 'process' else None,
                      'allow_child_processes': isolation == 'process'},
    }
