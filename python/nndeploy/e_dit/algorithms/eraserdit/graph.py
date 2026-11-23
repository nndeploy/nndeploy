"""Configurable complete EraserDiT graph for Python, JSON workflows and Server."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile

from nndeploy.base import Status
from nndeploy.dag import Edge, Graph, Node, NodeCreator, register_node, IOType
from nndeploy.e_dit.params import DiTParams


class EraserDiTNode(Graph):
    def __init__(self, name, inputs, outputs, params=None):
        super().__init__(name, inputs, outputs)
        self.set_key('nndeploy.e_dit.EraserDiT')
        self.set_desc('EraserDiT video inpainting; resident models and request-local sampling state')
        self.set_input_type(str)
        self.set_input_type(str)
        self.set_input_type(dict)
        self.set_output_type(bytes)
        self.params = params or DiTParams()
        self.model_path = self.params.model_path
        self.set_io_type(IOType.Model)
        self.add_io_param('model_path')
        self.resources = None
        self.last_metrics = None
        self.last_warmup = None
        from nndeploy.e_dit.algorithms.eraserdit.nodes.request_lifecycle import InferenceGraph
        from nndeploy.e_dit.nodes.graph import loading_graph
        self._inference = None
        if not loading_graph.get():
            self._inference = InferenceGraph()
            self.add_node(self._inference)

    def get_param(self):
        return self.params

    def set_param(self, param):
        # Python Params serialize explicitly below: the native Param JSON
        # overload currently exposes an unbound rapidjson argument to Python.
        if not isinstance(param, DiTParams):
            raise TypeError('EraserDiT requires DiTParams')
        if self.resources is not None:
            raise RuntimeError('Deinitialize before replacing instance parameters')
        self.params = param
        self.model_path = param.model_path
        return Status.ok()

    def init(self):
        from nndeploy.e_dit.resources import DiTResources
        params = self.params.to_dict()
        params['model_path'] = self.model_path
        resources = DiTResources(DiTParams(**params))
        try:
            resources.open()
            self._inference.bind(resources)
            status = super().init()
            from nndeploy.base import StatusCode
            if status != StatusCode.Ok:
                raise RuntimeError(f"Model graph initialization failed: {status}")
        except BaseException:
            from nndeploy.e_dit.nodes.graph import release_bindings
            try:
                release_bindings(self._inference)
            finally:
                resources.close()
            raise
        self.resources = resources
        return Status.ok()

    def deinit(self):
        if self.resources is not None:
            try:
                super().deinit()
            finally:
                self.resources.close()
                self.resources = None
        return Status.ok()

    def run(self):
        from nndeploy.e_dit.algorithms.eraserdit.config import EraserDiTEraseSamplingParams
        if self.resources is None:
            raise RuntimeError('Initialize EraserDiT before execution')
        video = self.get_input(0).get(self)
        mask = self.get_input(1).get(self)
        self.last_metrics = None
        self.last_warmup = None
        options = deepcopy(self.get_input(2).get(self) or {})
        if not isinstance(options, dict):
            raise ValueError('sampling must be a JSON object')
        for name in ('video_input_path', 'mask_input_path', 'output_path', 'output_file_name',
                     'runtime_workdir', 'runtime_state', 'save_output'):
            if name in options:
                raise ValueError(f'{name} is managed by the Graph, not a sampling override')
        for path in (video, mask):
            if not isinstance(path, str) or not Path(path).is_file():
                raise ValueError(f'Input is not a local file: {path}')
        with tempfile.TemporaryDirectory(prefix='nndeploy-dit-') as directory:
            from nndeploy.e_dit.video.static_mask import expand_static_mask
            prepared_mask = expand_static_mask(Path(video), Path(mask), Path(directory))
            sampling = EraserDiTEraseSamplingParams(
                video_input_path=video, mask_input_path=str(prepared_mask),
                output_path=directory, output_file_name='result.mp4',
                **options)
            from nndeploy.dag.execution import current_execution
            from nndeploy.e_dit.execution import CancellationProbe, ProgressSink
            control = current_execution()
            resources = self.resources
            if not resources._lock.acquire(blocking=False):
                raise RuntimeError('A model graph executes one request at a time')
            try:
                resources.adapter.validate_request(sampling, resources.config)
                extra = {}
                if control:
                    extra['service_cancellation_token'] = CancellationProbe(control.cancelled)
                    if control.publish:
                        extra['service_progress_state'] = ProgressSink(control.publish)
                        extra['runtime_progress_enabled'] = True
                warmup = resources.config.warmup_steps if resources.config.warmup and not resources._warmed_up else None
                with resources.adapter.config_scope(resources.config):
                    result = self._inference.infer(sampling, warmup_steps=warmup, request_extra=extra)
                resources._warmed_up = True
            finally:
                resources._lock.release()
            self.last_metrics = result.metrics
            self.last_warmup = getattr(result, 'extra', {}).get('warmup')
            self.get_output(0).set((Path(directory) / 'result.mp4').read_bytes())
            progress = extra.get('service_progress_state')
            if progress is not None:
                progress.complete()
        return Status.ok()

    def serialize(self):
        value = json.loads(super().serialize())
        value.update(params=self.params.to_dict(), model_path=self.model_path)
        return json.dumps(value)

    def deserialize(self, value):
        data = json.loads(value)
        self.params = DiTParams(**data.get('params', {}))
        self.model_path = data.get('model_path', self.params.model_path)
        from nndeploy.e_dit.nodes.graph import clear_ports, materialize_children
        if self._inference is None:
            materialize_children(self, data)
            self._inference = self.get_node("inference")
        if self._inference is not None:
            clear_ports(self._inference)
        status = super().deserialize(value)
        self._inference = self.get_node('inference')
        if self._inference is None:
            raise ValueError('Model graph must contain the inference graph')
        return status


class _Creator(NodeCreator):
    def create_node(self, name, inputs, outputs):
        from nndeploy.e_dit.nodes.graph import loading_graph
        token = loading_graph.set(True)
        try:
            self.node = EraserDiTNode(name, inputs, outputs)
            return self.node
        finally:
            loading_graph.reset(token)


_creator = _Creator()


class DiTGraph(Graph):
    def __init__(self, params: DiTParams, name='eraserdit'):
        inputs = [Edge('video'), Edge('mask'), Edge('sampling')]
        outputs = [Edge('result')]
        super().__init__(name, inputs, outputs)
        self.model = EraserDiTNode('eraserdit', inputs, outputs, params)
        self.set_loop_count(1)
        self.set_input_type(str)
        self.set_input_type(str)
        self.set_input_type(dict)
        self.set_output_type(bytes)

    @property
    def resources(self):
        return self.model.resources

    def infer(self, video, mask, **sampling):
        if self.resources is None:
            raise RuntimeError('Initialize the Graph before inference')
        self.get_input(0).set(str(video))
        self.get_input(1).set(str(mask))
        self.get_input(2).set(sampling)
        status = self.run()
        from nndeploy.base import StatusCode
        if status != StatusCode.Ok:
            raise RuntimeError(f'EraserDiT failed: {status}')
        return self.get_output(0).get_graph_output()


def build_graph(params=None, name='eraserdit'):
    return DiTGraph(params if isinstance(params, DiTParams) else DiTParams(**(params or {})), name)


def server_config(*, isolation='process'):
    return {
        'inputs': {
            'video': {'type': 'file', 'media_type': 'video/mp4', 'edge': 'video'},
            'mask': {'type': 'file', 'edge': 'mask'},
            'sampling': {'type': 'json', 'edge': 'sampling', 'required': False, 'default': {}},
        },
        'outputs': {'video': {'type': 'file', 'media_type': 'video/mp4', 'edge': 'result'}},
        'streaming': {'enabled': True, 'max_duration_seconds': 3600},
        'execution': {'isolation': isolation, 'startup_timeout_seconds': 1800, 'shutdown_timeout_seconds': 180,
                      'execution_timeout_seconds': 3600 if isolation == 'process' else None,
                      'allow_child_processes': isolation == 'process'},
    }
