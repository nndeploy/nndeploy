"""LTX-2 model node with resident resources and a serializable native DAG."""
from copy import deepcopy
import json
import torch
from nndeploy.base import Status, StatusCode
from nndeploy.dag import Graph, Edge, NodeCreator, IOType
from nndeploy.e_dit.params import DiTParams
from nndeploy.e_dit.nodes.graph import loading_graph, clear_ports, materialize_children, release_bindings
from .nodes.request import LTX2RequestGraph

class LTX2Node(Graph):
    def __init__(self, name, inputs, outputs, params=None):
        super().__init__(name,inputs,outputs)
        self.set_key('nndeploy.e_dit.LTX2')
        self.set_input_type(dict)
        self.set_output_type(bytes)
        self.params = params or DiTParams(model='ltx2')
        self.model_path = self.params.model_path
        self.set_io_type(IOType.Model)
        self.add_io_param('model_path')
        self.resources = None
        self.last_metrics = None
        self._inference = None
        if not loading_graph.get():
            self._inference = LTX2RequestGraph(two_stage=self.params.model=='ltx2_two_stage')
            self.add_node(self._inference)

    def get_param(self): return self.params

    def set_param(self, param):
        if not isinstance(param, DiTParams):
            raise TypeError('LTX-2 requires DiTParams')
        if self.resources is not None:
            raise RuntimeError('Deinitialize before changing model parameters')
        if param.model != self.params.model:
            raise ValueError('Build a new graph to change between single- and two-stage topology')
        self.params = param
        self.model_path = param.model_path
        return Status.ok()

    def init(self):
        if self.resources is not None: return Status.ok()
        from nndeploy.e_dit.resources import DiTResources
        params = self.params.to_dict()
        params['model_path'] = self.model_path
        resources = DiTResources(DiTParams(**params))
        try:
            resources.open()
            self._inference.bind(resources)
            status = super().init()
            if status != StatusCode.Ok:
                raise RuntimeError(f'LTX-2 graph initialization failed: {status}')
        except BaseException:
            try: release_bindings(self._inference)
            finally: resources.close()
            raise
        self.resources = resources
        return Status.ok()

    def deinit(self):
        if self.resources is not None:
            try: super().deinit()
            finally:
                self.resources.close()
                self.resources = None
        return Status.ok()

    def infer_batch(self, **options):
        from .sampling import LTX2SamplingParams
        from .state import LTX2State
        from nndeploy.dag.execution import current_execution
        from nndeploy.e_dit.execution import CancellationProbe, ProgressSink
        from nndeploy.e_dit.nodes.control import service_checkpoint
        resources = self.resources
        if resources is None: raise RuntimeError('Initialize LTX-2 before inference')
        sampling = LTX2SamplingParams(**deepcopy(options))
        resources.adapter.validate_request(sampling, resources.config)
        if not resources._lock.acquire(blocking=False):
            raise RuntimeError('A model graph executes one request at a time')
        try:
            state = LTX2State(sampling_params=sampling)
            from nndeploy.e_dit.utils.random import request_generators
            state.seeds, state.generator = request_generators(sampling.seed, state.batch_size, device=sampling.generator_device)
            control = current_execution()
            if control:
                state.extra['service_cancellation_token'] = CancellationProbe(control.cancelled)
                if control.publish: state.extra['service_progress_state'] = ProgressSink(control.publish)
            with resources.adapter.config_scope(resources.config), torch.inference_mode():
                service_checkpoint(state, resources.config, phase='request')
                result = self._inference.invoke(state)
            self.last_metrics = result.metrics
            progress = state.extra.get('service_progress_state')
            if progress: progress.complete()
            return result.extra['result']
        finally:
            try:
                # Failed requests must not strand models on the accelerator.
                for name, enabled in [('transformer',resources.config.dit_cpu_offload),
                                      ('text_encoder',resources.config.text_encoder_cpu_offload),
                                      ('vae',resources.config.vae_cpu_offload),
                                      ('audio_vae',resources.config.vae_cpu_offload),
                                      ('vocoder',resources.config.vae_cpu_offload)]:
                    if enabled: resources.modules[name].to('cpu')
            finally:
                resources._lock.release()

    def run(self):
        options = self.get_input(0).get(self) or {}
        if not isinstance(options,dict): raise ValueError('sampling must be an object')
        if isinstance(options.get('prompt'),list) or options.get('num_outputs_per_prompt',1) != 1:
            raise ValueError('The MP4 output port accepts one result; use infer_batch for batches')
        result = self.infer_batch(**options)
        from nndeploy.e_dit.video.audio_video import encode_audio_video
        audio = result.audio[0] if result.audio is not None else None
        self.get_output(0).set(encode_audio_video(result.output[0], fps=options.get('fps',24),
                                               audio=audio, sample_rate=result.audio_sample_rate))
        return Status.ok()

    def serialize(self):
        value = json.loads(super().serialize())
        value.update(params=self.params.to_dict(),model_path=self.model_path)
        return json.dumps(value)

    def deserialize(self,value):
        data = json.loads(value)
        self.params = DiTParams(**data['params'])
        self.model_path = data.get('model_path',self.params.model_path)
        if self._inference is None: materialize_children(self,data)
        else: clear_ports(self._inference)
        status = super().deserialize(value)
        self._inference = self.get_node('inference')
        if self._inference is None: raise ValueError('Missing LTX-2 inference graph')
        return status

class _Creator(NodeCreator):
    def create_node(self,name,inputs,outputs):
        token = loading_graph.set(True)
        try:
            self.node = LTX2Node(name,inputs,outputs)
            return self.node
        finally: loading_graph.reset(token)
_creator = _Creator()

class LTX2Graph(Graph):
    def __init__(self,params,name='ltx2'):
        inputs, outputs = [Edge('sampling')],[Edge('result')]
        super().__init__(name,inputs,outputs)
        self.set_input_type(dict)
        self.set_output_type(bytes)
        self.set_loop_count(1)
        self.model = LTX2Node('ltx2',inputs,outputs,params)

    @property
    def resources(self): return self.model.resources

    def infer(self,prompt=None,**sampling):
        if prompt is not None: sampling['prompt'] = prompt
        self.get_input(0).set(sampling)
        status = self.run()
        if status != StatusCode.Ok: raise RuntimeError(f'LTX-2 inference failed: {status}')
        return self.get_output(0).get_graph_output()

    def infer_batch(self,**sampling):
        return self.model.infer_batch(**sampling)

def server_config(*,isolation='process'):
    return {'inputs':{'sampling':{'type':'json','edge':'sampling'}},
            'outputs':{'video':{'type':'file','media_type':'video/mp4','edge':'result'}},
            'streaming':{'enabled':True,'max_duration_seconds':3600},
            'execution':{'isolation':isolation,'startup_timeout_seconds':1800,
                         'execution_timeout_seconds':3600 if isolation=='process' else None}}
