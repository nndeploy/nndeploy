"""Editor-ready full workflow with configurable inputs and a video file output."""
import json
from pathlib import Path

from nndeploy.base import Status
from nndeploy.basic import InputStr
from nndeploy.dag import Edge, Graph, Node, NodeCreator, NodeType, IOType, register_node
from nndeploy.e_dit.algorithms.eraserdit.graph import EraserDiTNode
from nndeploy.e_dit.params import DiTParams


class SamplingInput(Node):
    def __init__(self, name, inputs, outputs):
        super().__init__(name, inputs, outputs)
        self.set_key('nndeploy.e_dit.SamplingInput')
        self.set_node_type(NodeType.Input)
        self.set_io_type(IOType.Json)
        self.set_output_type(dict)
        self.sampling = dict(prompt='', seed=42, num_inference_steps=50,
                             guidance_scale=3., strength=.8,
                             runtime_mode='windowed_streaming', streaming_cache_dtype='uint8')

    def run(self):
        self.get_output(0).set(dict(self.sampling))
        return Status.ok()

    def serialize(self):
        value = json.loads(super().serialize())
        value['sampling'] = self.sampling
        return json.dumps(value)

    def deserialize(self, value):
        self.sampling = json.loads(value)['sampling']
        return super().deserialize(value)


class VideoOutput(Node):
    def __init__(self, name, inputs, outputs):
        super().__init__(name, inputs, outputs)
        self.set_key('nndeploy.e_dit.VideoOutput')
        self.set_node_type(NodeType.Output)
        self.set_io_type(IOType.Video)
        self.add_io_param('path_')
        self.set_input_type(bytes)
        self.path_ = 'result.mp4'

    def run(self):
        path = Path(self.path_)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.get_input(0).get(self))
        return Status.ok()

    def serialize(self):
        value = json.loads(super().serialize()); value['path_'] = self.path_
        return json.dumps(value)

    def deserialize(self, value):
        self.path_ = json.loads(value)['path_']
        return super().deserialize(value)


class _Creator(NodeCreator):
    def __init__(self, node_class):
        super().__init__(); self.node_class = node_class

    def create_node(self, name, inputs, outputs):
        self.node = self.node_class(name, inputs, outputs)
        return self.node


_creators = [_Creator(SamplingInput), _Creator(VideoOutput)]


def build_workflow(params, *, video='', mask='', output='result.mp4', sampling=None):
    graph = Graph('eraserdit_workflow')
    edges = [Edge(name) for name in ('video', 'mask', 'sampling', 'result')]
    graph.video = InputStr('video_input', [], [edges[0]]); graph.video.str_ = str(video)
    graph.mask = InputStr('mask_input', [], [edges[1]]); graph.mask.str_ = str(mask)
    graph.sampling = SamplingInput('sampling_input', [], [edges[2]])
    graph.sampling.sampling.update(sampling or {})
    graph.model = EraserDiTNode('eraserdit', edges[:3], [edges[3]],
                               params if isinstance(params, DiTParams) else DiTParams(**params))
    graph.output = VideoOutput('video_output', [edges[3]], []); graph.output.path_ = str(output)
    graph.set_loop_count(1)
    return graph

# Targets for lazy registration of serialized node keys.
_SamplingInput_creator, _VideoOutput_creator = _creators
