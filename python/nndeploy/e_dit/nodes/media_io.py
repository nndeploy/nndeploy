"""Algorithm-independent editor inputs and media output."""
import json
from pathlib import Path
from nndeploy.base import Status
from nndeploy.dag import Node, NodeCreator, NodeType, IOType

class GenerationSamplingInput(Node):
    def __init__(self,name,inputs,outputs):
        super().__init__(name,inputs,outputs)
        self.set_key('nndeploy.e_dit.GenerationSamplingInput')
        self.set_node_type(NodeType.Input)
        self.set_io_type(IOType.Json)
        self.set_output_type(dict)
        self.sampling = {}
    def run(self):
        self.get_output(0).set(dict(self.sampling))
        return Status.ok()
    def serialize(self):
        value = json.loads(super().serialize()); value['sampling'] = self.sampling
        return json.dumps(value)
    def deserialize(self,value):
        self.sampling = json.loads(value)['sampling']
        return super().deserialize(value)

class GeneratedVideoOutput(Node):
    def __init__(self,name,inputs,outputs):
        super().__init__(name,inputs,outputs)
        self.set_key('nndeploy.e_dit.GeneratedVideoOutput')
        self.set_node_type(NodeType.Output)
        self.set_io_type(IOType.Video)
        self.set_input_type(bytes)
        self.path_ = 'result.mp4'
        self.add_io_param('path_')
    def run(self):
        path = Path(self.path_); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(self.get_input(0).get(self))
        return Status.ok()
    def serialize(self):
        value = json.loads(super().serialize()); value['path_'] = self.path_
        return json.dumps(value)
    def deserialize(self,value):
        self.path_ = json.loads(value)['path_']
        return super().deserialize(value)

class Creator(NodeCreator):
    def __init__(self,cls):
        super().__init__(); self.cls = cls
    def create_node(self,name,inputs,outputs):
        self.node = self.cls(name,inputs,outputs)
        return self.node
_GenerationSamplingInput_creator = Creator(GenerationSamplingInput)
_GeneratedVideoOutput_creator = Creator(GeneratedVideoOutput)
