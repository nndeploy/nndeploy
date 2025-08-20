

from tempfile import tempdir
import numpy as np
import json

import nndeploy.dag

class TemplatePy(nndeploy.dag.Node):
    def __init__(self, name, inputs: list[nndeploy.dag.Edge] = None, outputs: list[nndeploy.dag.Edge] = None):
        super().__init__(name, inputs, outputs) # self, name, inputs, outputs
        super().set_key("nndeploy.template_py.TemplatePy") # key
        super().set_desc("TemplatePy: TemplatePy") # desc
        self.set_input_type(np.ndarray) # input_type
        self.set_output_type(np.ndarray) # output_type
        
    def run(self):
        input_edge = self.get_input(0) # input_edge
        input_numpy = input_edge.get(self) # input_numpy
        # bgr->gray
        gray = np.dot(input_numpy[...,:3], [0.114, 0.587, 0.299])
        gray = gray.astype(np.uint8)
        self.get_output(0).set(gray) # output_edge
        return nndeploy.base.Status.ok()
    
    def serialize(self):
        json_str = super().serialize()
        return json_str
    
    def deserialize(self, target: str):
        json_obj = json.loads(target)
        return super().deserialize(target)
    
    
class TemplatePyCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = TemplatePy(name, inputs, outputs)
        return self.node
    
template_py_node_creator = TemplatePyCreator()
nndeploy.dag.register_node("nndeploy.template_py.TemplatePy", template_py_node_creator)