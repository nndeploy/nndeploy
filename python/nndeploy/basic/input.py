
import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.device
import nndeploy.dag

import json

import numpy as np

class InputStr(nndeploy.dag.Node):
    def __init__(self, name, inputs: [nndeploy.dag.Edge] = [], outputs: [nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key("nndeploy.basic.InputStr")
        self.set_desc("Input Str on python")
        self.set_output_type(str)
        self.str_ = ""
    
    def run(self) -> bool:
        output_edge = self.get_output(0) # 获取输出边
        output_edge.set(self.str_) # 将输出写入到输出边中
        return nndeploy.base.Status.ok()
        
    def serialize(self):
        self.add_required_param("str_")
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        json_obj["str_"] = self.str_
        return json.dumps(json_obj)

    def deserialize(self, target: str):
        json_obj = json.loads(target)
        self.str_ = json_obj["str_"]
        return super().deserialize(target)

class InputStrCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = InputStr(name, inputs, outputs)
        return self.node

input_str_node_creator = InputStrCreator()
nndeploy.dag.register_node("nndeploy.basic.InputStr", input_str_node_creator)
