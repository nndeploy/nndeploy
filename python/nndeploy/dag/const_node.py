import nndeploy._nndeploy_internal as _C
import nndeploy.base
import nndeploy.device
from typing import List, Union, Optional

from .util import *
from .base import EdgeTypeInfo
from .edge import Edge
from .node import Node, NodeDesc

class ConstNode(_C.dag.ConstNode):
    def __init__(self, name: str, inputs=None, outputs=None):
        if inputs is None and outputs is None:
            super().__init__(name)
        elif isinstance(inputs, list) and isinstance(outputs, list):
            super().__init__(name, inputs, outputs)
        else:
            super().__init__(name, inputs, outputs)
            
    def set_output_type(self, output_type: type):
        edge_type_info = EdgeTypeInfo()
        edge_type_info.set_type(output_type)
        return _C.dag.Node.set_output_type_info(self, edge_type_info)
            
    def update_input(self):
        print("must be override")
        
    def init(self):
        return super().init()
        
    def deinit(self):
        return super().deinit()
        
    def run(self):
        print("must be override")

