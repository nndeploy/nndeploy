import nndeploy._nndeploy_internal as _C
import nndeploy.base
import nndeploy.device
from typing import List, Union, Optional

from .util import *
from .base import EdgeTypeInfo
from .edge import Edge
from .node import Node, NodeDesc

class CompositeNode(_C.dag.CompositeNode):
    def __init__(self, name: str, inputs=None, outputs=None):
        if inputs is None and outputs is None:
            super().__init__(name)
        elif isinstance(inputs, list) and isinstance(outputs, list):
            super().__init__(name, inputs, outputs)
        else:
            super().__init__(name, inputs, outputs)
            
    def set_input_type(self, input_type: type):
        edge_type_info = EdgeTypeInfo()
        edge_type_info.set_type(input_type)
        return _C.dag.Node.set_input_type_info(self, edge_type_info)
            
    def set_output_type(self, output_type: type):
        edge_type_info = EdgeTypeInfo()
        edge_type_info.set_type(output_type)
        return _C.dag.Node.set_output_type_info(self, edge_type_info)
            
    def set_input(self, input, index=-1):
        return super().set_input(input, index)
        
    def set_output(self, output, index=-1):
        return super().set_output(output, index)
        
    def set_inputs(self, inputs):
        return super().set_inputs(inputs)
        
    def set_outputs(self, outputs):
        return super().set_outputs(outputs)
        
    def create_edge(self, name):
        return super().create_edge(name)
        
    def add_edge(self, edge, is_external=True):
        return super().add_edge(edge, is_external)
        
    def get_edge(self, name):
        return super().get_edge(name)
        
    def update_edge(self, edge_wrapper, edge, is_external=True):
        return super().update_edge(edge_wrapper, edge, is_external)
        
    def create_node(self, desc):
        return super().create_node(desc)
        
    def set_node_desc(self, node, desc):
        return super().set_node_desc(node, desc)
        
    def add_node(self, node):
        return super().add_node(node)
        
    def update_node_io(self, node, inputs, outputs):
        return super().update_node_io(node, inputs, outputs)
        
    def mark_input_edge(self, inputs):
        return super().mark_input_edge(inputs)
        
    def mark_output_edge(self, outputs):
        return super().mark_output_edge(outputs)
        
    def get_node(self, name):
        return super().get_node(name)
        
    def get_node_by_key(self, key):
        return super().get_node_by_key(key)
        
    def get_nodes_by_key(self, key):
        return super().get_nodes_by_key(key)
        
    def set_node_param(self, node_name, param):
        return super().set_node_param(node_name, param)
        
    def get_node_param(self, node_name):
        return super().get_node_param(node_name)
        
    def default_param(self):
        return super().default_param()
        
    def init(self):
        return super().init()
        
    def deinit(self):
        return super().deinit()
        
    def run(self):
        print("must be override")
        
    def get_edge_wrapper(self, edge_or_name):
        return super().get_edge_wrapper(edge_or_name)
        
    def get_node_wrapper(self, node_or_name):
        return super().get_node_wrapper(node_or_name)
        
    def serialize(self, json=None, allocator=None):
        if json is not None and allocator is not None:
            return super().serialize(json, allocator)
        return super().serialize()
        
    def deserialize(self, json_or_str):
        return super().deserialize(json_or_str)
        
    def sort_dfs(self):
        return super().sort_dfs()

