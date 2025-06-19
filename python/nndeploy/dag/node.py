import nndeploy._nndeploy_internal as _C

from enum import Enum
from typing import Union
import numpy as np
import json

import nndeploy.base
import nndeploy.device

from .base import EdgeTypeInfo
from .edge import Edge

class NodeDesc(_C.dag.NodeDesc):
    def __init__(self, name: str, inputs: list[str] = None, outputs: list[str] = None, key: str = None):
        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = []
            
        if key is None:
            super().__init__(name, inputs, outputs)
        else:
            super().__init__(name, key, inputs, outputs)
    
    def get_key(self) -> str:
        return super().get_key()
        
    def get_name(self) -> str:
        return super().get_name()
        
    def get_inputs(self) -> list[str]:
        return super().get_inputs()
        
    def get_outputs(self) -> list[str]:
        return super().get_outputs()
    
    def serialize(self, target: str):
        return super().serialize(target)

    def deserialize(self, target: str):
        return super().deserialize(target)

class Node(_C.dag.Node):
    def __init__(self, name: str, inputs=None, outputs=None):
        if inputs is None and outputs is None:
            super().__init__(name)
        elif isinstance(inputs, list) and isinstance(outputs, list):
            super().__init__(name, inputs, outputs)
        else:
            super().__init__(name, inputs, outputs)
            
    def set_key(self, key: str):
        return super().set_key(key)
            
    def get_key(self) -> str:
        return super().get_key()
            
    def set_name(self, name: str):
        return super().set_name(name)
        
    def get_name(self) -> str:
        return super().get_name()
    
    def set_desc(self, desc: str):
        return super().set_desc(desc)
    
    def get_desc(self) -> str:
        return super().get_desc()
            
    def set_graph(self, graph):
        return super().set_graph(graph)
        
    def get_graph(self):
        return super().get_graph()
        
    def set_device_type(self, device_type: nndeploy.base.DeviceType):
        return super().set_device_type(device_type)
        
    def get_device_type(self) -> nndeploy.base.DeviceType:
        return super().get_device_type()
        
    def set_param(self, param):
        return super().set_param(param)
        
    def get_param(self):
        return super().get_param()
        
    def set_external_param(self, key: str, external_param):
        return super().set_external_param(key, external_param)
        
    def get_external_param(self, key: str):
        return super().get_external_param(key)
        
    def set_input(self, input, index: int = -1):
        return super().set_input(input, index)
        
    def set_output(self, output, index: int = -1):
        return super().set_output(output, index)
        
    def set_inputs(self, inputs):
        return super().set_inputs(inputs)
        
    def set_outputs(self, outputs):
        return super().set_outputs(outputs)
        
    # def set_input_shared_ptr(self, input, index: int = -1):
    #     return super().set_input_shared_ptr(input, index)
        
    # def set_output_shared_ptr(self, output, index: int = -1):
    #     return super().set_output_shared_ptr(output, index)
        
    # def set_inputs_shared_ptr(self, inputs):
    #     return super().set_inputs_shared_ptr(inputs)
        
    # def set_outputs_shared_ptr(self, outputs):
    #     return super().set_outputs_shared_ptr(outputs)
        
    def get_input(self, index: int = 0):
        return super().get_input(index)
        
    def get_output(self, index: int = 0):
        return super().get_output(index)
        
    def get_all_input(self):
        return super().get_all_input()
        
    def get_all_output(self):
        return super().get_all_output()
        
    def get_constructed(self) -> bool:
        return super().get_constructed()
        
    def set_parallel_type(self, parallel_type: nndeploy.base.ParallelType):
        return super().set_parallel_type(parallel_type)
        
    def get_parallel_type(self) -> nndeploy.base.ParallelType:
        return super().get_parallel_type()
        
    def set_inner_flag(self, flag: bool):
        return super().set_inner_flag(flag)
        
    def set_initialized_flag(self, flag: bool):
        return super().set_initialized_flag(flag)
        
    def get_initialized(self) -> bool:
        return super().get_initialized()
        
    def set_time_profile_flag(self, flag: bool):
        return super().set_time_profile_flag(flag)
        
    def get_time_profile_flag(self) -> bool:
        return super().get_time_profile_flag()
        
    def set_debug_flag(self, flag: bool):
        return super().set_debug_flag(flag)
        
    def get_debug_flag(self) -> bool:
        return super().get_debug_flag()
        
    def set_running_flag(self, flag: bool):
        return super().set_running_flag(flag)
        
    def is_running(self) -> bool:
        return super().is_running()
        
    def set_trace_flag(self, flag: bool):
        return super().set_trace_flag(flag)
        
    def get_trace_flag(self) -> bool:
        return super().get_trace_flag()
        
    def set_stream(self, stream):
        return super().set_stream(stream)
        
    def get_stream(self):
        return super().get_stream()
    
    def set_input_type_info(self, input_type_info: EdgeTypeInfo):
        return super().set_input_type_info(input_type_info)
    
    def set_input_type(self, input_type: type):
        """设置输入类型
        
        Args:
            input_type: 输入类型
            
        Returns:
            状态码
        """
        edge_type_info = EdgeTypeInfo()
        edge_type_info.set_type(input_type)
        return self.set_input_type_info(edge_type_info)
        
    def get_input_type_info(self) -> EdgeTypeInfo:
        return super().get_input_type_info()
    
    def set_output_type_info(self, output_type_info: EdgeTypeInfo):
        return super().set_output_type_info(output_type_info)
    
    def set_output_type(self, output_type: type):
        edge_type_info = EdgeTypeInfo()
        edge_type_info.set_type(output_type)
        return self.set_output_type_info(edge_type_info)
        
    def get_output_type_info(self) -> EdgeTypeInfo:
        return super().get_output_type_info()
        
    def init(self):
        return super().init()
        
    def deinit(self):
        return super().deinit()
        
    def get_memory_size(self) -> int:
        return super().get_memory_size()
        
    def set_memory(self, buffer: nndeploy.device.Buffer):
        return super().set_memory(buffer)
        
    def update_input(self):
        return super().update_input()
        
    def run(self):
        return super().run()
        
    def __call__(self, inputs):
        return super().__call__(inputs)
    
    def check_inputs(self, inputs) -> bool:
        return super().check_inputs(inputs)
        
    def check_outputs(self, outputs_name) -> bool:
        return super().check_outputs(outputs_name)
        
    def get_real_outputs_name(self, outputs_name):
        return super().get_real_outputs_name(outputs_name)
    
    def serialize(self) -> str:
        return super().serialize()
    
    def save_file(self, path: str):
        return super().save_file(path)
    
    def deserialize(self, target: str):
        return super().deserialize(target)
    
    def load_file(self, path: str):
        return super().load_file(path)


class NodeCreator(_C.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[Edge], outputs: list[Edge]):
        print("Must be implemented!!!")
        return None
            
        
def get_node_keys():
    return _C.dag.get_node_keys()
    
    
def register_node(node_key: str, node_creator: NodeCreator):
    return _C.dag.register_node(node_key, node_creator)


def create_node(node_key: str, node_name: str, inputs: list[Edge] = None, outputs: list[Edge] = None):
    if inputs is None and outputs is None:
        return _C.dag.create_node(node_key, node_name)
    else:
        return _C.dag.create_node(node_key, node_name, inputs, outputs)


def get_node_json(node_key: str):
    node_name = node_key.split("::")[-1]
    node_name = node_name.split(".")[-1]
    node = create_node(node_key, node_name)
    # print(node)
    if node is not None:
        json_str = node.serialize()
        return json_str


remove_node_keys = ["nndeploy::dag::Graph", "nndeploy::dag::RunningCondition"]


def add_remove_node_keys(node_keys: list[str]):
    global remove_node_keys
    remove_node_keys.extend(node_keys)
    
def sub_remove_node_keys(node_keys: list[str]):
    global remove_node_keys
    for node_key in node_keys:
        if node_key in remove_node_keys:
            remove_node_keys.remove(node_key)

def get_all_node_json():
    global remove_node_keys
    node_keys = get_node_keys()
    real_node_keys = []
    for node_key in node_keys:
        if node_key in remove_node_keys:
            continue
        real_node_keys.append(node_key)
    node_json = "{\"nodes\":["
    for node_key in real_node_keys:
        json = get_node_json(node_key)
        node_json += json
        if node_key != real_node_keys[-1]:
            node_json += ","
    node_json += "]}"
    # 美化json
    node_json = nndeploy.base.pretty_json_str(node_json)
    return node_json

