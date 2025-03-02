import nndeploy._nndeploy_internal as _C

from enum import Enum
from typing import Union
import numpy as np
import json

import nndeploy.base
import nndeploy.device


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

class Node(_C.dag.Node):
    def __init__(self, name: str, inputs=None, outputs=None):
        if inputs is None and outputs is None:
            super().__init__(name)
        elif isinstance(inputs, list) and isinstance(outputs, list):
            super().__init__(name, inputs, outputs)
        else:
            super().__init__(name, inputs, outputs)
            
    def get_name(self) -> str:
        return super().get_name()
        
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
        
    def set_compiled_flag(self, flag: bool):
        return super().set_compiled_flag(flag)
        
    def get_compiled_flag(self) -> bool:
        return super().get_compiled_flag()
        
    def set_stream(self, stream):
        return super().set_stream(stream)
        
    def get_stream(self):
        return super().get_stream()
        
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
        
    def __call__(self, inputs, outputs_name=None, param=None):
        if outputs_name is None:
            outputs_name = []
        return super().__call__(inputs, outputs_name, param)
        
    def functor_without_graph(self, inputs, outputs_name=None, param=None):
        if outputs_name is None:
            outputs_name = []
        return super().functor_without_graph(inputs, outputs_name, param)
        
    def functor_with_graph(self, inputs, outputs_name=None, param=None):
        if outputs_name is None:
            outputs_name = []
        return super().functor_with_graph(inputs, outputs_name, param)
        
    def functor_dynamic(self, inputs, outputs_name=None, param=None):
        if outputs_name is None:
            outputs_name = []
        return super().functor_dynamic(inputs, outputs_name, param)
        
    def check_inputs(self, inputs) -> bool:
        return super().check_inputs(inputs)
        
    def check_outputs(self, outputs_name) -> bool:
        return super().check_outputs(outputs_name)
        
    def get_real_outputs_name(self, outputs_name):
        return super().get_real_outputs_name(outputs_name)


