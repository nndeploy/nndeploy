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
        """
        初始化节点
        
        参数:
            name: 节点名称
            inputs: 输入边列表或单个输入边
            outputs: 输出边列表或单个输出边
        """
        if inputs is None and outputs is None:
            super().__init__(name)
        elif isinstance(inputs, list) and isinstance(outputs, list):
            super().__init__(name, inputs, outputs)
        else:
            super().__init__(name, inputs, outputs)
        self.inputs_type = []
        self.outputs_type = []
        
    def get_name(self) -> str:
        """获取节点名称"""
        return super().get_name()
        
    def set_graph(self, graph):
        """
        设置节点所属的图
        
        参数:
            graph: 图对象
        """
        return super().set_graph(graph)
        
    def get_graph(self):
        """获取节点所属的图"""
        return super().get_graph()
        
    def set_device_type(self, device_type: nndeploy.base.DeviceType):
        """
        设置设备类型
        
        参数:
            device_type: 设备类型
        """
        return super().set_device_type(device_type)
        
    def get_device_type(self) -> nndeploy.base.DeviceType:
        """获取设备类型"""
        return super().get_device_type()
        
    def set_param_cpp(self, param: nndeploy.base.Param):
        """
        设置C++参数
        
        参数:
            param: 参数对象
        """
        return super().set_param_cpp(param)
        
    def set_param(self, param):
        """
        设置参数
        
        参数:
            param: 参数对象
        """
        return super().set_param(param)
        
    def get_param_cpp(self) -> nndeploy.base.Param:
        """获取C++参数"""
        return super().get_param_cpp()
        
    def get_param(self):
        """获取参数"""
        return super().get_param()
        
    def set_external_param_cpp(self, external_param: nndeploy.base.Param):
        """
        设置外部C++参数
        
        参数:
            external_param: 外部参数对象
        """
        return super().set_external_param_cpp(external_param)
        
    def set_external_param(self, external_param):
        """
        设置外部参数
        
        参数:
            external_param: 外部参数对象
        """
        return super().set_external_param(external_param)
        
    def set_input(self, input, index: int = -1):
        """
        设置输入边
        
        参数:
            input: 输入边对象
            index: 输入边索引，默认为-1（添加到末尾）
        """
        return super().set_input(input, index)
        
    def set_output(self, output, index: int = -1):
        """
        设置输出边
        
        参数:
            output: 输出边对象
            index: 输出边索引，默认为-1（添加到末尾）
        """
        return super().set_output(output, index)
        
    def set_inputs(self, inputs):
        """
        设置所有输入边
        
        参数:
            inputs: 输入边对象列表
        """
        return super().set_inputs(inputs)
        
    def set_outputs(self, outputs):
        """
        设置所有输出边
        
        参数:
            outputs: 输出边对象列表
        """
        return super().set_outputs(outputs)
        
    def set_input_shared_ptr(self, input, index: int = -1):
        """
        设置输入边共享指针
        
        参数:
            input: 输入边对象
            index: 输入边索引，默认为-1（添加到末尾）
        """
        return super().set_input_shared_ptr(input, index)
        
    def set_output_shared_ptr(self, output, index: int = -1):
        """
        设置输出边共享指针
        
        参数:
            output: 输出边对象
            index: 输出边索引，默认为-1（添加到末尾）
        """
        return super().set_output_shared_ptr(output, index)
        
    def set_inputs_shared_ptr(self, inputs):
        """
        设置所有输入边共享指针
        
        参数:
            inputs: 输入边对象列表
        """
        return super().set_inputs_shared_ptr(inputs)
        
    def set_outputs_shared_ptr(self, outputs):
        """
        设置所有输出边共享指针
        
        参数:
            outputs: 输出边对象列表
        """
        return super().set_outputs_shared_ptr(outputs)
        
    def get_input(self, index: int = 0):
        """
        获取输入边
        
        参数:
            index: 输入边索引，默认为0
        """
        return super().get_input(index)
        
    def get_output(self, index: int = 0):
        """
        获取输出边
        
        参数:
            index: 输出边索引，默认为0
        """
        return super().get_output(index)
        
    def get_all_input(self):
        """获取所有输入边"""
        return super().get_all_input()
        
    def get_all_output(self):
        """获取所有输出边"""
        return super().get_all_output()
        
    def get_constructed(self) -> bool:
        """获取节点是否已构建"""
        return super().get_constructed()
        
    def set_parallel_type(self, parallel_type: nndeploy.base.ParallelType):
        """
        设置并行类型
        
        参数:
            parallel_type: 并行类型
        """
        return super().set_parallel_type(parallel_type)
        
    def get_parallel_type(self) -> nndeploy.base.ParallelType:
        """获取并行类型"""
        return super().get_parallel_type()
        
    def set_inner_flag(self, flag: bool):
        """
        设置内部标志
        
        参数:
            flag: 标志值
        """
        return super().set_inner_flag(flag)
        
    def set_initialized_flag(self, flag: bool):
        """
        设置初始化标志
        
        参数:
            flag: 标志值
        """
        return super().set_initialized_flag(flag)
        
    def get_initialized(self) -> bool:
        """获取初始化状态"""
        return super().get_initialized()
        
    def set_time_profile_flag(self, flag: bool):
        """
        设置时间分析标志
        
        参数:
            flag: 标志值
        """
        return super().set_time_profile_flag(flag)
        
    def get_time_profile_flag(self) -> bool:
        """获取时间分析标志"""
        return super().get_time_profile_flag()
        
    def set_debug_flag(self, flag: bool):
        """
        设置调试标志
        
        参数:
            flag: 标志值
        """
        return super().set_debug_flag(flag)
        
    def get_debug_flag(self) -> bool:
        """获取调试标志"""
        return super().get_debug_flag()
        
    def set_running_flag(self, flag: bool):
        """
        设置运行标志
        
        参数:
            flag: 标志值
        """
        return super().set_running_flag(flag)
        
    def is_running(self) -> bool:
        """获取运行状态"""
        return super().is_running()
        
    def set_stream(self, stream):
        """
        设置流
        
        参数:
            stream: 流对象
        """
        return super().set_stream(stream)
        
    def get_stream(self):
        """获取流"""
        return super().get_stream()
        
    def init(self):
        """初始化节点"""
        return super().init()
        
    def deinit(self):
        """反初始化节点"""
        return super().deinit()
        
    def get_memory_size(self) -> int:
        """获取内存大小"""
        return super().get_memory_size()
        
    def set_memory(self, buffer: nndeploy.device.Buffer):
        """
        设置内存
        
        参数:
            buffer: 缓冲区对象
        """
        return super().set_memory(buffer)
        
    def updata_input(self) -> nndeploy.base.EdgeUpdateFlag:
        """更新输入"""
        return super().updata_input()
        
    def run(self):
        """
        运行节点，需要在子类中实现
        """
        raise NotImplementedError("Node run method is not implemented")
        
    def __call__(self, inputs, outputs_name=None, param=None):
        """
        调用节点
        
        参数:
            inputs: 输入列表
            outputs_name: 输出名称列表，默认为空列表
            param: 参数对象，默认为None
        """
        if outputs_name is None:
            outputs_name = []
        if param is None:
            param = nndeploy.base.Param()
        return super().__call__(inputs, outputs_name, param)
        
    def functor_without_graph(self, inputs, outputs_name=None, param=None):
        """
        不使用图执行节点
        
        参数:
            inputs: 输入列表
            outputs_name: 输出名称列表，默认为空列表
            param: 参数对象，默认为None
        """
        if outputs_name is None:
            outputs_name = []
        return super().functor_without_graph(inputs, outputs_name, param)
        
    def functor_with_graph(self, inputs, outputs_name=None, param=None):
        """
        使用图执行节点
        
        参数:
            inputs: 输入列表
            outputs_name: 输出名称列表，默认为空列表
            param: 参数对象，默认为None
        """
        if outputs_name is None:
            outputs_name = []
        return super().functor_with_graph(inputs, outputs_name, param)
        
    def functor_dynamic(self, inputs, outputs_name=None, param=None):
        """
        动态执行节点
        
        参数:
            inputs: 输入列表
            outputs_name: 输出名称列表，默认为空列表
            param: 参数对象，默认为None
        """
        if outputs_name is None:
            outputs_name = []
        return super().functor_dynamic(inputs, outputs_name, param)
        
    def check_inputs(self, inputs) -> bool:
        """
        检查输入
        
        参数:
            inputs: 输入列表
        """
        return super().check_inputs(inputs)
        
    def check_outputs(self, outputs_name) -> bool:
        """
        检查输出
        
        参数:
            outputs_name: 输出名称列表
        """
        return super().check_outputs(outputs_name)


