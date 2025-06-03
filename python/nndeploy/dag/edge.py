import nndeploy._nndeploy_internal as _C

from enum import Enum
from typing import Union
import numpy as np
import json

import nndeploy.base
import nndeploy.device

from .base import EdgeTypeInfo

class Edge(_C.dag.Edge):
    def __init__(self, name: str = ""):
        super().__init__(name)
        self.parallel_type = nndeploy.base.ParallelType.kParallelTypeNone
        self.index = 0

    def get_name(self) -> str:
        return super().get_name()
    
    def set_queue_max_size(self, queue_max_size: int):
        return super().set_queue_max_size(queue_max_size)
        
    def get_queue_max_size(self) -> int:
        return super().get_queue_max_size()
        
    def get_parallel_type(self) -> nndeploy.base.ParallelType:
        return super().get_parallel_type()
        
    def set_parallel_type(self, parallel_type: nndeploy.base.ParallelType):
        return super().set_parallel_type(parallel_type)
        
    def construct(self):
        return super().construct()
        
    def set(self, data: any):
        if isinstance(data, (nndeploy.device.Buffer, nndeploy.device.Tensor)):
            # self.type_name = "nd." + type(data).__name__
            status = super().set(data, True)
            if status != nndeploy.base.StatusCode.Ok:
                raise ValueError("Failed to set data")
        else: # 处理其他类型的数据
            # self.type_name = type(data).__name__
            # if self.parallel_type == nndeploy.base.ParallelType.kParallelTypeNone or self.parallel_type == nndeploy.base.ParallelType.Sequential or self.parallel_type == nndeploy.base.ParallelType.Task:
            #     self.data = data
            # elif self.parallel_type == nndeploy.base.ParallelType.Pipeline:
            #     raise ValueError("Parallel type is not supported")
            # print(self.get_type_name())
            print(data)
            self.set_any(data)
            # print(self.get_type_name())
        self.index += 1
        self.position = self.index
        self.type_name = type(data).__module__ + "." + type(data).__name__
        return nndeploy.base.Status(nndeploy.base.StatusCode.Ok)
        
    def create_buffer(self, device: nndeploy.device.Device, desc: nndeploy.device.BufferDesc):
        # 创建Buffer时设置正确的类型名称
        # self.type_name = "nd.Buffer"
        buffer_type = nndeploy.device.Buffer
        self.type_name = buffer_type.__module__ + "." + buffer_type.__name__
        return super().create(device, desc)

    def create_tensor(self, device: nndeploy.device.Device, desc: nndeploy.device.TensorDesc, tensor_name: str = ""):
        # self.type_name = "nd.Tensor"
        tensor_type = nndeploy.device.Tensor
        self.type_name = tensor_type.__module__ + "." + tensor_type.__name__
        return super().create(device, desc, tensor_name)
        
    def notify_written(self, data: Union[nndeploy.device.Buffer, nndeploy.device.Tensor]):
        return super().notify_written(data)
        
    def get_buffer(self, node: _C.dag.Node) -> nndeploy.device.Buffer:
        return super().get_buffer(node)
        
    def get_graph_output_buffer(self) -> nndeploy.device.Buffer:
        return super().get_graph_output_buffer()
        
    def get_tensor(self, node: _C.dag.Node) -> nndeploy.device.Tensor:
        return super().get_tensor(node)
        
    def get_graph_output_tensor(self) -> nndeploy.device.Tensor:
        return super().get_graph_output_tensor()
        
    def get(self, node: _C.dag.Node = None):
        # if self.parallel_type == nndeploy.base.ParallelType.Sequential or self.parallel_type == nndeploy.base.ParallelType.kParallelTypeNone or self.parallel_type == nndeploy.base.ParallelType.Task:
        #     return self.data
        # elif self.parallel_type == nndeploy.base.ParallelType.Pipeline:
        #     raise ValueError("Parallel type is not supported")
        return self.get_any(node)
        
    def get_graph_output(self):
        # if self.parallel_type == nndeploy.base.ParallelType.Sequential or self.parallel_type == nndeploy.base.ParallelType.kParallelTypeNone or self.parallel_type == nndeploy.base.ParallelType.Task:
        #     return self.data
        # elif self.parallel_type == nndeploy.base.ParallelType.Pipeline:
        #     raise ValueError("Parallel type is not supported")
        return self.get_graph_output_any()
    def get_index(self, node: _C.dag.Node) -> int:
        if self.parallel_type == nndeploy.base.ParallelType.Sequential or self.parallel_type == nndeploy.base.ParallelType.kParallelTypeNone or self.parallel_type == nndeploy.base.ParallelType.Task:
            return self.index
        elif self.parallel_type == nndeploy.base.ParallelType.Pipeline:
            raise ValueError("Parallel type is not supported")
        
    def reset_index(self):
        return super().reset_index()
        
    def get_graph_output_index(self) -> int:
        if self.parallel_type == nndeploy.base.ParallelType.Sequential or self.parallel_type == nndeploy.base.ParallelType.kParallelTypeNone or self.parallel_type == nndeploy.base.ParallelType.Task:
            return self.index
        elif self.parallel_type == nndeploy.base.ParallelType.Pipeline:
            raise ValueError("Parallel type is not supported")
        
    def get_position(self, node: _C.dag.Node) -> int:
        if self.parallel_type == nndeploy.base.ParallelType.Sequential or self.parallel_type == nndeploy.base.ParallelType.kParallelTypeNone or self.parallel_type == nndeploy.base.ParallelType.Task:
            return self.position
        elif self.parallel_type == nndeploy.base.ParallelType.Pipeline:
            raise ValueError("Parallel type is not supported")
        
    def get_graph_output_position(self) -> int:
        if self.parallel_type == nndeploy.base.ParallelType.Sequential or self.parallel_type == nndeploy.base.ParallelType.kParallelTypeNone or self.parallel_type == nndeploy.base.ParallelType.Task:
            return self.position
        elif self.parallel_type == nndeploy.base.ParallelType.Pipeline:
            raise ValueError("Parallel type is not supported")
        
    def update(self, node: _C.dag.Node) -> nndeploy.base.EdgeUpdateFlag:
        if self.parallel_type == nndeploy.base.ParallelType.Sequential or self.parallel_type == nndeploy.base.ParallelType.kParallelTypeNone or self.parallel_type == nndeploy.base.ParallelType.Task:
            return super().update(node)
        elif self.parallel_type == nndeploy.base.ParallelType.Pipeline:
            raise ValueError("Parallel type is not supported")
        
    def mark_graph_output(self) -> bool:
        return super().mark_graph_output()
        
    def increase_producers(self, producers: list[_C.dag.Node]):
        return super().increase_producers(producers)
        
    def increase_consumers(self, consumers: list[_C.dag.Node]):
        return super().increase_consumers(consumers)
        
    def request_terminate(self) -> bool:
        return super().request_terminate()
      
    def get_type_name(self) -> str:
        return self.type_name
    
    def set_type_info(self, type_info: EdgeTypeInfo):
        return super().set_type_info(type_info)
    
    def get_type_info(self) -> EdgeTypeInfo:
        return super().get_type_info()
    
    def check_type_info(self, type_info: EdgeTypeInfo) -> bool:
        return super().check_type_info(type_info)
