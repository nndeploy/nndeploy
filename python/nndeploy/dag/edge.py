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
        # 检查传入的数据是否为nndeploy框架中的Buffer或Tensor类型
        # isinstance()函数用于判断对象是否为指定类型的实例
        # 这里使用元组(nndeploy.device.Buffer, nndeploy.device.Tensor)来同时检查两种类型
        # 如果data是Buffer或Tensor中的任意一种类型，条件为True
        if isinstance(data, (nndeploy.device.Buffer, nndeploy.device.Tensor)):
            status = super().set(data, True)
        elif isinstance(data, np.ndarray):
            status = super().set(data)
        elif issubclass(type(data), nndeploy.base.Param):
            status = super().set(data, True)
        else: # 处理其他类型的数据
            status = self.set(data)
        if status != nndeploy.base.StatusCode.Ok:
            raise ValueError("Failed to set data")
        return nndeploy.base.Status.ok()
        
    def create_buffer(self, device: nndeploy.device.Device, desc: nndeploy.device.BufferDesc):
        return super().create(device, desc)

    def create_tensor(self, device: nndeploy.device.Device, desc: nndeploy.device.TensorDesc, tensor_name: str = ""):
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
    
    def get_numpy(self, node: _C.dag.Node) -> np.ndarray:
        return super().get_numpy(node)
        
    def get_graph_output_numpy(self) -> np.ndarray:
        return super().get_graph_output_numpy()
    
    def get_param(self, node: _C.dag.Node) -> nndeploy.base.Param:
        return super().get_param(node)
        
    def get_graph_output_param(self) -> nndeploy.base.Param:
        return super().get_graph_output_param()
        
    def get(self, node: _C.dag.Node = None):
        return self.get(node)
        
    def get_graph_output(self):
        return self.get_graph_output()
        
    def get_index(self, node: _C.dag.Node) -> int:
        return super().get_index(node)
        
    def reset_index(self):
        return super().reset_index()
        
    def get_graph_output_index(self) -> int:
        return super().get_graph_output_index()
        
    def get_position(self, node: _C.dag.Node) -> int:
        return super().get_position(node)
        
    def get_graph_output_position(self) -> int:
        return super().get_graph_output_position()
        
    def update(self, node: _C.dag.Node) -> nndeploy.base.EdgeUpdateFlag:
        return super().update(node)
        
    def mark_graph_output(self) -> bool:
        return super().mark_graph_output()
        
    def increase_producers(self, producers: list[_C.dag.Node]):
        return super().increase_producers(producers)
        
    def increase_consumers(self, consumers: list[_C.dag.Node]):
        return super().increase_consumers(consumers)
        
    def request_terminate(self) -> bool:
        return super().request_terminate()
      
    def set_type_name(self, type_name: str):
        return super().set_type_name(type_name)

    def get_type_name(self) -> str:
        return super().get_type_name()
    
    def set_type_info(self, type_info: EdgeTypeInfo):
        return super().set_type_info(type_info)
    
    def get_type_info(self) -> EdgeTypeInfo:
        return super().get_type_info()
    
    def check_type_info(self, type_info: EdgeTypeInfo) -> bool:
        return super().check_type_info(type_info)


accepted_edge_type_map = {
    "nndeploy.device.Buffer": ["nndeploy::device::Buffer"],
    "nndeploy::device::Buffer": ["nndeploy.device.Buffer"],
    "nndeploy.device.Tensor": ["nndeploy::device::Tensor"],
    "nndeploy::device::Tensor": ["nndeploy.device.Tensor"],
    "nndeploy.base.Param": ["nndeploy::base::Param"],
    "nndeploy::base::Param": ["nndeploy.base.Param"],
}

def add_accepted_edge_type_map(edge_type_map):
    global accepted_edge_type_map
    if isinstance(edge_type_map, dict):
        # 检查值的类型来判断是哪种字典类型
        if edge_type_map and isinstance(next(iter(edge_type_map.values())), list):
            accepted_edge_type_map.update(edge_type_map)
        else:
            for edge_type, edge_type_item in edge_type_map.items():
                if edge_type in accepted_edge_type_map:
                    accepted_edge_type_map[edge_type].append(edge_type_item)        
                else:
                    accepted_edge_type_map[edge_type] = [edge_type_item]
                    
                if edge_type_item in accepted_edge_type_map:
                    accepted_edge_type_map[edge_type_item].append(edge_type)
                else:
                    accepted_edge_type_map[edge_type_item] = [edge_type]
                    
    
def sub_accepted_edge_type_map(edge_type_map: Union[dict[str, list[str]], dict[str, str]]):
    global accepted_edge_type_map
    for edge_type, edge_type_list in edge_type_map.items():
        if edge_type in accepted_edge_type_map:
            for edge_type_item in edge_type_list:
                if edge_type_item in accepted_edge_type_map[edge_type]:
                    accepted_edge_type_map[edge_type].remove(edge_type_item)
                    
def get_accepted_edge_type_map():
    global accepted_edge_type_map
    return accepted_edge_type_map

def get_accepted_edge_type_json():
    import json
    
    # 将accepted_edge_type_map转换为JSON格式
    edge_type_data = {"accepted_edge_types": accepted_edge_type_map}
    
    # 序列化为JSON字符串
    edge_type_json = json.dumps(edge_type_data, ensure_ascii=False, indent=2)
    
    return edge_type_json