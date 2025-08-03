import nndeploy._nndeploy_internal as _C

from enum import Enum
from typing import Union
import numpy as np
import json

import nndeploy.base
import nndeploy.device


name_to_node_type = {
    "Input": _C.dag.NodeType.Input,
    "Output": _C.dag.NodeType.Output,
    "Intermediate": _C.dag.NodeType.Intermediate,
}


node_type_to_name = {v: k for k, v in name_to_node_type.items()}


class NodeType(_C.dag.NodeType):
    Input = _C.dag.NodeType.Input
    Output = _C.dag.NodeType.Output
    Intermediate = _C.dag.NodeType.Intermediate

    @classmethod
    def from_name(cls, name: str):
        name_to_node_type = {
            "Input": cls.Input,
            "Output": cls.Output,
            "Intermediate": cls.Intermediate
        }
        if name not in name_to_node_type:
            raise ValueError(f"不支持的节点类型: {name}")
        else:
            return cls(name_to_node_type[name])



name_to_edge_type_flag = {
    "kBuffer": _C.dag.EdgeTypeFlag.kBuffer,
    "kCvMat": _C.dag.EdgeTypeFlag.kCvMat,
    "kTensor": _C.dag.EdgeTypeFlag.kTensor,
    "kParam": _C.dag.EdgeTypeFlag.kParam,
    "kAny": _C.dag.EdgeTypeFlag.kAny,
    "kNone": _C.dag.EdgeTypeFlag.kNone,
}


edge_type_flag_to_name = {v: k for k, v in name_to_edge_type_flag.items()}


class EdgeTypeFlag(_C.dag.EdgeTypeFlag):
    kBuffer = _C.dag.EdgeTypeFlag.kBuffer
    kCvMat = _C.dag.EdgeTypeFlag.kCvMat  
    kTensor = _C.dag.EdgeTypeFlag.kTensor
    kParam = _C.dag.EdgeTypeFlag.kParam
    kAny = _C.dag.EdgeTypeFlag.kAny
    kNone = _C.dag.EdgeTypeFlag.kNone

    @classmethod
    def from_name(cls, name: str):
        if name not in name_to_edge_type_flag:
            raise ValueError(f"Unsupported edge type flag: {name}")
        else:
            return cls(name_to_edge_type_flag[name])
        


class EdgeTypeInfo(_C.dag.EdgeTypeInfo):
    def __init__(self):
        super().__init__()

    @property
    def type_name(self) -> str:
        return self.type_name_
        
    @type_name.setter 
    def type_name(self, type_name: str):
        self.type_name_ = type_name
        
    @property
    def type(self) -> EdgeTypeFlag:
        return self.type_
        
    @type.setter
    def type(self, type_val: EdgeTypeFlag):
        self.type_ = type_val

    @property 
    def edge_name(self) -> str:
        return self.edge_name_
        
    @edge_name.setter
    def edge_name(self, edge_name: str):
        self.edge_name_ = edge_name

    def set_buffer_type(self):
        super().set_buffer_type()

    def set_cvmat_type(self):
        super().set_cvmat_type()

    def set_tensor_type(self):
        super().set_tensor_type()

    def set_param_type(self):
        super().set_param_type()
        
    def set_type(self, type_val: type):
        """设置输入输出类型信息
        
        Args:
            type: 类型
            
        Returns:
            None
        """
        super().set_type(type_val)
        

    def get_type(self) -> EdgeTypeFlag:
        return super().get_type()
    
    def set_type_name(self, type_name: str):
        super().set_type_name(type_name)

    def get_type_name(self) -> str:
        return super().get_type_name()

    def get_unique_type_name(self) -> str:
        return super().get_unique_type_name()

    def get_type_ptr(self):
        return super().get_type_ptr()

    def is_buffer_type(self) -> bool:
        return super().is_buffer_type()

    def is_cvmat_type(self) -> bool:
        return super().is_cvmat_type()

    def is_tensor_type(self) -> bool:
        return super().is_tensor_type()

    def is_param_type(self) -> bool:
        return super().is_param_type()

    def check_buffer_type(self) -> bool:
        return super().check_buffer_type()

    def check_cvmat_type(self) -> bool:
        return super().check_cvmat_type()

    def check_tensor_type(self) -> bool:
        return super().check_tensor_type()

    def check_param_type(self) -> bool:
        return super().check_param_type()

    def set_edge_name(self, edge_name: str):
        super().set_edge_name(edge_name)

    def get_edge_name(self) -> str:
        return super().get_edge_name()


def node_type_to_string(node_type) -> str:
    return _C.dag.node_type_to_string(node_type)

def string_to_node_type(node_type_str: str):
    return _C.dag.string_to_node_type(node_type_str)

def edge_type_to_string(edge_type) -> str:
    return _C.dag.edge_type_to_string(edge_type)

def string_to_edge_type(edge_type_str: str):
    return _C.dag.string_to_edge_type(edge_type_str)



