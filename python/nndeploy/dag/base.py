import nndeploy._nndeploy_internal as _C

from enum import Enum
from typing import Union
import numpy as np
import json

import nndeploy.base
import nndeploy.device


name_to_node_type = {
    "kInput": _C.dag.NodeType.kNodeTypeInput,
    "kOutput": _C.dag.NodeType.kNodeTypeOutput,
    "kIntermediate": _C.dag.NodeType.kNodeTypeIntermediate,
}


node_type_to_name = {v: k for k, v in name_to_node_type.items()}


class NodeType(_C.dag.NodeType):
    kInput = _C.dag.NodeType.kNodeTypeInput
    kOutput = _C.dag.NodeType.kNodeTypeOutput
    kIntermediate = _C.dag.NodeType.kNodeTypeIntermediate

    @classmethod
    def from_name(cls, name: str):
        name_to_node_type = {
            "kNodeTypeInput": cls.kNodeTypeInput,
            "kNodeTypeOutput": cls.kNodeTypeOutput,
            "kNodeTypeIntermediate": cls.kNodeTypeIntermediate
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

  

