import nndeploy._nndeploy_internal as _C

from enum import Enum
from typing import Union, List
import numpy as np
import json

import nndeploy.base
import nndeploy.device
from nndeploy.dag.node import Node
from nndeploy.dag.edge import Edge

class NodeWrapper(_C.dag.NodeWrapper):
    def __init__(self):
        super().__init__()
        
    @property
    def is_external(self) -> bool:
        return self.is_external_
        
    @is_external.setter 
    def is_external(self, value: bool):
        self.is_external_ = value
        
    @property
    def node(self) -> Node:
        return self.node_
        
    @node.setter
    def node(self, value: Node):
        self.node_ = value
        
    @property
    def name(self) -> str:
        return self.name_
        
    @name.setter
    def name(self, value: str):
        self.name_ = value
        
    @property
    def predecessors(self) -> List['NodeWrapper']:
        return self.predecessors_
        
    @predecessors.setter
    def predecessors(self, value: List['NodeWrapper']):
        self.predecessors_ = value
        
    @property
    def successors(self) -> List['NodeWrapper']:
        return self.successors_
        
    @successors.setter
    def successors(self, value: List['NodeWrapper']):
        self.successors_ = value
        
    @property
    def color(self) -> nndeploy.base.NodeColorType:
        return self.color_
        
    @color.setter
    def color(self, value: nndeploy.base.NodeColorType):
        self.color_ = value
        
class EdgeWrapper(_C.dag.EdgeWrapper):
    def __init__(self):
        super().__init__()
        
    @property
    def is_external(self) -> bool:
        return self.is_external_
        
    @is_external.setter
    def is_external(self, value: bool):
        self.is_external_ = value
        
    @property
    def edge(self) -> Edge:
        return self.edge_
        
    @edge.setter
    def edge(self, value: Edge):
        self.edge_ = value
        
    @property
    def name(self) -> str:
        return self.name_
        
    @name.setter
    def name(self, value: str):
        self.name_ = value
        
    @property
    def producers(self) -> List[NodeWrapper]:
        return self.producers_
        
    @producers.setter
    def producers(self, value: List[NodeWrapper]):
        self.producers_ = value
        
    @property
    def consumers(self) -> List[NodeWrapper]:
        return self.consumers_
        
    @consumers.setter
    def consumers(self, value: List[NodeWrapper]):
        self.consumers_ = value

def get_edge(edge_repository: List[EdgeWrapper], edge_name: str) -> Edge:
    return _C.dag.get_edge(edge_repository, edge_name)

def find_edge_wrapper(edge_repository: List[EdgeWrapper], edge_name_or_edge: Union[str, Edge]) -> EdgeWrapper:
    if isinstance(edge_name_or_edge, str):
        return _C.dag.find_edge_wrapper(edge_repository, edge_name_or_edge)
    else:
        return _C.dag.find_edge_wrapper(edge_repository, edge_name_or_edge)

def find_start_edges(edge_repository: List[EdgeWrapper]) -> List[EdgeWrapper]:
    return _C.dag.find_start_edges(edge_repository)

def find_end_edges(edge_repository: List[EdgeWrapper]) -> List[EdgeWrapper]:
    return _C.dag.find_end_edges(edge_repository)

def get_node(node_repository: List[NodeWrapper], node_name: str) -> Node:
    return _C.dag.get_node(node_repository, node_name)

def find_node_wrapper(node_repository: List[NodeWrapper], node_name_or_node: Union[str, Node]) -> NodeWrapper:
    if isinstance(node_name_or_node, str):
        return _C.dag.find_node_wrapper(node_repository, node_name_or_node)
    else:
        return _C.dag.find_node_wrapper(node_repository, node_name_or_node)

def find_start_nodes(node_repository: List[NodeWrapper]) -> List[NodeWrapper]:
    return _C.dag.find_start_nodes(node_repository)

def find_end_nodes(node_repository: List[NodeWrapper]) -> List[NodeWrapper]:
    return _C.dag.find_end_nodes(node_repository)

def set_color(node_repository: List[NodeWrapper], color: nndeploy.base.NodeColorType) -> nndeploy.base.Status:
    return _C.dag.set_color(node_repository, color)

def dump_dag(edge_repository: List[EdgeWrapper], node_repository: List[NodeWrapper], 
             graph_inputs: List[Edge], graph_outputs: List[Edge], name: str, oss: str) -> nndeploy.base.Status:
    return _C.dag.dump_dag(edge_repository, node_repository, graph_inputs, graph_outputs, name, oss)

def check_unuse_node(node_repository: List[NodeWrapper]) -> List[NodeWrapper]:
    return _C.dag.check_unuse_node(node_repository)

def check_unuse_edge(node_repository: List[NodeWrapper], edge_repository: List[EdgeWrapper]) -> List[EdgeWrapper]:
    return _C.dag.check_unuse_edge(node_repository, edge_repository)

def topo_sort_bfs(node_repository: List[NodeWrapper], topo_sort_node: List[NodeWrapper]) -> nndeploy.base.Status:
    return _C.dag.topo_sort_bfs(node_repository, topo_sort_node)

def topo_sort_dfs(node_repository: List[NodeWrapper], topo_sort_node: List[NodeWrapper]) -> nndeploy.base.Status:
    return _C.dag.topo_sort_dfs(node_repository, topo_sort_node)

def topo_sort(node_repository: List[NodeWrapper], topo_sort_type: nndeploy.base.TopoSortType, 
              topo_sort_node: List[NodeWrapper]) -> nndeploy.base.Status:
    return _C.dag.topo_sort(node_repository, topo_sort_type, topo_sort_node)

def check_edge(src_edges: List[Edge], dst_edges: List[Edge]) -> bool:
    return _C.dag.check_edge(src_edges, dst_edges)
