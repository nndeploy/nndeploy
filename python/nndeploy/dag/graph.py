import nndeploy._nndeploy_internal as _C
import nndeploy.base
import nndeploy.device
from typing import List, Union, Optional
from .edge import Edge
from .node import Node

class Graph(_C.dag.Graph):
    def __init__(self, name: str, inputs: Union[Edge, List[Edge]] = None, outputs: Union[Edge, List[Edge]] = None):
        """
        初始化Graph对象
        
        参数:
            name: 图名称
            inputs: 输入边或输入边列表
            outputs: 输出边或输出边列表
        """
        if inputs is None and outputs is None:
            super().__init__(name)
        elif isinstance(inputs, Edge) and isinstance(outputs, Edge):
            super().__init__(name, inputs, outputs)
        elif isinstance(inputs, list) and isinstance(outputs, list):
            super().__init__(name, inputs, outputs)
        else:
            raise ValueError("Invalid inputs or outputs type")

    def create_edge(self, name: str) -> Edge:
        """
        创建边
        
        参数:
            name: 边名称
        """
        return super().create_edge(name)

    def create_edge_shared_ptr(self, name: str) -> Edge:
        """
        创建边共享指针
        
        参数:
            name: 边名称
        """
        return super().create_edge_shared_ptr(name)

    def add_edge(self, edge: Edge, is_external: bool = True):
        """
        添加边
        
        参数:
            edge: 边对象
            is_external: 是否为外部边
        """
        return super().add_edge(edge, is_external)

    def add_edge_shared_ptr(self, edge: Edge):
        """
        添加边共享指针
        
        参数:
            edge: 边对象
        """
        return super().add_edge_shared_ptr(edge)

    def remove_edge(self, edge: Edge):
        """
        移除边
        
        参数:
            edge: 边对象
        """
        return super().remove_edge(edge)

    def get_edge(self, name: str) -> Edge:
        """
        获取边
        
        参数:
            name: 边名称
        """
        return super().get_edge(name)

    def get_edge_shared_ptr(self, name: str) -> Edge:
        """
        获取边共享指针
        
        参数:
            name: 边名称
        """
        return super().get_edge_shared_ptr(name)

    def create_node(self, desc: dict) -> Node:
        """
        创建节点
        
        参数:
            desc: 节点描述字典
        """
        return super().create_node(desc)

    def add_node(self, node: Node, is_external: bool = True):
        """
        添加节点
        
        参数:
            node: 节点对象
            is_external: 是否为外部节点
        """
        return super().add_node(node, is_external)

    def add_node_shared_ptr(self, node: Node):
        """
        添加节点共享指针
        
        参数:
            node: 节点对象
        """
        return super().add_node_shared_ptr(node)

    def set_node_param(self, node_name: str, param: nndeploy.base.Param):
        """
        设置节点参数
        
        参数:
            node_name: 节点名称
            param: 参数对象
        """
        return super().set_node_param(node_name, param)

    def get_node_param(self, node_name: str) -> nndeploy.base.Param:
        """
        获取节点参数
        
        参数:
            node_name: 节点名称
        """
        return super().get_node_param(node_name)

    def set_graph_node_share_stream(self, flag: bool):
        """
        设置图节点共享流标志
        
        参数:
            flag: 标志值
        """
        return super().set_graph_node_share_stream(flag)

    def get_graph_node_share_stream(self) -> bool:
        """获取图节点共享流标志"""
        return super().get_graph_node_share_stream()

    def update_node_io(self, node: Node, inputs: List[Edge], outputs_name: List[str]) -> List[Edge]:
        """
        更新节点输入输出
        
        参数:
            node: 节点对象
            inputs: 输入边列表
            outputs_name: 输出边名称列表
        """
        return super().update_node_io(node, inputs, outputs_name)

    def init(self):
        """初始化图"""
        return super().init()

    def deinit(self):
        """反初始化图"""
        return super().deinit()

    def run(self):
        """运行图"""
        return super().run()

    def dump(self):
        """导出图信息"""
        return super().dump()
