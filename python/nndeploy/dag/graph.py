import nndeploy._nndeploy_internal as _C
import nndeploy.base
import nndeploy.device
from typing import List, Union, Optional

from .util import *
from .base import EdgeTypeInfo
from .edge import Edge
from .node import Node, NodeDesc, NodeCreator, register_node


class Graph(_C.dag.Graph):
    def __init__(self, name: str, inputs: Union[Edge, List[Edge]] = None, outputs: Union[Edge, List[Edge]] = None):
        """
        初始化 Graph 对象
        
        参数:
            name: 图名称
            inputs: 输入边或输入边列表
            outputs: 输出边或输出边列表
        """
        if inputs is None and outputs is None:
            super().__init__(name)
        elif isinstance(inputs, list) and isinstance(outputs, list):
            super().__init__(name, inputs, outputs)
        else:
            raise ValueError("无效的输入或输出类型")
        self.set_key("nndeploy.dag.Graph")
        self.set_desc("Graph: Graph for nndeploy in python")
        self.nodes = []
        
    def __del__(self):
        if self.get_initialized():
            self.deinit()
            self.set_initialized_flag(False)
        # super().__del__()
        
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

    def set_output_type(self, output_type: type):
        edge_type_info = EdgeTypeInfo()
        edge_type_info.set_type(output_type)
        return self.set_output_type_info(edge_type_info)
    
    def set_edge_queue_max_size(self, queue_max_size: int):
        """设置边队列最大大小
        
        参数:
            queue_max_size: 队列最大大小
        """
        return super().set_edge_queue_max_size(queue_max_size)
        
    def get_edge_queue_max_size(self) -> int:
        """获取边队列最大大小
        
        返回:
            队列最大大小
        """
        return super().get_edge_queue_max_size()
        
    def set_input(self, input: Edge, index: int = -1):
        """设置输入边
        
        参数:
            input: 输入边
            index: 输入边索引,默认为-1
        """
        return super().set_input(input, index)
        
    def set_output(self, output: Edge, index: int = -1):
        """设置输出边
        
        参数:
            output: 输出边
            index: 输出边索引,默认为-1
        """
        return super().set_output(output, index)
        
    def set_inputs(self, inputs: List[Edge]):
        """设置输入边列表
        
        参数:
            inputs: 输入边列表
        """
        return super().set_inputs(inputs)
        
    def set_outputs(self, outputs: List[Edge]):
        """设置输出边列表
        
        参数:
            outputs: 输出边列表
        """
        return super().set_outputs(outputs)

    def create_edge(self, name: str) -> Edge:
        """
        创建一个边
        
        参数:
            name: 边名称
        返回:
            Edge 对象
        """
        return super().create_edge(name)

    def add_edge(self, edge: Edge):
        """
        添加一个边
        
        参数:
            edge: Edge 对象
        """
        return super().add_edge(edge)

    def update_edge(self, edge_wrapper: Edge, edge: Edge, is_external: bool = True):
        """
        更新边
        
        参数:
            edge_wrapper: 边包装器
            edge: 边对象
            is_external: 是否为外部边
        """
        return super().update_edge(edge_wrapper, edge, is_external)

    def get_edge(self, name: str) -> Edge:
        """
        通过名称获取边
        
        参数:
            name: 边名称
        返回:
            Edge 对象
        """
        return super().get_edge(name)

    def create_node(self, key_or_desc: Union[str, NodeDesc], name: str = "") -> Node:
        """
        创建节点
        
        参数:
            key_or_desc: 节点key或节点描述对象
            name: 节点名称,仅当key_or_desc为str时有效,默认为空字符串
        返回:
            Node 对象
        """
        if isinstance(key_or_desc, str):
            return super().create_node(key_or_desc, name)
        elif isinstance(key_or_desc, NodeDesc):
            return super().create_node(key_or_desc)
        else:
            raise ValueError("无效的节点描述对象")
        
    def set_node_desc(self, node: Node, desc: NodeDesc):
        """
        设置节点描述
        
        参数:
            node: 节点对象
            desc: 节点描述对象
            
        """
        return super().set_node_desc(node, desc)      
        
    def add_node(self, node: _C.dag.Node):
        """
        添加节点
        
        参数:
            node: Node 对象
        """
        return super().add_node(node)
    
    def get_node(self, name_or_index: Union[str, int]) -> Node:
        """
        通过名称获取节点
        
        参数:
            name: 节点名称
        返回:
            Node 对象
        """
        return super().get_node(name_or_index)
        
    def get_node_by_key(self, key: str) -> Node:
        """
        通过key获取节点
        
        参数:
            key: 节点key
        返回:
            Node 对象
        """
        return super().get_node_by_key(key)
        
    def get_nodes_by_key(self, key: str) -> List[Node]:
        """
        通过key获取所有匹配的节点
        
        参数:
            key: 节点key
        返回:
            Node对象列表
        """
        return super().get_nodes_by_key(key)
    
    def get_node_count(self) -> int:
        return super().get_node_count()
    
    def get_nodes(self) -> List[Node]:
        return super().get_nodes()

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
        返回:
            参数对象
        """
        return super().get_node_param(node_name)
    
    def set_external_param(self, key: str, param: nndeploy.base.Param):
        return super().set_external_param(key, param)
    
    def get_external_param(self, key: str) -> nndeploy.base.Param:
        return super().get_external_param(key)
    
    def set_node_parallel_type(self, node_name: str, parallel_type: nndeploy.base.ParallelType):
        return super().set_node_parallel_type(node_name, parallel_type)

    def set_graph_node_share_stream(self, flag: bool):
        """
        设置图节点流共享标志
        
        参数:
            flag: 标志值
        """
        return super().set_graph_node_share_stream(flag)

    def get_graph_node_share_stream(self) -> bool:
        """
        获取图节点流共享标志
        
        返回:
            流共享标志
        """
        return super().get_graph_node_share_stream()

    def update_node_io(self, node: Node, inputs: List[Edge], outputs: List[str]):
        """
        更新节点输入和输出
        
        参数:
            node: 节点对象
            inputs: 输入边列表
            outputs: 输出边名称列表
        """
        return super().update_node_io(node, inputs, outputs)
    
    def mark_input_edge(self, inputs: List[Edge]):
        return super().mark_input_edge(inputs)
    
    def mark_output_edge(self, outputs: List[Edge]):
        return super().mark_output_edge(outputs)
    
    def default_param(self):
        return super().default_param()

    def init(self):
        """初始化图"""
        return super().init()

    def deinit(self):
        """反初始化图"""
        return super().deinit()

    def run(self):
        """运行图"""
        return super().run()

    def synchronize(self):
        """同步图"""
        return super().synchronize()

    # def __call__(self, inputs):
    #     """
    #     调用图
        
    #     参数:
    #         inputs: 输入
    #     """
    #     return super().__call__(inputs)

    def dump(self):
        """输出图信息到标准输出"""
        return super().dump()
    
    def set_trace_flag(self, flag: bool):
        """
        设置追踪标志
        
        参数:
            flag: 追踪标志
        """
        return super().set_trace_flag(flag)
        
    def trace(self, inputs: Union[List[Edge], Edge, None] = None) -> List[Edge]:
        """
        追踪图的执行流程
        
        参数:
            inputs: 输入边列表、单个边或None。如果为None则使用默认输入
            
        返回:
            追踪后的边列表
        """
        if inputs is None:
            return super().trace()
        elif isinstance(inputs, Edge):
            return super().trace(inputs)
        elif isinstance(inputs, list):
            return super().trace(inputs)
        else:
            raise ValueError("inputs must be List[Edge], Edge or None")
        
    def to_static_graph(self):
        return super().to_static_graph()
    
    def get_edge_wrapper(self, edge: Union[Edge, str]) -> EdgeWrapper:
        return super().get_edge_wrapper(edge)
    
    def get_node_wrapper(self, node: Union[Node, str]) -> NodeWrapper:
        return super().get_node_wrapper(node)
        
    def serialize(self) -> str:
        return super().serialize()
    
    def save_file(self, path: str):
        return super().save_file(path)
        
    def deserialize(self, json_str: str):
        json_obj = json.loads(json_str)
        node_count = self.get_node_count()
        # 解析node_repository数组
        if "node_repository_" in json_obj:
            for i, node_json in enumerate(json_obj["node_repository_"]):
                # 创建节点描述对象
                # node_desc = NodeDesc()
                # node_desc.deserialize(json.dumps(node_json))
                # print(f"node_desc: {node_desc.get_name()}, {node_desc.get_key()}")
                node_key = node_json["key_"]
                
                node = None
                # 如果节点已存在则更新描述
                if node_count > i:
                    node = self.get_node(i)
                    # self.set_node_desc(node, node_desc)
                elif len(self.nodes) > i:
                    node = self.nodes[i]
                    # self.set_node_desc(node, node_desc)
                else:
                    # 否则创建新节点
                    node = self.create_node(node_key)
                    # print(f"create node: {node_key}, {node}")
                    if node is None:
                        raise RuntimeError("创建节点失败")
                    if node not in self.nodes:
                        print(f"add node: {node_key}, {node}")
                        self.nodes.append(node)
        return super().deserialize(json_str)
    
    def load_file(self, path: str):
        with open(path, "r") as f:
            json_str = f.read()
            self.deserialize(json_str)
    
    def __setattr__(self, name, value):
        """Override __setattr__ method to implement automatic node addition
        
        When assigning a Node type value to a Graph object attribute,
        automatically add that node to the graph
        
        Args:
            name: Attribute name
            value: Attribute value
        """
        # First call parent class's __setattr__ to set the attribute
        super().__setattr__(name, value)
        
        # Check if automatic node/edge addition should be enabled
        if self._should_auto_add_node(name, value):
            self._auto_add_node(value)
        elif self._should_auto_add_edge(name, value):
            self._auto_add_edge(value)
        else:
            pass

    def _should_auto_add_node(self, name: str, value) -> bool:
        """Determine if a node should be automatically added
        
        Args:
            name: Attribute name
            value: Attribute value
            
        Returns:
            bool: Whether the node should be automatically added
        """
        # Skip private and special attributes
        if name.startswith('_'):
            return False
            
        # Check if it's a Node type
        if not self._is_node_type(value):
            return False
        
        # Check if it has a node name
        if not hasattr(value, 'get_name') or not callable(getattr(value, 'get_name')):
            return False
            
        return True
    
    def _is_node_type(self, value) -> bool:
        """Check if an object is of Node type
        
        Args:
            value: Object to check
            
        Returns:
            bool: Whether it's a Node type
        """
        try:
            # Check if it's an instance of Node or its subclasses
            return isinstance(value, (_C.dag.Node, Node))
        except:
            # Return False if any exception occurs during check
            return False
    
    def _auto_add_node(self, node: Union[_C.dag.Node, Node]):
        """Automatically add a node to the graph
        
        Args:
            node: Node to add
        """
        try:
            node_name = node.get_name()
            
            # Check if node already exists in graph to avoid duplicates
            existing_node = None
            try:
                existing_node = self.get_node_wrapper(node)
            except:
                pass
            
            if existing_node is None:
                # Add node to graph
                self.add_node(node)
                # print(f"Automatically added node '{node_name}' to graph '{self.get_name()}'")
            else:
                # print(f"Node '{node_name}' already exists in graph, skipping addition")
                pass
                
        except Exception as e:
            # Print warning if error occurs during addition but continue execution
            print(f"Exception during automatic node addition: {str(e)}")
            
    def _should_auto_add_edge(self, name: str, value) -> bool:
        """Determine if an edge should be automatically added
        
        Args:
            name: Attribute name
            value: Attribute value
            
        Returns:
            bool: Whether the edge should be automatically added
        """
        # Skip private and special attributes
        if name.startswith('_'):
            return False
            
        # Check if it's an Edge type
        if not self._is_edge_type(value):
            return False
        
        # Check if it has an edge name
        if not hasattr(value, 'get_name') or not callable(getattr(value, 'get_name')):
            return False
            
        return True
    
    def _is_edge_type(self, value) -> bool:
        """Check if an object is of Edge type
        
        Args:
            value: Object to check
            
        Returns:
            bool: Whether it's an Edge type
        """
        try:
            # Check if it's an instance of Edge or its subclasses
            return isinstance(value, (_C.dag.Edge, Edge))
        except:
            # Return False if any exception occurs during check
            return False
    
    def _auto_add_edge(self, edge: Union[_C.dag.Edge, Edge]):
        """Automatically add an edge to the graph
        
        Args:
            edge: Edge to add
        """
        try:
            edge_name = edge.get_name()
            
            # Check if edge already exists in graph to avoid duplicates
            existing_edge = None
            try:
                existing_edge = self.get_edge_wrapper(edge)
            except:
                pass
            
            if existing_edge is None:
                # Add edge to graph
                self.add_edge(edge)
                # print(f"Automatically added edge '{edge_name}' to graph '{self.get_name()}'")
            else:
                # print(f"Edge '{edge_name}' already exists in graph, skipping addition")
                pass
                
        except Exception as e:
            # Print warning if error occurs during addition but continue execution
            print(f"Exception during automatic edge addition: {str(e)}")

def serialize(graph: Graph) -> str:
    return graph.serialize()

def save_file(graph: Graph, path: str):
    """
    保存图到文件
    
    参数:
        path: 文件路径
    """
    return graph.save_file(path)

def deserialize(json_str: str) -> Graph:
    graph = Graph("")
    graph.deserialize(json_str)
    return graph
   
def load_file(path: str) -> Graph:
    """
    从文件加载图
    
    参数:
        path: 文件路径
    """
    graph = Graph("")
    graph.load_file(path)
    return graph

class GraphCreator(NodeCreator):
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[Edge], outputs: list[Edge]):
        self.node = Graph(name, inputs, outputs)
        return self.node
      
graph_node_creator = GraphCreator()
register_node("nndeploy.dag.Graph", graph_node_creator)   

