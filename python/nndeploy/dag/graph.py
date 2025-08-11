import nndeploy._nndeploy_internal as _C
import nndeploy.base
import nndeploy.device
from typing import List, Union, Optional

from .util import *
from .base import EdgeTypeInfo
from .edge import Edge, get_accepted_edge_type_json, get_accepted_edge_type_map
from .node import Node, NodeDesc, NodeCreator, register_node, get_all_node_json


class Graph(_C.dag.Graph):
    def __init__(self, name: str, inputs: Union[Edge, List[Edge]] = None, outputs: Union[Edge, List[Edge]] = None):
        """
        Initialize Graph object
        
        Args:
            name: Graph name
            inputs: Input edge or list of input edges
            outputs: Output edge or list of output edges
        """
        if inputs is None and outputs is None:
            super().__init__(name)
        elif isinstance(inputs, list) and isinstance(outputs, list):
            super().__init__(name, inputs, outputs)
        else:
            raise ValueError("Invalid input or output type")
        self.set_key("nndeploy.dag.Graph")
        self.set_desc("Graph: Graph for nndeploy in python")
        self.nodes = []
        
    def __del__(self):
        """Destructor to clean up graph resources"""
        if self.get_initialized():
            self.deinit()
            self.set_initialized_flag(False)
        # super().__del__()
        
    def add_image_url(self, url: str):
        """添加图像URL
        
        Args:
            url: URL地址
            
        Returns:
            状态码
        """
        return super().add_image_url(url)
        
    def remove_image_url(self, url: str):
        """移除图像URL
        
        Args:
            url: URL地址
            
        Returns:
            状态码
        """
        return super().remove_image_url(url)
        
    def add_video_url(self, url: str):
        """添加视频URL
        
        Args:
            url: URL地址
            
        Returns:
            状态码
        """
        return super().add_video_url(url)
        
    def remove_video_url(self, url: str):
        """移除视频URL
        
        Args:
            url: URL地址
            
        Returns:
            状态码
        """
        return super().remove_video_url(url)
        
    def add_audio_url(self, url: str):
        """添加音频URL
        
        Args:
            url: URL地址
            
        Returns:
            状态码
        """
        return super().add_audio_url(url)
        
    def remove_audio_url(self, url: str):
        """移除音频URL
        
        Args:
            url: URL地址
            
        Returns:
            状态码
        """
        return super().remove_audio_url(url)
        
    def add_model_url(self, url: str):
        """添加模型URL
        
        Args:
            url: URL地址
            
        Returns:
            状态码
        """
        return super().add_model_url(url)
        
    def remove_model_url(self, url: str):
        """移除模型URL
        
        Args:
            url: URL地址
            
        Returns:
            状态码
        """
        return super().remove_model_url(url)
        
    def add_other_url(self, url: str):
        """添加其他URL
        
        Args:
            url: URL地址
            
        Returns:
            状态码
        """
        return super().add_other_url(url)
        
    def remove_other_url(self, url: str):
        """移除其他URL
        
        Args:
            url: URL地址
            
        Returns:
            状态码
        """
        return super().remove_other_url(url)
        
    def get_image_url(self):
        """获取图像URL列表
        
        Returns:
            URL地址列表
        """
        return super().get_image_url()
        
    def get_video_url(self):
        """获取视频URL列表
        
        Returns:
            URL地址列表
        """
        return super().get_video_url()
        
    def get_audio_url(self):
        """获取音频URL列表
        
        Returns:
            URL地址列表
        """
        return super().get_audio_url()
        
    def get_model_url(self):
        """获取模型URL列表
        
        Returns:
            URL地址列表
        """
        return super().get_model_url()
        
    def get_other_url(self):
        """获取其他URL列表
        
        Returns:
            URL地址列表
        """
        return super().get_other_url()
        
    def set_input_type(self, input_type: type, desc: str = ""):
        """Set input type
        
        Args:
            input_type: Input type
            
        Returns:
            Status code
        """
        edge_type_info = EdgeTypeInfo()
        edge_type_info.set_type(input_type)
        return self.set_input_type_info(edge_type_info, desc)

    def set_output_type(self, output_type: type, desc: str = ""):
        """Set output type
        
        Args:
            output_type: Output type
            
        Returns:
            Status code
        """
        edge_type_info = EdgeTypeInfo()
        edge_type_info.set_type(output_type)
        return self.set_output_type_info(edge_type_info, desc)
    
    def set_edge_queue_max_size(self, queue_max_size: int):
        """Set edge queue maximum size
        
        Args:
            queue_max_size: Maximum queue size
        """
        return super().set_edge_queue_max_size(queue_max_size)
        
    def get_edge_queue_max_size(self) -> int:
        """Get edge queue maximum size
        
        Returns:
            Maximum queue size
        """
        return super().get_edge_queue_max_size()
        
    def set_input(self, input: Edge, index: int = -1):
        """Set input edge
        
        Args:
            input: Input edge
            index: Input edge index, default is -1
        """
        return super().set_input(input, index)
        
    def set_output(self, output: Edge, index: int = -1):
        """Set output edge
        
        Args:
            output: Output edge
            index: Output edge index, default is -1
        """
        return super().set_output(output, index)
        
    def set_inputs(self, inputs: List[Edge]):
        """Set input edge list
        
        Args:
            inputs: List of input edges
        """
        return super().set_inputs(inputs)
        
    def set_outputs(self, outputs: List[Edge]):
        """Set output edge list
        
        Args:
            outputs: List of output edges
        """
        return super().set_outputs(outputs)

    def create_edge(self, name: str) -> Edge:
        """
        Create an edge
        
        Args:
            name: Edge name
        Returns:
            Edge object
        """
        return super().create_edge(name)

    def add_edge(self, edge: Edge):
        """
        Add an edge
        
        Args:
            edge: Edge object
        """
        return super().add_edge(edge)

    def update_edge(self, edge_wrapper: Edge, edge: Edge, is_external: bool = True):
        """
        Update edge
        
        Args:
            edge_wrapper: Edge wrapper
            edge: Edge object
            is_external: Whether it's an external edge
        """
        return super().update_edge(edge_wrapper, edge, is_external)

    def get_edge(self, name: str) -> Edge:
        """
        Get edge by name
        
        Args:
            name: Edge name
        Returns:
            Edge object
        """
        return super().get_edge(name)

    def create_node(self, key_or_desc: Union[str, NodeDesc], name: str = "") -> Node:
        """
        Create node
        
        Args:
            key_or_desc: Node key or node description object
            name: Node name, only valid when key_or_desc is str, default is empty string
        Returns:
            Node object
        """
        if isinstance(key_or_desc, str):
            return super().create_node(key_or_desc, name)
        elif isinstance(key_or_desc, NodeDesc):
            return super().create_node(key_or_desc)
        else:
            raise ValueError("Invalid node description object")
        
    def set_node_desc(self, node: Node, desc: NodeDesc):
        """
        Set node description
        
        Args:
            node: Node object
            desc: Node description object
            
        """
        return super().set_node_desc(node, desc)      
        
    def add_node(self, node: _C.dag.Node):
        """
        Add node
        
        Args:
            node: Node object
        """
        return super().add_node(node)
    
    def get_node(self, name_or_index: Union[str, int]) -> Node:
        """
        Get node by name or index
        
        Args:
            name_or_index: Node name or index
        Returns:
            Node object
        """
        return super().get_node(name_or_index)
        
    def get_node_by_key(self, key: str) -> Node:
        """
        Get node by key
        
        Args:
            key: Node key
        Returns:
            Node object
        """
        return super().get_node_by_key(key)
        
    def get_nodes_by_key(self, key: str) -> List[Node]:
        """
        Get all matching nodes by key
        
        Args:
            key: Node key
        Returns:
            List of Node objects
        """
        return super().get_nodes_by_key(key)
    
    def get_node_count(self) -> int:
        """Get node count in the graph
        
        Returns:
            Number of nodes
        """
        return super().get_node_count()
    
    def get_nodes(self) -> List[Node]:
        """Get all nodes in the graph
        
        Returns:
            List of all nodes
        """
        return super().get_nodes()

    def set_node_param(self, node_name: str, param: nndeploy.base.Param):
        """
        Set node parameter
        
        Args:
            node_name: Node name
            param: Parameter object
        """
        return super().set_node_param(node_name, param)

    def get_node_param(self, node_name: str) -> nndeploy.base.Param:
        """
        Get node parameter
        
        Args:
            node_name: Node name
        Returns:
            Parameter object
        """
        return super().get_node_param(node_name)
    
    def set_external_param(self, key: str, param: nndeploy.base.Param):
        """Set external parameter
        
        Args:
            key: Parameter key
            param: Parameter object
        """
        return super().set_external_param(key, param)
    
    def get_external_param(self, key: str) -> nndeploy.base.Param:
        """Get external parameter
        
        Args:
            key: Parameter key
        Returns:
            Parameter object
        """
        return super().get_external_param(key)
    
    def set_node_parallel_type(self, node_name: str, parallel_type: nndeploy.base.ParallelType):
        """Set node parallel type
        
        Args:
            node_name: Node name
            parallel_type: Parallel type
        """
        return super().set_node_parallel_type(node_name, parallel_type)

    def set_graph_node_share_stream(self, flag: bool):
        """
        Set graph node stream sharing flag
        
        Args:
            flag: Flag value
        """
        return super().set_graph_node_share_stream(flag)

    def get_graph_node_share_stream(self) -> bool:
        """
        Get graph node stream sharing flag
        
        Returns:
            Stream sharing flag
        """
        return super().get_graph_node_share_stream()

    def update_node_io(self, node: Node, inputs: List[Edge], outputs: List[str]):
        """
        Update node inputs and outputs
        
        Args:
            node: Node object
            inputs: List of input edges
            outputs: List of output edge names
        """
        return super().update_node_io(node, inputs, outputs)
    
    def mark_input_edge(self, inputs: List[Edge]):
        """Mark edges as input edges
        
        Args:
            inputs: List of input edges
        """
        return super().mark_input_edge(inputs)
    
    def mark_output_edge(self, outputs: List[Edge]):
        """Mark edges as output edges
        
        Args:
            outputs: List of output edges
        """
        return super().mark_output_edge(outputs)
    
    def default_param(self):
        """Initialize graph with default parameters
        
        Returns:
            Status code
        """
        return super().default_param()

    def init(self):
        """Initialize graph"""
        return super().init()

    def deinit(self):
        """Deinitialize graph"""
        return super().deinit()

    def run(self):
        """Run graph execution"""
        return super().run()

    def synchronize(self):
        """Synchronize graph execution"""
        return super().synchronize()

    # def __call__(self, inputs):
    #     """
    #     Call graph
        
    #     Args:
    #         inputs: Input data
    #     """
    #     return super().__call__(inputs)

    def dump(self):
        """Dump graph information to standard output"""
        return super().dump()
    
    def set_trace_flag(self, flag: bool):
        """
        Set trace flag
        
        Args:
            flag: Trace flag
        """
        return super().set_trace_flag(flag)
        
    def trace(self, inputs: Union[List[Edge], Edge, None] = None) -> List[Edge]:
        """
        Trace graph execution flow
        
        Args:
            inputs: List of input edges, single edge or None. If None, use default inputs
            
        Returns:
            List of traced edges
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
        """Convert to static graph representation
        
        Returns:
            Static graph object
        """
        return super().to_static_graph()
    
    def get_edge_wrapper(self, edge: Union[Edge, str]) -> EdgeWrapper:
        """Get edge wrapper
        
        Args:
            edge: Edge object or edge name
        Returns:
            Edge wrapper object
        """
        return super().get_edge_wrapper(edge)
    
    def get_node_wrapper(self, node: Union[Node, str]) -> NodeWrapper:
        """Get node wrapper
        
        Args:
            node: Node object or node name
        Returns:
            Node wrapper object
        """
        return super().get_node_wrapper(node)
        
    def serialize(self) -> str:
        """Serialize graph to JSON string
        
        Returns:
            JSON string representation of the graph
        """
        return super().serialize()
    
    def save_file(self, path: str):
        """Save graph to file
        
        Args:
            path: File path
        """
        return super().save_file(path)
        
    def deserialize(self, json_str: str):
        """Deserialize graph from JSON string
        
        Args:
            json_str: JSON string to deserialize
        """
        json_obj = json.loads(json_str)
        node_count = self.get_node_count()
        # Parse node_repository array
        if "node_repository_" in json_obj:
            for i, node_json in enumerate(json_obj["node_repository_"]):
                # Create node description object
                # node_desc = NodeDesc()
                # node_desc.deserialize(json.dumps(node_json))
                # print(f"node_desc: {node_desc.get_name()}, {node_desc.get_key()}")
                node_key = node_json["key_"]
                
                node = None
                # Update description if node already exists
                if node_count > i:
                    node = self.get_node(i)
                    # self.set_node_desc(node, node_desc)
                elif len(self.nodes) > i:
                    node = self.nodes[i]
                    # self.set_node_desc(node, node_desc)
                else:
                    # Otherwise create new node
                    node = self.create_node(node_key)
                    if node is None:
                        print(f"create node: {node_key}, {node}")
                        print(f"node_count: {node_count}, i: {i}")
                        raise RuntimeError("Failed to create node")
                    if node not in self.nodes:
                        # print(f"add node: {node_key}, {node}")
                        self.nodes.append(node)
        return super().deserialize(json_str)
    
    def load_file(self, path: str):
        """Load graph from file
        
        Args:
            path: File path
        """
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
    """Serialize graph to JSON string
    
    Args:
        graph: Graph object to serialize
    Returns:
        JSON string representation
    """
    return graph.serialize()

def save_file(graph: Graph, path: str):
    """
    Save graph to file
    
    Args:
        graph: Graph object to save
        path: File path
    """
    return graph.save_file(path)

def deserialize(json_str: str) -> Graph:
    """Deserialize graph from JSON string
    
    Args:
        json_str: JSON string to deserialize
    Returns:
        Graph object
    """
    graph = Graph("")
    graph.deserialize(json_str)
    return graph
   
def load_file(path: str) -> Graph:
    """
    Load graph from file
    
    Args:
        path: File path
    Returns:
        Graph object
    """
    graph = Graph("")
    graph.load_file(path)
    return graph

class GraphCreator(NodeCreator):
    """Graph creator for creating graph nodes"""
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[Edge], outputs: list[Edge]):
        """Create a graph node
        
        Args:
            name: Node name
            inputs: Input edges
            outputs: Output edges
        Returns:
            Graph node
        """
        self.node = Graph(name, inputs, outputs)
        return self.node
      
# Register graph creator
graph_node_creator = GraphCreator()
register_node("nndeploy.dag.Graph", graph_node_creator)   


def get_graph_json():
    """Get graph JSON representation
    
    Returns:
        JSON string containing graph information
    """
    graph = Graph("Graph")
    status = graph.default_param()
    if status != nndeploy.base.StatusCode.Ok:
        raise RuntimeError(f"graph default_param failed: {status}")
    graph.set_inner_flag(False)
    json_str = graph.serialize()
    graph_json = "{\"graph\":" + json_str + "}"
    # Beautify JSON
    graph_json = nndeploy.base.pretty_json_str(graph_json)
    return graph_json


def get_dag_json():
    """Get DAG JSON representation containing graph, nodes and edge types
    
    Returns:
        JSON string containing complete DAG information
    """
    graph_json = get_graph_json()
    node_json = get_all_node_json()
    # edge_json = get_accepted_edge_type_json()
    
    # Parse JSON strings and extract content
    import json
    graph_data = json.loads(graph_json)
    node_data = json.loads(node_json)
    # edge_data = json.loads(edge_json)
    
    # Merge into specified JSON format
    dag_data = {
        "graph": graph_data["graph"],
        "nodes": node_data["nodes"],
        "accepted_edge_types": get_accepted_edge_type_map()
    }
    
    dag_json = json.dumps(dag_data, ensure_ascii=False, indent=2)
    
    return dag_json
