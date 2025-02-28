import nndeploy._nndeploy_internal as _C
import nndeploy.base
import nndeploy.device
from typing import List, Union, Optional
from .edge import Edge
from .node import Node

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
        elif isinstance(inputs, Edge) and isinstance(outputs, Edge):
            super().__init__(name, inputs, outputs)
        elif isinstance(inputs, list) and isinstance(outputs, list):
            super().__init__(name, inputs, outputs)
        else:
            raise ValueError("Invalid inputs or outputs type")

    def create_edge(self, name: str) -> Edge:
        """
        Create an edge
        
        Args:
            name: Edge name
        Returns:
            Edge object with reference policy
        """
        return super().create_edge(name)

    def add_edge(self, edge: Edge):
        """
        Add an edge with keep_alive policy
        
        Args:
            edge: Edge object
        """
        return super().add_edge(edge)

    def remove_edge(self, edge: Edge):
        """
        Remove an edge
        
        Args:
            edge: Edge object
        """
        return super().remove_edge(edge)

    def get_edge(self, name: str) -> Edge:
        """
        Get an edge by name with reference policy
        
        Args:
            name: Edge name
        Returns:
            Edge object
        """
        return super().get_edge(name)

    def create_node(self, desc: dict) -> Node:
        """
        Create a node by key with reference policy
        
        Args:
            desc: Node description dictionary
        Returns:
            Node object
        """
        return super().create_node(desc)

    def add_node(self, node: Node):
        """
        Add a node with keep_alive policy
        
        Args:
            node: Node object
        """
        return super().add_node(node)

    def set_node_param(self, node_name: str, param: nndeploy.base.Param):
        """
        Set node shared pointer parameters
        
        Args:
            node_name: Node name
            param: Parameter object
        """
        return super().set_node_param(node_name, param)

    def get_node_param(self, node_name: str) -> nndeploy.base.Param:
        """
        Get node shared pointer parameters
        
        Args:
            node_name: Node name
        Returns:
            Parameter object
        """
        return super().get_node_param(node_name)

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
            bool: Stream sharing flag
        """
        return super().get_graph_node_share_stream()

    def update_node_io(self, node: Node, inputs: List[Edge], outputs_name: List[str], param: Optional[nndeploy.base.Param] = None) -> List[Edge]:
        """
        Update node inputs and outputs with overloaded versions
        
        Args:
            node: Node object
            inputs: List of input edges
            outputs_name: List of output edge names
            param: Optional parameter object
        Returns:
            List of Edge objects
        """
        if param is None:
            return super().update_node_io(node, inputs, outputs_name)
        else:
            return super().update_node_io(node, inputs, outputs_name, param)

    def init(self):
        """Initialize graph"""
        return super().init()

    def deinit(self):
        """Deinitialize graph"""
        return super().deinit()

    def run(self):
        """Run graph"""
        return super().run()

    def dump(self):
        """Dump graph information to stdout"""
        return super().dump()
