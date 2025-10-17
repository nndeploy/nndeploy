import nndeploy._nndeploy_internal as _C
import nndeploy.base
import nndeploy.device
from typing import List, Union, Optional

from .util import *
from .base import EdgeTypeInfo
from .edge import Edge
from .node import Node, NodeDesc

class CompositeNode(_C.dag.CompositeNode):
    def __init__(self, name: str, inputs=None, outputs=None):
        if inputs is None and outputs is None:
            super().__init__(name)
        elif isinstance(inputs, list) and isinstance(outputs, list):
            super().__init__(name, inputs, outputs)
        else:
            super().__init__(name, inputs, outputs)
            
    def set_input_type(self, input_type: type, desc: str = ""):
        edge_type_info = EdgeTypeInfo()
        edge_type_info.set_type(input_type)
        return _C.dag.Node.set_input_type_info(self, edge_type_info, desc)  
            
    def set_output_type(self, output_type: type, desc: str = ""):
        edge_type_info = EdgeTypeInfo()
        edge_type_info.set_type(output_type)
        return _C.dag.Node.set_output_type_info(self, edge_type_info, desc)
            
    def set_input(self, input, index=-1):
        return super().set_input(input, index)
        
    def set_output(self, output, index=-1):
        return super().set_output(output, index)
        
    def set_inputs(self, inputs):
        return super().set_inputs(inputs)
        
    def set_outputs(self, outputs):
        return super().set_outputs(outputs)
        
    def create_edge(self, name):
        return super().create_edge(name)
        
    def add_edge(self, edge, is_external=True):
        return super().add_edge(edge, is_external)
        
    def get_edge(self, name):
        return super().get_edge(name)
        
    def update_edge(self, edge_wrapper, edge, is_external=True):
        return super().update_edge(edge_wrapper, edge, is_external)
        
    def create_node(self, desc):
        return super().create_node(desc)
        
    def set_node_desc(self, node, desc):
        return super().set_node_desc(node, desc)
        
    def add_node(self, node):
        return super().add_node(node)
        
    def update_node_io(self, node, inputs, outputs):
        return super().update_node_io(node, inputs, outputs)
        
    def mark_input_edge(self, inputs):
        return super().mark_input_edge(inputs)
        
    def mark_output_edge(self, outputs):
        return super().mark_output_edge(outputs)
        
    def get_node(self, name):
        return super().get_node(name)
        
    def get_node_by_key(self, key):
        return super().get_node_by_key(key)
        
    def get_nodes_by_key(self, key):
        return super().get_nodes_by_key(key)
        
    def set_node_param(self, node_name, param):
        return super().set_node_param(node_name, param)
        
    def get_node_param(self, node_name):
        return super().get_node_param(node_name)
        
    def default_param(self):
        return super().default_param()
        
    def init(self):
        return super().init()
        
    def deinit(self):
        return super().deinit()
        
    def run(self):
        print("must be override")
        
    def get_edge_wrapper(self, edge_or_name):
        return super().get_edge_wrapper(edge_or_name)
        
    def get_node_wrapper(self, node_or_name):
        return super().get_node_wrapper(node_or_name)
        
    def serialize(self, json=None, allocator=None):
        if json is not None and allocator is not None:
            return super().serialize(json, allocator)
        return super().serialize()
        
    def deserialize(self, json_or_str):
        return super().deserialize(json_or_str)
        
    def sort_dfs(self):
        return super().sort_dfs()
    
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

