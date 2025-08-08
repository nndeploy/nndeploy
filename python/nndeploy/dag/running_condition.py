import nndeploy._nndeploy_internal as _C
import nndeploy.base
import nndeploy.device
from typing import List, Union, Optional

from .util import *
from .base import EdgeTypeInfo
from .edge import Edge
from .node import Node, NodeDesc
from .graph import Graph

class RunningCondition(_C.dag.RunningCondition):
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
    
    def choose(self):
        print("must be override")
        
    def run(self):
        return super().run()
    
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

