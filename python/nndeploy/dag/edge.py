import nndeploy._nndeploy_internal as _C

from enum import Enum
from typing import Union
import numpy as np
import json

import nndeploy.base
import nndeploy.device

from .base import EdgeTypeInfo

class Edge(_C.dag.Edge):
    """
    Edge class is an edge object used to connect nodes in the nndeploy framework, inheriting from the C++ Edge class.
    Edge is responsible for passing data between nodes, supporting multiple data types and parallel modes.
    
    Main functions:
    - Data transfer: Pass various types of data between nodes
    - Type management: Manage and validate data type information
    - Queue control: Control data queue size and overflow policies
    - Parallel support: Support different parallel execution modes
    """
    
    def __init__(self, name: str = ""):
        """
        Initialize Edge object
        
        Args:
            name (str): Name of the Edge, default is empty string
        """
        super().__init__(name)

    def get_name(self) -> str:
        """
        Get the name of the Edge
        
        Returns:
            str: Name of the Edge
        """
        return super().get_name()
    
    def set_queue_max_size(self, queue_max_size: int):
        """
        Set the maximum capacity of the queue
        
        Args:
            queue_max_size (int): Maximum queue capacity, used to control memory usage
        """
        return super().set_queue_max_size(queue_max_size)
    
    def get_queue_max_size(self) -> int:
        """
        Get the maximum capacity of the queue
        
        Returns:
            int: Currently set maximum queue capacity
        """
        return super().get_queue_max_size()
    
    def set_queue_overflow_policy(self, policy, drop_count: int = 1):
        """
        Set queue overflow policy
        
        Args:
            policy: Processing policy when queue overflows (such as node backpressure, data dropping, etc.)
            drop_count (int): Number of data items to drop when overflow occurs, default is 1
        """
        return super().set_queue_overflow_policy(policy, drop_count)
    
    def get_queue_overflow_policy(self):
        """
        Get the current queue overflow policy
        
        Returns:
            Queue overflow policy object
        """
        return super().get_queue_overflow_policy()
    
    def get_queue_drop_count(self) -> int:
        """
        Get the number of data items dropped when queue overflows
        
        Returns:
            int: Number of data items dropped
        """
        return super().get_queue_drop_count()
        
    def get_parallel_type(self) -> nndeploy.base.ParallelType:
        """
        Get the parallel type of the Edge
        
        Returns:
            nndeploy.base.ParallelType: Current parallel execution type
        """
        return super().get_parallel_type()
        
    def set_parallel_type(self, parallel_type: nndeploy.base.ParallelType):
        """
        Set the parallel type of the Edge
        
        Args:
            parallel_type (nndeploy.base.ParallelType): Parallel execution type to set
        """
        return super().set_parallel_type(parallel_type)
    
    def empty(self) -> bool:
        """
        Check if the Edge is empty (no data)
        
        Returns:
            bool: True indicates Edge is empty, False indicates there is data
        """
        return super().empty()
        
    def construct(self):
        """
        Construct Edge object, initialize internal data structures
        """
        return super().construct()

    def set(self, data: any):
        """
        Set data into the Edge
        
        Args:
            data (any): Data to set, can be any type
        """
        return super().set(data)
    
    def get(self, node: _C.dag.Node):
        """
        Get data from the Edge (for a specific node)
        
        Args:
            node (_C.dag.Node): Node requesting data
            
        Returns:
            Data retrieved from the Edge
        """
        return super().get(node)
        
    def get_graph_output(self):
        """
        Get graph output data (when Edge serves as graph output)
        
        Returns:
            Graph output data
        """
        return super().get_graph_output()
        
    def create_buffer(self, device: nndeploy.device.Device, desc: nndeploy.device.BufferDesc):
        """
        Create Buffer on specified device
        
        Args:
            device (nndeploy.device.Device): Target device
            desc (nndeploy.device.BufferDesc): Buffer description information
            
        Returns:
            Created Buffer object
        """
        return super().create(device, desc)

    def create_tensor(self, device: nndeploy.device.Device, desc: nndeploy.device.TensorDesc, tensor_name: str = ""):
        """
        Create Tensor on specified device
        
        Args:
            device (nndeploy.device.Device): Target device
            desc (nndeploy.device.TensorDesc): Tensor description information
            tensor_name (str): Tensor name, default is empty
            
        Returns:
            Created Tensor object
        """
        return super().create(device, desc, tensor_name)
        
    def notify_written(self, data: Union[nndeploy.device.Buffer, nndeploy.device.Tensor]):
        """
        Notify that data has been written
        
        Args:
            data (Union[nndeploy.device.Buffer, nndeploy.device.Tensor]): Data object that has been written
        """
        return super().notify_written(data)
        
    def get_index(self, node: _C.dag.Node) -> int:
        """
        Get the index of specified node in the Edge
        
        Args:
            node (_C.dag.Node): Target node
            
        Returns:
            int: Index of the node in the Edge
        """
        return super().get_index(node)
        
    def reset_index(self):
        """
        Reset the index counter of the Edge
        """
        return super().reset_index()
        
    def get_graph_output_index(self) -> int:
        """
        Get the index of graph output
        
        Returns:
            int: Graph output index
        """
        return super().get_graph_output_index()
        
    def get_position(self, node: _C.dag.Node) -> int:
        """
        Get the position of specified node in the Edge
        
        Args:
            node (_C.dag.Node): Target node
            
        Returns:
            int: Position of the node in the Edge
        """
        return super().get_position(node)
        
    def get_graph_output_position(self) -> int:
        """
        Get the position of graph output
        
        Returns:
            int: Graph output position
        """
        return super().get_graph_output_position()
        
    def update(self, node: _C.dag.Node) -> nndeploy.base.EdgeUpdateFlag:
        """
        Update Edge status (for a specific node)
        
        Args:
            node (_C.dag.Node): Node triggering the update
            
        Returns:
            nndeploy.base.EdgeUpdateFlag: Update flag
        """
        return super().update(node)
        
    def mark_graph_output(self) -> bool:
        """
        Mark the Edge as graph output
        
        Returns:
            bool: Whether marking was successful
        """
        return super().mark_graph_output()
        
    def increase_producers(self, producers: list[_C.dag.Node]):
        """
        Add producer nodes list
        
        Args:
            producers (list[_C.dag.Node]): List of producer nodes to add
        """
        return super().increase_producers(producers)
        
    def increase_consumers(self, consumers: list[_C.dag.Node]):
        """
        Add consumer nodes list
        
        Args:
            consumers (list[_C.dag.Node]): List of consumer nodes to add
        """
        return super().increase_consumers(consumers)
    
    def get_producers(self) -> list[_C.dag.Node]:
        """
        Get all producer nodes
        
        Returns:
            list[_C.dag.Node]: List of producer nodes
        """
        return super().get_producers()
    
    def get_consumers(self) -> list[_C.dag.Node]:
        """
        Get all consumer nodes
        
        Returns:
            list[_C.dag.Node]: List of consumer nodes
        """
        return super().get_consumers()
        
    def request_terminate(self) -> bool:
        """
        Request termination of Edge execution
        
        Returns:
            bool: Whether termination request was successful
        """
        return super().request_terminate()
    
    def set_type(self, type_val):
        """
        Set the data type of the Edge
        
        Args:
            type_val: Data type value, can be Python type or type object
        """
        return super().set_type(type_val)
      
    def set_type_name(self, type_name: str):
        """
        Set the type name of the Edge
        
        Args:
            type_name (str): Type name string
        """
        return super().set_type_name(type_name)

    def get_type_name(self) -> str:
        """
        Get the type name of the Edge
        
        Returns:
            str: Type name string
        """
        return super().get_type_name()
    
    def set_type_info(self, type_info: EdgeTypeInfo):
        """
        Set the type information object of the Edge
        
        Args:
            type_info (EdgeTypeInfo): Type information object
        """
        return super().set_type_info(type_info)
    
    def get_type_info(self) -> EdgeTypeInfo:
        """
        Get the type information object of the Edge
        
        Returns:
            EdgeTypeInfo: Type information object
        """
        return super().get_type_info()
    
    def check_type_info(self, type_info: EdgeTypeInfo) -> bool:
        """
        Check if type information matches
        
        Args:
            type_info (EdgeTypeInfo): Type information to check
            
        Returns:
            bool: True indicates type matches, False indicates mismatch
        """
        return super().check_type_info(type_info)


# Global variable: Store acceptable Edge type mapping relationships
# Used for type conversion and compatibility checking between Python and C++
accepted_edge_type_map = {
    # Mapping between Python Buffer type and C++ Buffer type
    "nndeploy.device.Buffer": ["nndeploy::device::Buffer"],
    "nndeploy::device::Buffer": ["nndeploy.device.Buffer"],
    
    # Mapping between Python Tensor type and C++ Tensor type
    "nndeploy.device.Tensor": ["nndeploy::device::Tensor"],
    "nndeploy::device::Tensor": ["nndeploy.device.Tensor"],
    
    # Mapping between Python Param type and C++ Param type
    "nndeploy.base.Param": ["nndeploy::base::Param"],
    "nndeploy::base::Param": ["nndeploy.base.Param"],
}

def add_accepted_edge_type_map(edge_type_map):
    """
    Add acceptable Edge type mapping relationships
    
    Supports two input formats:
    1. dict[str, list[str]]: One type corresponds to a list of multiple compatible types
    2. dict[str, str]: One type corresponds to one compatible type
    
    Args:
        edge_type_map: Type mapping dictionary to add
            - Format 1: {"type1": ["compatible_type1", "compatible_type2"]}
            - Format 2: {"type1": "compatible_type1"}
    
    Example:
        # Add custom type mapping
        add_accepted_edge_type_map({
            "MyCustomType": ["nndeploy.device.Tensor", "numpy.ndarray"]
        })
    """
    global accepted_edge_type_map
    if isinstance(edge_type_map, dict):
        # Check value type to determine which dictionary type it is
        if edge_type_map and isinstance(next(iter(edge_type_map.values())), list):
            # Format 1: dict[str, list[str]] - direct update
            accepted_edge_type_map.update(edge_type_map)
        else:
            # Format 2: dict[str, str] - need to convert to bidirectional mapping
            for edge_type, edge_type_item in edge_type_map.items():
                # Add forward mapping edge_type -> [edge_type_item]
                if edge_type in accepted_edge_type_map:
                    accepted_edge_type_map[edge_type].append(edge_type_item)        
                else:
                    accepted_edge_type_map[edge_type] = [edge_type_item]
                    
                # Add reverse mapping edge_type_item -> [edge_type]
                if edge_type_item in accepted_edge_type_map:
                    accepted_edge_type_map[edge_type_item].append(edge_type)
                else:
                    accepted_edge_type_map[edge_type_item] = [edge_type]
                    
    
def sub_accepted_edge_type_map(edge_type_map: Union[dict[str, list[str]], dict[str, str]]):
    """
    Remove acceptable Edge type mapping relationships
    
    Args:
        edge_type_map: Type mapping dictionary to remove
            Format: {"type1": ["compatible_type1", "compatible_type2"]}
    
    Example:
        # Remove specific type mapping
        sub_accepted_edge_type_map({
            "MyCustomType": ["nndeploy.device.Tensor"]
        })
    """
    global accepted_edge_type_map
    for edge_type, edge_type_list in edge_type_map.items():
        if edge_type in accepted_edge_type_map:
            for edge_type_item in edge_type_list:
                if edge_type_item in accepted_edge_type_map[edge_type]:
                    accepted_edge_type_map[edge_type].remove(edge_type_item)
                    
def get_accepted_edge_type_map():
    """
    Get current acceptable Edge type mapping relationships
    
    Returns:
        dict: Current type mapping dictionary
    """
    global accepted_edge_type_map
    return accepted_edge_type_map

def get_accepted_edge_type_json():
    """
    Get JSON format string of acceptable Edge type mapping relationships
    
    Mainly used for:
    - Debugging and logging
    - Interface interaction with other systems
    - Configuration file export
    
    Returns:
        str: JSON format type mapping string containing complete type compatibility information
    
    Example:
        Return format:
        {
          "accepted_edge_types": {
            "nndeploy.device.Buffer": ["nndeploy::device::Buffer"],
            "nndeploy::device::Buffer": ["nndeploy.device.Buffer"],
            ...
          }
        }
    """
    import json
    
    # Convert accepted_edge_type_map to JSON format
    edge_type_data = {"accepted_edge_types": accepted_edge_type_map}
    
    # Serialize to JSON string, ensure Chinese characters display correctly, and format output
    edge_type_json = json.dumps(edge_type_data, ensure_ascii=False, indent=2)
    
    return edge_type_json