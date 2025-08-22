import nndeploy._nndeploy_internal as _C

import sys
from enum import Enum
from typing import Union
import numpy as np
import json
import importlib

import nndeploy.base
import nndeploy.device

from .base import EdgeTypeInfo
from .edge import Edge

class NodeDesc(_C.dag.NodeDesc):
    """Node description class that wraps the C++ NodeDesc.
    
    This class provides a Python interface for node descriptions,
    including name, inputs, outputs, and optional key.
    """
    def __init__(self, name: str = "", inputs: list[str] = None, outputs: list[str] = None, key: str = None):
        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = []
            
        if key is None:
            super().__init__(name, inputs, outputs)
        else:
            super().__init__(name, key, inputs, outputs)
    
    def get_key(self) -> str:
        """Get the unique key of the node."""
        return super().get_key()
        
    def get_name(self) -> str:
        """Get the name of the node."""
        return super().get_name()
        
    def get_inputs(self) -> list[str]:
        """Get the list of input names."""
        return super().get_inputs()
        
    def get_outputs(self) -> list[str]:
        """Get the list of output names."""
        return super().get_outputs()
    
    def serialize(self, target: str):
        """Serialize the node description to a string."""
        return super().serialize(target)

    def deserialize(self, target: str):
        """Deserialize the node description from a string."""
        return super().deserialize(target)

class Node(_C.dag.Node):
    """Main Node class that wraps the C++ Node implementation.
    
    This class provides a Python interface for graph nodes, including
    initialization, execution, parameter management, and I/O handling.
    """
    def __init__(self, name: str, inputs=None, outputs=None):
        if inputs is None and outputs is None:
            super().__init__(name)
        elif isinstance(inputs, list) and isinstance(outputs, list):
            super().__init__(name, inputs, outputs)
        else:
            super().__init__(name, inputs, outputs)
        
    def __del__(self):
        """Destructor that ensures proper cleanup of initialized nodes."""
        if self.get_initialized():
            self.deinit()
            self.set_initialized_flag(False)
        # super().__del__()
            
    def set_key(self, key: str):
        """Set the unique key for the node."""
        return super().set_key(key)
            
    def get_key(self) -> str:
        """Get the unique key of the node."""
        return super().get_key()
            
    def set_name(self, name: str):
        """Set the name of the node."""
        return super().set_name(name)
        
    def get_name(self) -> str:
        """Get the name of the node."""
        return super().get_name()
    
    def set_developer(self, developer: str):
        """Set the developer information for the node."""
        return super().set_developer(developer)
    
    def get_developer(self) -> str:
        """Get the developer information of the node."""
        return super().get_developer()
    
    def set_source(self, source: str):
        """Set the Source repository information for the node."""
        return super().set_source(source)
    
    def get_source(self) -> str:
        """Get the Source repository information of the node."""
        return super().get_source()
    
    def set_desc(self, desc: str):
        """Set the description of the node."""
        return super().set_desc(desc)
    
    def get_desc(self) -> str:
        """Get the description of the node."""
        return super().get_desc()
            
    def set_graph(self, graph):
        """Set the parent graph for this node."""
        return super().set_graph(graph)
        
    def get_graph(self):
        """Get the parent graph of this node."""
        return super().get_graph()
        
    def set_device_type(self, device_type: nndeploy.base.DeviceType):
        """Set the device type for node execution."""
        return super().set_device_type(device_type)
        
    def get_device_type(self) -> nndeploy.base.DeviceType:
        """Get the device type of the node."""
        return super().get_device_type()
        
    def set_param(self, param):
        """Set the parameters for the node."""
        return super().set_param(param)
        
    def get_param(self):
        """Get the parameters of the node."""
        return super().get_param()
        
    def set_external_param(self, key: str, external_param):
        """Set external parameters using a key-value pair."""
        return super().set_external_param(key, external_param)
        
    def get_external_param(self, key: str):
        """Get external parameter by key."""
        return super().get_external_param(key)
    
    def set_version(self, version: str):
        """设置节点版本。"""
        return super().set_version(version)
    
    def get_version(self) -> str:
        """获取节点版本。"""
        return super().get_version()
    
    def set_required_params(self, required_params: list):
        """设置必需参数列表。"""
        return super().set_required_params(required_params)
    
    def add_required_param(self, required_param: str):
        """添加必需参数。"""
        return super().add_required_param(required_param)
    
    def remove_required_param(self, required_param: str):
        """移除必需参数。"""
        return super().remove_required_param(required_param)
    
    def clear_required_params(self):
        """清空必需参数列表。"""
        return super().clear_required_params()
    
    def get_required_params(self) -> list:
        """获取必需参数列表。"""
        return super().get_required_params()
        
    def set_input(self, input, index: int = -1):
        """Set input at the specified index."""
        return super().set_input(input, index)
        
    def set_output(self, output, index: int = -1):
        """Set output at the specified index."""
        return super().set_output(output, index)
        
    def set_inputs(self, inputs):
        """Set all inputs for the node."""
        return super().set_inputs(inputs)
        
    def set_outputs(self, outputs):
        """Set all outputs for the node."""
        return super().set_outputs(outputs)
        
    # def set_input_shared_ptr(self, input, index: int = -1):
    #     return super().set_input_shared_ptr(input, index)
        
    # def set_output_shared_ptr(self, output, index: int = -1):
    #     return super().set_output_shared_ptr(output, index)
        
    # def set_inputs_shared_ptr(self, inputs):
    #     return super().set_inputs_shared_ptr(inputs)
        
    # def set_outputs_shared_ptr(self, outputs):
    #     return super().set_outputs_shared_ptr(outputs)
        
    def get_input(self, index: int = 0):
        """Get input at the specified index."""
        return super().get_input(index)
        
    def get_output(self, index: int = 0):
        """Get output at the specified index."""
        return (super().get_output(index))
        
    def get_all_input(self):
        """Get all inputs of the node."""
        return super().get_all_input()
        
    def get_all_output(self):
        """Get all outputs of the node."""
        return super().get_all_output()
        
    def get_constructed(self) -> bool:
        """Check if the node has been constructed."""
        return super().get_constructed()
        
    def set_parallel_type(self, parallel_type: nndeploy.base.ParallelType):
        """Set the parallel execution type for the node."""
        return super().set_parallel_type(parallel_type)
        
    def get_parallel_type(self) -> nndeploy.base.ParallelType:
        """Get the parallel execution type of the node."""
        return super().get_parallel_type()
        
    def set_inner_flag(self, flag: bool):
        """Set the inner flag for the node."""
        return super().set_inner_flag(flag)
        
    def set_initialized_flag(self, flag: bool):
        """Set the initialization flag for the node."""
        return super().set_initialized_flag(flag)
        
    def get_initialized(self) -> bool:
        """Check if the node has been initialized."""
        return super().get_initialized()
        
    def set_time_profile_flag(self, flag: bool):
        """Enable or disable time profiling for the node."""
        return super().set_time_profile_flag(flag)
        
    def get_time_profile_flag(self) -> bool:
        """Check if time profiling is enabled."""
        return super().get_time_profile_flag()
        
    def set_debug_flag(self, flag: bool):
        """Enable or disable debug mode for the node."""
        return super().set_debug_flag(flag)
        
    def get_debug_flag(self) -> bool:
        """Check if debug mode is enabled."""
        return super().get_debug_flag()
        
    def set_running_flag(self, flag: bool):
        """Set the running flag for the node."""
        return super().set_running_flag(flag)
        
    def is_running(self) -> bool:
        """Check if the node is currently running."""
        return super().is_running()
        
    def set_trace_flag(self, flag: bool):
        """Enable or disable tracing for the node."""
        return super().set_trace_flag(flag)
        
    def get_trace_flag(self) -> bool:
        """Check if tracing is enabled."""
        return super().get_trace_flag()
        
    def set_stream(self, stream):
        """Set the execution stream for the node."""
        return super().set_stream(stream)
        
    def get_stream(self):
        """Get the execution stream of the node."""
        return super().get_stream()
    
    def set_input_type_info(self, input_type_info: EdgeTypeInfo, desc: str = ""):
        """Set the input type information for the node."""
        return super().set_input_type_info(input_type_info, desc)
    
    def set_input_type(self, input_type: type, desc: str = ""):
        """Set the input type for the node.
        
        Args:
            input_type: The input type
            
        Returns:
            Status code
        """
        edge_type_info = EdgeTypeInfo()
        edge_type_info.set_type(input_type)
        return self.set_input_type_info(edge_type_info, desc)
        
    def get_input_type_info(self) -> EdgeTypeInfo:
        """Get the input type information of the node."""
        return super().get_input_type_info()
    
    def set_output_type_info(self, output_type_info: EdgeTypeInfo, desc: str = ""):
        """Set the output type information for the node."""
        return super().set_output_type_info(output_type_info, desc)
    
    def set_output_type(self, output_type: type, desc: str = ""):
        """Set the output type for the node."""
        edge_type_info = EdgeTypeInfo()
        edge_type_info.set_type(output_type)
        return self.set_output_type_info(edge_type_info, desc)
        
    def get_output_type_info(self) -> EdgeTypeInfo:
        """Get the output type information of the node."""
        return super().get_output_type_info()
        
    def init(self):
        """Initialize the node."""
        return super().init()
        
    def deinit(self):
        """Deinitialize the node and clean up resources."""
        return super().deinit()
        
    def get_memory_size(self) -> int:
        """Get the memory size required by the node."""
        return super().get_memory_size()
        
    def set_memory(self, buffer: nndeploy.device.Buffer):
        """Set the memory buffer for the node."""
        return super().set_memory(buffer)
        
    def update_input(self):
        """Update the input data for the node."""
        return super().update_input()
        
    def run(self):
        """Execute the node's main computation."""
        return super().run()
    
    def synchronize(self) -> bool:
        """Synchronize the node execution."""
        return super().synchronize()
        
    def __call__(self, inputs):
        """Make the node callable with inputs."""
        return super().__call__(inputs)
    
    def check_inputs(self, inputs) -> bool:
        """Validate the input data."""
        return super().check_inputs(inputs)
        
    def check_outputs(self, outputs_name) -> bool:
        """Validate the output names."""
        return super().check_outputs(outputs_name)
        
    def get_real_outputs_name(self):
        """Get the actual output names."""
        return super().get_real_outputs_name()
    
    def serialize(self) -> str:
        """Serialize the node to a JSON string."""
        return super().serialize()
    
    def save_file(self, path: str):
        """Save the node configuration to a file."""
        return super().save_file(path)
    
    def deserialize(self, target: str):
        """Deserialize the node from a JSON string."""
        return super().deserialize(target)
    
    def load_file(self, path: str):
        """Load the node configuration from a file."""
        return super().load_file(path)


class NodeCreator(_C.dag.NodeCreator):
    """Abstract base class for creating nodes.
    
    This class should be subclassed to implement specific node creation logic.
    """
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[Edge], outputs: list[Edge]):
        """Create a new node instance.
        
        This method must be implemented by subclasses.
        
        Args:
            name: The name of the node
            inputs: List of input edges
            outputs: List of output edges
            
        Returns:
            The created node instance
        """
        print("Must be implemented!!!")
        return None
            
        
def get_node_keys():
    """Get all registered node keys."""
    return _C.dag.get_node_keys()
    
    
def register_node(node_key: str, node_creator: NodeCreator):
    """Register a node creator with a specific key.
    
    Args:
        node_key: The unique key for the node type
        node_creator: The creator instance for this node type
    """
    return _C.dag.register_node(node_key, node_creator)


def create_node(node_key: str, node_name: str, inputs: list[Edge] = None, outputs: list[Edge] = None):
    """Create a node instance using the registered creator.
    
    Args:
        node_key: The key of the node type to create
        node_name: The name for the new node instance
        inputs: Optional list of input edges
        outputs: Optional list of output edges
        
    Returns:
        The created node instance
    """
    if inputs is None and outputs is None:
        return _C.dag.create_node(node_key, node_name)
    else:
        return _C.dag.create_node(node_key, node_name, inputs, outputs)


def get_node_json(node_key: str):
    """Get the JSON representation of a node type.
    
    This function creates a temporary node instance, initializes it with
    default parameters, and serializes it to JSON format.
    
    Args:
        node_key: The key of the node type
        
    Returns:
        JSON string representation of the node, or empty string if failed
    """
    node_name = node_key.split("::")[-1]
    node_name = node_name.split(".")[-1]
    # print(node_key)
    node = create_node(node_key, node_name)
    if node is None:
        raise RuntimeError(f"create_node failed: {node_key}")
    
    status = nndeploy.base.StatusCode.Ok
    status = node.default_param()
    if status != nndeploy.base.StatusCode.Ok:
        print(f"node key[{node.get_key()}] default_param failed: {status}")
        return ""
    
    # print(node)   
    is_graph = node.get_graph_flag()
    is_graph_type = isinstance(node, _C.dag.Graph)
    
    if is_graph and not is_graph_type:
        print(f"node key[{node.get_key()}] is graph, but not export python graph type")
        return ""
    if is_graph and is_graph_type:
        # print(node)        
        node.set_inner_flag(True)
        status = node.to_static_graph()
        if status != nndeploy.base.StatusCode.Ok:
            print(f"node key[{node.get_key()}] to_static_graph failed: {status}")   
            return ""
    if status == nndeploy.base.StatusCode.Ok:
        json_str = node.serialize()
        return json_str
    else:
        return ""
            
class ImportLib:
    """Library import manager for handling dynamic library and Python module imports.
    
    This class manages the import of various types of libraries and modules,
    including dynamic libraries (.so, .dll, .dylib), Python files, and modules.
    """
    def __init__(self):
        self.library_path_set = dict() # key: path, value: false or true
        self.path_set = dict() # key: path, value: false or true
        self.py_file_set = dict() # key: path, value: false or true
        self.module_name_set = dict() # key: module_name, value: false or true
        self.class_name_set = dict() # key: (module_name, class_name), value: false or true
        self.function_name_set = dict() # key: (module_name, function_name), value: false or true
        
    def add_path(self, path: str, update: bool = False):
        """Add a path for import (library, Python file, or directory).
        
        Args:
            path: The path to add
            update: Whether to update if already exists
        """
        # Distinguish between dynamic libraries and Python files by file extension
        if path.endswith(('.so', '.dll', '.dylib')):
            # Dynamic library file
            if path not in self.library_path_set or update:
                self.library_path_set[path] = False
        elif path.endswith('.py'):
            # Python file
            if path not in self.py_file_set or update:
                self.py_file_set[path] = False
        else:
            # Python file path/directory
            if path not in self.path_set or update:
                self.path_set[path] = False
        
    def add_module(self, module_name: str, update: bool = False):
        """Add a module name for import.
        
        Args:
            module_name: The name of the module to import
            update: Whether to update if already exists
        """
        if module_name not in self.module_name_set or update:
            self.module_name_set[module_name] = False
        
    def add_class(self, module_name: str, class_name: str, update: bool = False):
        """Add a class to import from a specific module.
        
        Args:
            module_name: The name of the module containing the class
            class_name: The name of the class to import
            update: Whether to update if already exists
        """
        key = (module_name, class_name)
        if key not in self.class_name_set or update:
            self.class_name_set[key] = False
        
    def add_function(self, module_name: str, function_name: str, update: bool = False):
        """Add a function to import from a specific module.
        
        Args:
            module_name: The name of the module containing the function
            function_name: The name of the function to import
            update: Whether to update if already exists
        """
        key = (module_name, function_name)
        if key not in self.function_name_set or update:
            self.function_name_set[key] = False
        
    def import_all(self):
        """Import all registered libraries, modules, classes, and functions."""
        # Load dynamic libraries
        for library_path in self.library_path_set:
            if not self.library_path_set[library_path]:
                print(f"load_library_from_path: {library_path}")
                nndeploy.base.load_library_from_path(library_path, True)
                self.library_path_set[library_path] = True
        
        # Add paths to sys.path
        for path in self.path_set:
            if not self.path_set[path]:
                sys.path.append(path)
                self.path_set[path] = True
        
        # Load Python files directly
        for py_file in self.py_file_set:
            if not self.py_file_set[py_file]:
                print(f"load_py_file: {py_file}")
                self.load_py_file(py_file)
                self.py_file_set[py_file] = True
        
        # Import modules
        for module_name in self.module_name_set:
            if not self.module_name_set[module_name]:
                importlib.import_module(module_name)
                self.module_name_set[module_name] = True
        
        # Import classes
        for module_name, class_name in self.class_name_set:
            if not self.class_name_set[(module_name, class_name)]:
                self.import_class(module_name, class_name)
                self.class_name_set[(module_name, class_name)] = True
        
        # Import functions
        for module_name, function_name in self.function_name_set:
            if not self.function_name_set[(module_name, function_name)]:
                self.import_function(module_name, function_name)
                self.function_name_set[(module_name, function_name)] = True
    
    def load_py_file(self, py_file_path: str):
        """Load a Python file directly as a module.
        
        Args:
            py_file_path: Path to the Python file
            
        Returns:
            The loaded module or None if failed
        """
        import os
        import importlib.util
        
        # Use filename as module name
        module_name = os.path.splitext(os.path.basename(py_file_path))[0]
        
        # Create module specification
        spec = importlib.util.spec_from_file_location(module_name, py_file_path)
        if spec is None:
            print(f"not found: {py_file_path}")
            return None
        
        # Create module
        module = importlib.util.module_from_spec(spec)
        
        # Execute module
        if spec.loader is not None:
            spec.loader.exec_module(module)
        
        # Add module to sys.modules
        sys.modules[module_name] = module
        
        return module
    
    def import_module(self, module_name: str):
        """Import a module by name."""
        return importlib.import_module(module_name)
    
    def import_class(self, module_name: str, class_name: str):
        """Import a class from a module."""
        module = self.import_module(module_name)
        return getattr(module, class_name)
    
    def import_function(self, module_name: str, function_name: str):
        """Import a function from a module."""
        module = self.import_module(module_name)
        return getattr(module, function_name)
           
           
# Global import library instance
global_import_lib = ImportLib()

def add_global_import_lib(path: str):
    """Add a path to the global import library.
    
    Args:
        path: The path to add (file or directory)
    """
    import os
    if os.path.exists(path):
        global_import_lib.add_path(path)
    else:
        print(f"file or dir not found: {path}")
    
def add_global_import_lib_module(module_name: str):
    """Add a module to the global import library."""
    global_import_lib.add_module(module_name)
    
def add_global_import_lib_class(module_name: str, class_name: str):
    """Add a class to the global import library."""
    global_import_lib.add_class(module_name, class_name)
    
def add_global_import_lib_function(module_name: str, function_name: str):
    """Add a function to the global import library."""
    global_import_lib.add_function(module_name, function_name)
    
def import_global_import_lib():
    """Import all items in the global import library."""
    global_import_lib.import_all()
    
    
# List of node keys to exclude from node listing
remove_node_keys = [
    "nndeploy::dag::Graph", "nndeploy.dag.Graph", "nndeploy::dag::RunningCondition", "nndeploy::dag::Comment",
    "nndeploy::codec::BatchOpenCvDecode", "nndeploy::codec::BatchOpenCvEncode",
    "nndeploy::super_resolution::SuperResolutionGraph", "nndeploy::super_resolution::SuperResolutionPostProcess",
    "nndeploy::preprocess::BatchPreprocess"
]


def add_remove_node_keys(node_keys: list[str]):
    """Add node keys to the removal list.
    
    Args:
        node_keys: List of node keys to exclude
    """
    global remove_node_keys
    remove_node_keys.extend(node_keys)
    
def sub_remove_node_keys(node_keys: list[str]):
    """Remove node keys from the removal list.
    
    Args:
        node_keys: List of node keys to include back
    """
    global remove_node_keys
    for node_key in node_keys:
        if node_key in remove_node_keys:
            remove_node_keys.remove(node_key)    
    

# def get_all_node_json():
#     # Import all required modules
#     add_global_import_lib("/home/always/source/public/nndeploy/build/libnndeploy_plugin_template.so")
#     add_global_import_lib("/home/always/source/public/nndeploy/build/tensor/tensor_node.py")
#     import_global_import_lib()
    
#     global remove_node_keys
#     node_keys = get_node_keys()
#     real_node_keys = []
#     for node_key in node_keys:
#         if node_key in remove_node_keys:
#             continue
#         real_node_keys.append(node_key)
        
#     # Sort the keys
#     real_node_keys.sort()
    
#     node_json = "{\"nodes\":["
#     for node_key in real_node_keys:
#         json = get_node_json(node_key)
#         if json == "":
#             continue
#         node_json += json
#         if node_key != real_node_keys[-1]:
#             node_json += ","
#     node_json += "]}"
    
#     # print(node_json)
#     # Beautify json
#     node_json = nndeploy.base.pretty_json_str(node_json)
#     return node_json


def get_all_node_json():
    """Get JSON representation of all registered nodes organized in a tree structure.
    
    This function creates a hierarchical representation of all nodes based on their
    namespace structure, with branch nodes representing namespaces and leaf nodes
    representing actual node implementations.
    
    Returns:
        JSON string containing all nodes in tree format
    """
    # Import all required modules
    # add_global_import_lib("/home/always/source/public/nndeploy/build/libnndeploy_plugin_template.so")
    # add_global_import_lib("/home/always/source/public/nndeploy/build/tensor/tensor_node.py")
    import_global_import_lib()
    
    global remove_node_keys
    node_keys = get_node_keys()
    real_node_keys = []
    for node_key in node_keys:
        if node_key in remove_node_keys:
            continue
        real_node_keys.append(node_key)
        
    # Sort the keys
    real_node_keys.sort()
    
    # Initialize node list
    nodes = []
    
    # Group by namespace and create multi-level directory structure
    namespace_groups = {}
    for node_key in real_node_keys:
        # Handle both . and :: separators
        if "::" in node_key:
            parts = node_key.split("::")
        else:
            parts = node_key.split(".")
        
        # Skip first-level namespace 'nndeploy', start from second level
        if len(parts) > 1 and parts[0] == "nndeploy":
            if len(parts) > 2:
                # Build from second-level namespace, remove first-level 'nndeploy'
                if "::" in node_key:
                    namespace = "::".join(parts[1:-1])
                else:
                    namespace = ".".join(parts[1:-1])
            else:
                namespace = ""
        else:
            # Get namespace (excluding the last node name)
            if len(parts) > 1:
                namespace = "::".join(parts[:-1]) if "::" in node_key else ".".join(parts[:-1])
            else:
                namespace = ""
        
        # Add node to corresponding namespace group
        if namespace not in namespace_groups:
            namespace_groups[namespace] = []
        namespace_groups[namespace].append(node_key)
    
    # Collect all directory paths that need to be created
    all_directories = set()
    for namespace in namespace_groups.keys():
        if namespace != "":
            # Split namespace based on separator type
            if "::" in namespace:
                parts = namespace.split("::")
                separator = "::"
            else:
                parts = namespace.split(".")
                separator = "."
            
            # Generate all levels of directory paths
            current_path = ""
            for part in parts:
                current_path = f"{current_path}{separator}{part}" if current_path else part
                all_directories.add(current_path)
    
    # Sort by path length to ensure parent directories are created first
    sorted_directories = sorted(all_directories, key=lambda x: len(x.split("::" if "::" in x else ".")))
    
    # Create directory nodes
    for directory_path in sorted_directories:
        # Split path based on separator type
        if "::" in directory_path:
            parts = directory_path.split("::")
            separator = "::"
        else:
            parts = directory_path.split(".")
            separator = "."
        
        # Determine parent directory ID
        if len(parts) > 1:
            parent_path = separator.join(parts[:-1])
        else:
            parent_path = ""
        
        # Create directory node
        branch_node = {
            "id": directory_path,
            "name": parts[-1],
            "desc": f"{directory_path}",
            "parentId": parent_path,
            "type": "branch"
        }
        nodes.append(branch_node)
    
    # Add leaf nodes
    for namespace, node_keys_in_namespace in namespace_groups.items():
        for node_key in node_keys_in_namespace:
            json_str = get_node_json(node_key)
            if json_str != "":
                node_data = json.loads(json_str)
                
                # Get node name (last part)
                if "::" in node_key:
                    node_name = node_key.split("::")[-1]
                else:
                    node_name = node_key.split(".")[-1]
                
                leaf_node = {
                    "id": node_key,
                    "name": node_name,
                    "parentId": namespace,
                    "desc": f"{node_key}",
                    "type": "leaf",
                    "nodeEntity": node_data
                }
                nodes.append(leaf_node)
    
    result = {"nodes": nodes}
    # Beautify json
    node_json = json.dumps(result, ensure_ascii=False, indent=2)
    return node_json


# def get_all_node_json():
#     # Import all required modules
#     add_global_import_lib("/home/always/source/public/nndeploy/build/libnndeploy_plugin_template.so")
#     add_global_import_lib("/home/always/source/public/nndeploy/build/tensor/tensor_node.py")
#     import_global_import_lib()
    
#     global remove_node_keys
#     node_keys = get_node_keys()
#     real_node_keys = []
#     for node_key in node_keys:
#         if node_key in remove_node_keys:
#             continue
#         real_node_keys.append(node_key)
        
#     real_node_keys.sort()
        
#     # Parse namespace
#     namespace_tree = {}
    
#     for node_key in real_node_keys:
#         json_str = get_node_json(node_key)
#         if json_str == "":
#             continue
        
#         node_data = json.loads(json_str)
        
#         # Analyze namespace path
#         if "::" in node_key:
#             parts = node_key.split("::")
#             separator = "::"
#         else:
#             parts = node_key.split(".")
#             separator = "."
            
#         # Skip first-level namespace, start from second level
#         namespace_parts = parts[1:-1]  # Skip first level and last node name
#         node_name = parts[-1]
        
#         # Build nested structure
#         current_tree = namespace_tree
        
#         part = ""
#         for i, part in enumerate(namespace_parts):
#             if part not in current_tree:
#                 # Calculate current node's full ID and parent ID
#                 current_id = separator.join(parts[1:i+2])  # Path from second level
                
#                 if i == 0:
#                     # Second-level node, parent ID is empty (root level)
#                     parent_id = ""
#                 else:
#                     # Deeper level, parent ID is the full path of previous level
#                     parent_id = separator.join(parts[1:i+1])
                
#                 current_tree[part] = {
#                     "id": current_id,
#                     "parentId": parent_id,
#                     "type": "branch",
#                     "children_dict": {},
#                     "leaf_nodes": []
#                 }
            
#             # Move to next level
#             current_tree = current_tree[part]
        
#         # Add leaf node to the last level
#         if "leaf_nodes" not in current_tree:
#             current_tree["leaf_nodes"] = []
#         current_tree["leaf_nodes"].append(node_data)
        
    
#     result = {"nodes": namespace_tree}
    
#     # Beautify json
#     node_json = json.dumps(result, ensure_ascii=False, indent=2)
#     return node_json
