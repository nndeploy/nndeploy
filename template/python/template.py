import numpy as np
import json

import nndeploy.dag

class TemplatePy(nndeploy.dag.Node):
    def __init__(self, name, inputs: list[nndeploy.dag.Edge] = None, outputs: list[nndeploy.dag.Edge] = None):
        super().__init__(name, inputs, outputs) 
        super().set_key("nndeploy.template_py.TemplatePy") # The unique identifier for this node
        super().set_desc("User-defined custom node") # Description of this node
        self.set_input_type(np.ndarray) # Input type of the node
        self.set_output_type(np.ndarray) # Output type of the node
        self.frontend_show_param = 0.0 # Parameters to be displayed in the frontend, such as arrays, bool values, strings, lists, dicts, etc. The frontend will choose appropriate UI components based on data types
                
    def run(self):
        input_edge = self.get_input(0) # Get the input edge
        input_numpy = input_edge.get(self) # Get the input numpy array
        gray = np.dot(input_numpy[...,:3], [0.114, 0.587, 0.299])# bgr->gray
        gray = gray.astype(np.uint8)
        output_edge = self.get_output(0) # Get the output edge
        output_edge.set(gray) # Write the output to the output edge
        return nndeploy.base.Status.ok()
    
    def serialize(self):
        # If there are parameters that need to be displayed in the frontend, you need to override the serialize and deserialize methods
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        json_obj["frontend_show_param"] = self.frontend_show_param # Example parameter frontend_show_param, the frontend will choose appropriate UI components based on data types
        return json.dumps(json_obj)

    def deserialize(self, target: str):
        json_obj = json.loads(target)
        self.frontend_show_param = json_obj["frontend_show_param"] # Example parameter frontend_show_param, get the parameters adjusted by the frontend
        return super().deserialize(target)

# Node creator class for creating TemplatePy node instances
class TemplatePyCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        # Create and return TemplatePy node instance
        self.node = TemplatePy(name, inputs, outputs)
        return self.node

# Create node creator instance
template_py_node_creator = TemplatePyCreator()
# The first parameter is the unique identifier of the node, which must be consistent with the identifier set by set_key() in the node class. The second parameter is the node creator instance used to create nodes of this type
nndeploy.dag.register_node("nndeploy.template_py.TemplatePy", template_py_node_creator)