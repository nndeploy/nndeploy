from typing import Any, List
import gfpgan
import os
import numpy as np
import json

import nndeploy.base
import nndeploy.device
import nndeploy.dag

class GFPGAN(nndeploy.dag.Node):
    def __init__(self, name, inputs: list[nndeploy.dag.Edge] = None, outputs: list[nndeploy.dag.Edge] = None):
        super().__init__(name, inputs, outputs)
        super().set_key("nndeploy.gan.GFPGAN")
        super().set_desc("GFPGAN: Make faces clearer")
        self.set_input_type(np.ndarray)
        self.set_output_type(np.ndarray)
        
        self.model_path_ = "GFPGANv1.4.pth"
        self.upscale_ = 1
        self.device_, _ = nndeploy.device.get_available_device()
        
        # print(self.device_)
        
    def init(self):
        self.gfpgan = gfpgan.GFPGANer(self.model_path_, upscale=self.upscale_, device=self.device_)
        return nndeploy.base.Status.ok()
        
    def run(self):
        input_edge = self.get_input(0)
        input_numpy = input_edge.get(self)
        _, _, self.temp_frame = self.gfpgan.enhance(input_numpy, paste_back=True)
        self.get_output(0).set(self.temp_frame)
        return nndeploy.base.Status.ok()
    
    def serialize(self):
        self.add_required_param("model_path_")
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        json_obj["model_path_"] = self.model_path_
        json_obj["upscale_"] = self.upscale_
        return json.dumps(json_obj)
    
    def deserialize(self, target: str):
        json_obj = json.loads(target)
        self.model_path_ = json_obj["model_path_"]
        self.upscale_ = json_obj["upscale_"]
        return super().deserialize(target)
    
    
class GFPGANCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = GFPGAN(name, inputs, outputs)
        return self.node
    
gfpgan_node_creator = GFPGANCreator()
nndeploy.dag.register_node("nndeploy.gan.GFPGAN", gfpgan_node_creator)