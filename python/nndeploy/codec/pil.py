
import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.device
import nndeploy.dag

import json

import PIL
import numpy as np

class PILImage2Numpy(nndeploy.dag.Node):
    def __init__(self, name, inputs: [nndeploy.dag.Edge] = [], outputs: [nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key("nndeploy.codec.PILImage2Numpy")
        self.set_desc("PIL Image to Numpy")
        self.set_input_type(PIL.Image)
        self.set_output_type(np.ndarray)
        
    def run(self) -> bool:
        input_edge = self.get_input(0) # 获取输入边
        image = input_edge.get(self) # 获取输入的image
        image_array = np.array(image)
        cv_mat = image_array[:, :, ::-1]  # 使用numpy切片将RGB转换为BGR
        output_edge = self.get_output(0) # 获取输出边
        output_edge.set(cv_mat) # 将输出写入到输出边中
        return nndeploy.base.Status.ok()
      
    def serialize(self):
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        return json.dumps(json_obj)
        
    def deserialize(self, target: str):
        json_obj = json.loads(target)
        return super().deserialize(target)
      
class PILImage2NumpyCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = PILImage2Numpy(name, inputs, outputs)
        return self.node

pil_image2numpy_node_creator = PILImage2NumpyCreator()
nndeploy.dag.register_node("nndeploy.codec.PILImage2Numpy", pil_image2numpy_node_creator)
    

class PILImageEncodec(nndeploy.dag.Node):
    def __init__(self, name, inputs: [nndeploy.dag.Edge] = [], outputs: [nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key("nndeploy.codec.PILImageEncodec")
        self.set_desc("PIL Image Encodec")
        self.set_input_type(PIL.Image)
        self.path_ = "resources/images/output.jpg"
    
    def run(self) -> bool:
        input_edge = self.get_input(0) # 获取输入边
        image = input_edge.get(self) # 获取输入的image
        image.save(self.path_)
        return nndeploy.base.Status.ok()
        
    def serialize(self):
        self.add_required_param("path_")
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        json_obj["path_"] = self.path_
        return json.dumps(json_obj)

    def deserialize(self, target: str):
        json_obj = json.loads(target)
        self.path_ = json_obj["path_"]
        return super().deserialize(target)

class PILImageEncodecCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = PILImageEncodec(name, inputs, outputs)
        return self.node

pil_image_encodec_node_creator = PILImageEncodecCreator()
nndeploy.dag.register_node("nndeploy.codec.PILImageEncodec", pil_image_encodec_node_creator)
