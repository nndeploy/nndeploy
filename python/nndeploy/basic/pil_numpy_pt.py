
import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.device
import nndeploy.dag

import json

from PIL import Image
import numpy as np  

import torch


class PILImage2Numpy(nndeploy.dag.Node):
    def __init__(self, name, inputs: [nndeploy.dag.Edge] = [], outputs: [nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key("nndeploy.basic.PILImage2Numpy")
        self.set_desc("PIL Image to Numpy")
        self.set_input_type(Image)
        self.set_output_type(np.ndarray)
        
    def run(self) -> bool:
        input_edge = self.get_input(0) # 获取输入边
        image = input_edge.get(self) # 获取输入的image
        image_array = np.array(image)
        cv_mat = image_array
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
nndeploy.dag.register_node("nndeploy.basic.PILImage2Numpy", pil_image2numpy_node_creator)


class Numpy2PILImage(nndeploy.dag.Node):
    def __init__(self, name, inputs: [nndeploy.dag.Edge] = [], outputs: [nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key("nndeploy.basic.Numpy2PILImage")
        self.set_desc("Numpy to PIL Image")
        self.set_input_type(np.ndarray)
        self.set_output_type(Image)
        
    def run(self) -> bool:
        input_edge = self.get_input(0) # 获取输入边
        numpy_array = input_edge.get(self) # 获取输入的numpy数组
        
        # 确保数组格式正确
        if numpy_array.dtype != np.uint8:
            # 如果是浮点数，假设范围是[0,1]，转换为[0,255]
            if numpy_array.dtype in [np.float32, np.float64]:
                numpy_array = (numpy_array * 255).astype(np.uint8)
            else:
                numpy_array = numpy_array.astype(np.uint8)
        
        # 转换为PIL Image
        pil_image = Image.fromarray(numpy_array)
        
        output_edge = self.get_output(0) # 获取输出边
        output_edge.set(pil_image) # 将输出写入到输出边中
        return nndeploy.base.Status.ok()
      
    def serialize(self):
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        return json.dumps(json_obj)
        
    def deserialize(self, target: str):
        json_obj = json.loads(target)
        return super().deserialize(target)
      
class Numpy2PILImageCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = Numpy2PILImage(name, inputs, outputs)
        return self.node

numpy2pil_image_node_creator = Numpy2PILImageCreator()
nndeploy.dag.register_node("nndeploy.basic.Numpy2PILImage", numpy2pil_image_node_creator)

# PILImage2Pt && Pt2PILImage
class PILImage2Pt(nndeploy.dag.Node):
    def __init__(self, name, inputs: [nndeploy.dag.Edge] = [], outputs: [nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key("nndeploy.basic.PILImage2Pt")
        self.set_desc("PIL Image to PyTorch Tensor")
        self.set_input_type(Image)
        self.set_output_type(torch.Tensor)
    
    def run(self) -> bool:
        input_edge = self.get_input(0)  # 获取输入边
        image = input_edge.get(self)    # 获取输入的PIL Image
        image_array = np.array(image)
        # 不做通道转换，直接转为torch tensor
        tensor = torch.from_numpy(image_array)
        output_edge = self.get_output(0)
        output_edge.set(tensor)
        return nndeploy.base.Status.ok()
    
    def serialize(self):
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        return json.dumps(json_obj)
    
    def deserialize(self, target: str):
        json_obj = json.loads(target)
        return super().deserialize(target)

class PILImage2PtCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
    
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = PILImage2Pt(name, inputs, outputs)
        return self.node

pil_image2pt_node_creator = PILImage2PtCreator()
nndeploy.dag.register_node("nndeploy.basic.PILImage2Pt", pil_image2pt_node_creator)


class Pt2PILImage(nndeploy.dag.Node):
    def __init__(self, name, inputs: [nndeploy.dag.Edge] = [], outputs: [nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key("nndeploy.basic.Pt2PILImage")
        self.set_desc("PyTorch Tensor to PIL Image")
        self.set_input_type(torch.Tensor)
        self.set_output_type(Image)
    
    def run(self) -> bool:
        input_edge = self.get_input(0)
        tensor = input_edge.get(self)
        # 确保tensor在cpu且为uint8或float
        if tensor.is_cuda:
            tensor = tensor.cpu()
        if tensor.dtype == torch.float32 or tensor.dtype == torch.float64:
            # 假设范围为[0,1]
            tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
        elif tensor.dtype != torch.uint8:
            tensor = tensor.to(torch.uint8)
        # 不做通道转换，直接转为numpy
        array = tensor.numpy()
        pil_image = Image.fromarray(array)
        output_edge = self.get_output(0)
        output_edge.set(pil_image)
        return nndeploy.base.Status.ok()
    
    def serialize(self):
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        return json.dumps(json_obj)
    
    def deserialize(self, target: str):
        json_obj = json.loads(target)
        return super().deserialize(target)

class Pt2PILImageCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
    
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = Pt2PILImage(name, inputs, outputs)
        return self.node

pt2pil_image_node_creator = Pt2PILImageCreator()
nndeploy.dag.register_node("nndeploy.basic.Pt2PILImage", pt2pil_image_node_creator)


# Numpy2Pt
class Numpy2Pt(nndeploy.dag.Node):
    def __init__(self, name, inputs: [nndeploy.dag.Edge] = [], outputs: [nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key("nndeploy.basic.Numpy2Pt")
        self.set_desc("Numpy ndarray to PyTorch Tensor")
        self.set_input_type(np.ndarray)
        self.set_output_type(torch.Tensor)

    def run(self) -> bool:
        input_edge = self.get_input(0)
        array = input_edge.get(self)
        tensor = torch.from_numpy(array)
        output_edge = self.get_output(0)
        output_edge.set(tensor)
        return nndeploy.base.Status.ok()

    def serialize(self):
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        return json.dumps(json_obj)

    def deserialize(self, target: str):
        json_obj = json.loads(target)
        return super().deserialize(target)

class Numpy2PtCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()

    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = Numpy2Pt(name, inputs, outputs)
        return self.node

numpy2pt_node_creator = Numpy2PtCreator()
nndeploy.dag.register_node("nndeploy.basic.Numpy2Pt", numpy2pt_node_creator)

# Pt2Numpy
class Pt2Numpy(nndeploy.dag.Node):
    def __init__(self, name, inputs: [nndeploy.dag.Edge] = [], outputs: [nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key("nndeploy.basic.Pt2Numpy")
        self.set_desc("PyTorch Tensor to Numpy ndarray")
        self.set_input_type(torch.Tensor)
        self.set_output_type(np.ndarray)

    def run(self) -> bool:
        input_edge = self.get_input(0)
        tensor = input_edge.get(self)
        if tensor.is_cuda:
            tensor = tensor.cpu()
        array = tensor.numpy()
        output_edge = self.get_output(0)
        output_edge.set(array)
        return nndeploy.base.Status.ok()

    def serialize(self):
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        return json.dumps(json_obj)

    def deserialize(self, target: str):
        json_obj = json.loads(target)
        return super().deserialize(target)

class Pt2NumpyCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()

    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = Pt2Numpy(name, inputs, outputs)
        return self.node

pt2numpy_node_creator = Pt2NumpyCreator()
nndeploy.dag.register_node("nndeploy.basic.Pt2Numpy", pt2numpy_node_creator)

