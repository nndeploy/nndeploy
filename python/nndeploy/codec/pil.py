
import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.device
import nndeploy.dag

import json

from PIL import Image
import numpy as np
   

class PILImageEncodec(nndeploy.dag.Node):
    def __init__(self, name, inputs: [nndeploy.dag.Edge] = [], outputs: [nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key("nndeploy.codec.PILImageEncodec")
        self.set_desc("PIL Image Encodec")
        self.set_input_type(Image)
        self.set_node_type(nndeploy.dag.NodeType.Output)
        self.set_io_type(nndeploy.dag.IOType.Image)
        self.path_ = "resources/images/output.jpg"
    
    def run(self) -> bool:
        input_edge = self.get_input(0) # 获取输入边
        image = input_edge.get(self) # 获取输入的image
        image.save(self.path_)
        return nndeploy.base.Status.ok()
        
    def serialize(self):
        self.add_io_param("path_")
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

supported_color_modes = [
    "RGB",
    "RGBA",
    "L",
    "CMYK",
]

class PILImageDecodec(nndeploy.dag.Node):
    def __init__(self, name, inputs: [nndeploy.dag.Edge] = [], outputs: [nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key("nndeploy.codec.PILImageDecodec")
        self.set_desc("PIL Image Decodec with Color Space Conversion")
        self.set_output_type(Image)
        self.set_node_type(nndeploy.dag.NodeType.Input)
        self.set_io_type(nndeploy.dag.IOType.Image)
        self.path_ = ""
        self.color_mode_ = "RGB"  # 支持的颜色模式：RGB, RGBA, L, CMYK等
    
    def run(self) -> bool:
        try:
            # 从文件路径加载图像
            image = Image.open(self.path_)
            
            # 执行颜色空间转换
            if self.color_mode_ and image.mode != self.color_mode_:
                if self.color_mode_ == "RGB" and image.mode == "RGBA":
                    # RGBA转RGB，使用白色背景
                    background = Image.new("RGB", image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1])  # 使用alpha通道作为mask
                    image = background
                else:
                    # 通用转换
                    image = image.convert(self.color_mode_)
            
            # 输出到输出边
            output_edge = self.get_output(0)
            output_edge.set(image)
            
            return nndeploy.base.Status.ok()
            
        except Exception as e:
            print(f"PIL图像解码失败: {e}")
            return nndeploy.base.Status.error()
        
    def serialize(self):
        self.add_io_param("path_")
        self.add_required_param("path_")
        self.add_dropdown_param("color_mode_", supported_color_modes)
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        json_obj["path_"] = self.path_
        json_obj["color_mode_"] = self.color_mode_
        return json.dumps(json_obj, ensure_ascii=False, indent=2)

    def deserialize(self, target: str):
        try:
            json_obj = json.loads(target)
            if "path_" in json_obj:
                self.path_ = json_obj["path_"]
            if "color_mode_" in json_obj:
                self.color_mode_ = json_obj["color_mode_"]
            return super().deserialize(target)
        except Exception as e:
            print(f"PIL图像解码节点反序列化失败: {e}")
            return nndeploy.base.Status.error()

class PILImageDecodecCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = PILImageDecodec(name, inputs, outputs)
        return self.node

pil_image_decodec_node_creator = PILImageDecodecCreator()
nndeploy.dag.register_node("nndeploy.codec.PILImageDecodec", pil_image_decodec_node_creator)
