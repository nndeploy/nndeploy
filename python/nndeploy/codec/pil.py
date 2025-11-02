
import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.device
import nndeploy.dag

import json

from PIL import Image
import numpy as np
from typing import List

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

# PILImage2Gif

# PILImage2Video

# MakeImageGrid
class MakeImageGrid(nndeploy.dag.Node):
    def __init__(self, name, inputs: [nndeploy.dag.Edge] = [], outputs: [nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key("nndeploy.codec.MakeImageGrid")
        self.set_desc("Concatenate multiple PIL images into a grid")
        self.rows = 1
        self.cols = 1
        self.resize_h = -1
        self.resize_w = -1
        self.set_input_type(Image)  # 动态输入，类型为PIL.Image
        self.set_output_type(Image)
        self.set_dynamic_input(True)

    def run(self) -> bool:
        try:
            # 动态输入，收集所有输入边的图像
            images: List[Image.Image] = []
            for i in range(len(self.get_all_input())):
                edge = self.get_input(i)
                img = edge.get(self)
                if img is not None:
                    images.append(img)
            if len(images) != self.rows * self.cols:
                print(f"MakeImageGrid: 输入图像数量({len(images)})与网格(rows*cols={self.rows*self.cols})不符")
                return nndeploy.base.Status.error()
            # 可选resize
            if self.resize_h != -1 and self.resize_w != -1:
                images = [img.resize((self.resize_w, self.resize_h)) for img in images]
            w, h = images[0].size
            grid = Image.new("RGB", size=(self.cols * w, self.rows * h))
            for i, img in enumerate(images):
                grid.paste(img, box=(i % self.cols * w, i // self.cols * h))
            # 输出到输出边
            output_edge = self.get_output(0)
            output_edge.set(grid)
            return nndeploy.base.Status.ok()
        except Exception as e:
            print(f"MakeImageGrid节点运行失败: {e}")
            return nndeploy.base.Status.error()

    def serialize(self):
        self.add_required_param("rows")
        self.add_required_param("cols")
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        json_obj["rows"] = self.rows
        json_obj["cols"] = self.cols
        json_obj["resize_h"] = self.resize_h
        json_obj["resize_w"] = self.resize_w
        return json.dumps(json_obj, ensure_ascii=False, indent=2)

    def deserialize(self, target: str):
        try:
            json_obj = json.loads(target)
            if "rows" in json_obj:
                self.rows = int(json_obj["rows"])
            if "cols" in json_obj:
                self.cols = int(json_obj["cols"])
            if "resize_h" in json_obj:
                self.resize_h = int(json_obj["resize_h"])
            if "resize_w" in json_obj:
                self.resize_w = int(json_obj["resize_w"])
            return super().deserialize(target)
        except Exception as e:
            print(f"MakeImageGrid节点反序列化失败: {e}")
            return nndeploy.base.Status.error()

class MakeImageGridCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = MakeImageGrid(name, inputs, outputs)
        return self.node

make_image_grid_node_creator = MakeImageGridCreator()
nndeploy.dag.register_node("nndeploy.codec.MakeImageGrid", make_image_grid_node_creator)

