import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.device
import nndeploy.dag

import json
import torch
import time
from diffusers import DiffusionPipeline

from typing import Dict, Any, List, Optional
import numpy as np
import PIL.Image

from huggingface_hub import list_models

from .diffusers_util import get_diffusers_util

diffuser_model_ids = get_diffusers_util().get_supported_models(limit=10)

def add_diffuser_model_id(model_id: str):
    diffuser_model_ids.append(model_id)
    
def remove_diffuser_model_id(model_id: str):
    diffuser_model_ids.remove(model_id)
    
def get_diffuser_models_enum_json():
    enum_list = []
    for diffuser_model_id in diffuser_model_ids:
        enum_list.append(diffuser_model_id)
    precision_type_enum = {}
    for single_enum in enum_list:
        precision_type_enum[f"{single_enum}"] = enum_list
    return precision_type_enum

nndeploy.base.all_type_enum.append(get_diffuser_models_enum_json)
    
# print(diffuser_model_ids)

# def get_cache_dir():
#     """获取 Hugging Face 模型缓存目录路径"""
#     import os
#     # 优先使用环境变量 HF_HOME 设置的路径
#     hf_home = os.environ.get('HF_HOME')
#     if hf_home:
#         return hf_home
    
#     # 其次使用环境变量 HUGGINGFACE_HUB_CACHE 设置的路径
#     hf_cache = os.environ.get('HUGGINGFACE_HUB_CACHE')
#     if hf_cache:
#         return hf_cache
    
#     # 默认使用项目相对路径
#     return "resources/models"

class DiffusersPipeline(nndeploy.dag.Node):
    def __init__(self, name, inputs: [nndeploy.dag.Edge] = [], outputs: [nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key("nndeploy.diffusers.DiffusersPipeline")
        self.set_desc("Diffusers Pipeline")
        self.set_input_type(str)
        self.set_output_type(PIL.Image)
        # self.set_output_type(np.ndarray)
        self.set_device_type(nndeploy.base.DeviceType(nndeploy.base.DeviceTypeCode.cuda, 0))
        
        self.diffuser_model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        self.torch_dtype = torch.float16
        # self.cache_dir = get_cache_dir()
        # print(f"cache_dir: {self.cache_dir}")
        
        self.pipeline = None
        
    def init(self) -> bool:
        print("init")
        try:
            self.pipeline = DiffusionPipeline.from_pretrained(
                self.diffuser_model_id, 
                torch_dtype=self.torch_dtype,
                local_files_only=True,  # 强制只使用本地文件
                use_safetensors=True     # 优先使用safetensors格式
            )
        except Exception as e:
            self.pipeline = DiffusionPipeline.from_pretrained(
                self.diffuser_model_id, 
                torch_dtype=self.torch_dtype,
                resume_download=True,    # 支持断点续传
                force_download=False 
            )
        
        print("init pipeline")
        device_type = self.get_device_type()
        if device_type.code_ == nndeploy.base.DeviceTypeCode.cuda:
            device_id = device_type.device_id_
            self.pipeline.to(f"cuda:{device_id}")
        elif device_type.code_ == nndeploy.base.DeviceTypeCode.cpu:
            self.pipeline.to("cpu")
        else:
            self.pipeline.to("cpu")
        
        return nndeploy.base.Status.ok()
    
    def run(self) -> bool:
       input_edge = self.get_input(0) # 获取输入的edge
       input_str = input_edge.get(self) # 获取输入的str
       print(input_str)
       image = self.pipeline(input_str).images[0]
       # image = self.pipeline("An image of a squirrel in Picasso style").images[0]
       # image.save("image.png") 
       # image_array = np.array(image)
       output_edge = self.get_output(0) # 获取输出边
       output_edge.set(image) # 将输出写入到输出边中
       return nndeploy.base.Status.ok()
   
    def serialize(self):
        self.add_required_param("diffuser_model_id")
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        json_obj["diffuser_model_id"] = self.diffuser_model_id
        return json.dumps(json_obj)

    def deserialize(self, target: str):
        json_obj = json.loads(target)
        self.diffuser_model_id = json_obj["diffuser_model_id"]
        return super().deserialize(target)


class DiffusersPipelineCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = DiffusersPipeline(name, inputs, outputs)
        return self.node

diffusers_pipeline_node_creator = DiffusersPipelineCreator()
nndeploy.dag.register_node("nndeploy.diffusers.DiffusersPipeline", diffusers_pipeline_node_creator)
