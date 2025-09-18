import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.device
import nndeploy.dag
from nndeploy.base import get_torch_dtype

import json
import torch
import time
import os

from diffusers import DiffusionPipeline

from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image

class FromPretrainedParam(nndeploy.base.Param):
    def __init__(self):
        super().__init__()
        # 基础参数
        self.pretrained_model_name_or_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        self.torch_dtype = "float16"  # 使用字符串表示，便于序列化
        
        # 下载和缓存相关参数
        self.mirror = "https://hf-mirror.com"
        
        # 设备和内存管理参数
        # self.device_map = "balanced"
        self.low_cpu_mem_usage = True
    
    def serialize(self) -> str:
        """序列化参数为JSON字符串"""
        param_dict = {
            "pretrained_model_name_or_path": self.pretrained_model_name_or_path,
            "torch_dtype": self.torch_dtype,
            "mirror": self.mirror,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
        }
        return json.dumps(param_dict, ensure_ascii=False, indent=2)
    
    def deserialize(self, json_str: str) -> nndeploy.base.Status:
        """从JSON字符串反序列化参数"""
        try:
            param_dict = json.loads(json_str)
            
            # 基础参数
            self.pretrained_model_name_or_path = param_dict.get("pretrained_model_name_or_path", "stable-diffusion-v1-5/stable-diffusion-v1-5")
            self.torch_dtype = param_dict.get("torch_dtype", "float16")
            
            # 下载和缓存相关参数
            self.mirror = param_dict.get("mirror", "")
            
            # 设备和内存管理参数
            # self.device_map = param_dict.get("device_map", "balanced")
            self.low_cpu_mem_usage = param_dict.get("low_cpu_mem_usage", True)            
            return nndeploy.base.Status.ok()
        except Exception as e:
            print(f"反序列化参数失败: {e}")
            return nndeploy.base.Status.error()
    
class Text2Image(nndeploy.dag.Node):
    """
    基于nndeploy框架的Diffusers Pipeline节点
    
    该节点封装了Hugging Face Diffusers库的文本到图像生成功能，
    支持Stable Diffusion等多种扩散模型的推理。
    """
    def __init__(self, name: str, inputs: list[nndeploy.dag.Edge] = [], outputs: list[nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key("nndeploy.diffusers.Text2Image")
        self.set_desc("Diffusers Pipeline for text-to-image generation")
        self.set_input_type(str, "Input text prompt")  # Input text prompt
        self.set_input_type(str, "Input negative text prompt")  # Input negative text prompt
        self.set_input_type(torch.Tensor, "Input latent")  # latent
        self.set_output_type(Image, "Output PIL image")  # Output ndarray image
        self.set_dynamic_output(True)
        
        # 初始化参数
        self.param = FromPretrainedParam()
        
        # 设置默认设备为CUDA
        self.set_device_type(nndeploy.base.DeviceType(nndeploy.base.DeviceTypeCode.cuda, 0))
        
        # 参数
        self.num_inference_steps = 50
        self.guidance_scale = 8.0
        self.eta = 0.0
        self.guidance_rescale = 0.0
        self.timesteps: List[int] = None,
        self.sigmas: List[float] = None,
        
        # 管道实例
        self.pipeline = None
        
    def init(self):        
        try:
            try:
                # 首先尝试本地加载
                self.pipeline = DiffusionPipeline.from_pretrained(
                    pretrained_model_name_or_path=self.param.pretrained_model_name_or_path,
                    local_files_only=True,
                    torch_dtype=get_torch_dtype(self.param.torch_dtype),
                    low_cpu_mem_usage=self.param.low_cpu_mem_usage
                )
            except Exception as local_error:                
                self.pipeline = DiffusionPipeline.from_pretrained(
                    pretrained_model_name_or_path=self.param.pretrained_model_name_or_path,
                    torch_dtype=get_torch_dtype(self.param.torch_dtype),
                    mirror=self.param.mirror,
                    low_cpu_mem_usage=self.param.low_cpu_mem_usage
                )
            
            # 根据设备类型移动管道
            device_type = self.get_device_type()
            if device_type.code_ == nndeploy.base.DeviceTypeCode.cuda:
                device_id = device_type.device_id_
                target_device = f"cuda:{device_id}"
                self.pipeline.to(target_device)
            elif device_type.code_ == nndeploy.base.DeviceTypeCode.cpu:
                self.pipeline.to("cpu")
            else:
                self.pipeline.to("cpu")
                
            return nndeploy.base.Status.ok()
            
        except Exception as e:
            print(f"初始化Diffusers管道失败: {e}")
            return nndeploy.base.Status.error()
    
    def run(self) -> nndeploy.base.Status:
        """执行文本到图像生成"""
        try:
            # 获取输入文本提示词
            input_edge = self.get_input(0)
            prompt = input_edge.get(self)
            
            input_edge = self.get_input(1)
            negative_prompt = input_edge.get(self)
            
            input_edge = self.get_input(2)
            latent = input_edge.get(self)
            
            timesteps=None
            if self.timesteps != [None]:
                print(f"timesteps: {self.timesteps}")
                timesteps = self.timesteps
            sigmas = None
            if self.sigmas != [None]:
                print(f"sigmas: {self.sigmas}")
                sigmas = self.sigmas
            
            num_images_per_prompt = latent.shape[0]
            
            result = self.pipeline(
                prompt=prompt,
                num_inference_steps=self.num_inference_steps,
                timesteps=timesteps,
                sigmas=sigmas,
                guidance_scale=self.guidance_scale,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                eta=self.eta,
                latents=latent,
                guidance_rescale=self.guidance_rescale
            )
                            
            # 输出到输出边
            min_len = min(len(result.images), len(self.get_all_output()))
            for i in range(min_len):
                output_edge = self.get_output(i)
                generated_image = result.images[i]
                output_edge.set(generated_image)
            
            return nndeploy.base.Status.ok()            
        except Exception as e:
            return nndeploy.base.Status.error()
    
    def serialize(self) -> str:
        """序列化节点参数"""
        try:
            # 添加必需参数
            self.add_required_param("param")
            self.add_required_param("num_inference_steps")
            
            # 获取基类序列化结果
            base_json = super().serialize()
            json_obj = json.loads(base_json)
            
            # 序列化推理参数
            json_obj["num_inference_steps"] = self.num_inference_steps
            json_obj["guidance_scale"] = self.guidance_scale
            json_obj["eta"] = self.eta
            json_obj["guidance_rescale"] = self.guidance_rescale
            json_obj["timesteps"] = self.timesteps if self.timesteps is not None else List[int]
            json_obj["sigmas"] = self.sigmas if self.sigmas is not None else List[float]
            
             # 添加自定义参数
            json_obj["param"] = json.loads(self.param.serialize())
            
            return json.dumps(json_obj, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"序列化失败: {e}")
            return "{}"
    
    def deserialize(self, json_str: str) -> nndeploy.base.Status:
        """反序列化节点参数"""
        try:
            json_obj = json.loads(json_str)
            
            # 反序列化自定义参数
            if "param" in json_obj:
                param_json = json.dumps(json_obj["param"])
                self.param.deserialize(param_json)
            
            # 反序列化推理参数
            if "num_inference_steps" in json_obj:
                self.num_inference_steps = json_obj["num_inference_steps"]
            if "guidance_scale" in json_obj:
                self.guidance_scale = json_obj["guidance_scale"]
            if "eta" in json_obj:
                self.eta = json_obj["eta"]
            if "guidance_rescale" in json_obj:
                self.guidance_rescale = json_obj["guidance_rescale"]
            if "timesteps" in json_obj:
                self.timesteps = json_obj["timesteps"]
            if "sigmas" in json_obj:
                self.sigmas = json_obj["sigmas"]
            
            # 调用基类反序列化
            return super().deserialize(json_str)
            
        except Exception as e:
            print(f"反序列化失败: {e}")
            return nndeploy.base.Status.ok()


class Text2ImageCreator(nndeploy.dag.NodeCreator):
    """Diffusers Pipeline节点创建器"""
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        """创建Diffusers Pipeline节点实例"""
        self.node = Text2Image(name, inputs, outputs)
        return self.node


# 注册节点创建器
text2image_creator = Text2ImageCreator()
nndeploy.dag.register_node("nndeploy.diffusers.Text2Image", text2image_creator)
