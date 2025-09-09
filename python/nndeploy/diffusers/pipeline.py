import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.device
import nndeploy.dag

import json
import torch
import time
import os

from diffusers import DiffusionPipeline, StableDiffusionPipeline

from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image


def get_torch_dtype(self):
    """获取torch数据类型对象"""
    if self.torch_dtype == "float16":
        return torch.float16
    elif self.torch_dtype == "float32":
        return torch.float32
    elif self.torch_dtype == "bfloat16":
        return torch.bfloat16
    else:
        return None

class FromPretrainedParam(nndeploy.base.Param):
    def __init__(self):
        super().__init__()
        # 基础参数
        self.pretrained_model_name_or_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        self.torch_dtype = "float16"  # 使用字符串表示，便于序列化
        
        # 下载和缓存相关参数
        self.mirror = "https://hf-mirror.com"
        
        # 设备和内存管理参数
        self.device_map = "balanced"
        self.low_cpu_mem_usage = True
                
        # 其他参数
        self.output_loading_info = False
    
    def serialize(self) -> str:
        """序列化参数为JSON字符串"""
        param_dict = {
            "pretrained_model_name_or_path": self.pretrained_model_name_or_path,
            "torch_dtype": self.torch_dtype,
            "mirror": self.mirror,
            "device_map": self.device_map,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
            "output_loading_info": self.output_loading_info
        }
        return json.dumps(param_dict, ensure_ascii=False, indent=2)
    
    def deserialize(self, json_str: str) -> bool:
        """从JSON字符串反序列化参数"""
        try:
            param_dict = json.loads(json_str)
            
            # 基础参数
            self.pretrained_model_name_or_path = param_dict.get("pretrained_model_name_or_path", "stable-diffusion-v1-5/stable-diffusion-v1-5")
            self.torch_dtype = param_dict.get("torch_dtype", "float16")
            
            # 下载和缓存相关参数
            self.mirror = param_dict.get("mirror", "")
            
            # 设备和内存管理参数
            self.device_map = param_dict.get("device_map", "balanced")
            self.low_cpu_mem_usage = param_dict.get("low_cpu_mem_usage", True)
            
            # 其他参数
            self.output_loading_info = param_dict.get("output_loading_info", False)
            
            return True
        except Exception as e:
            print(f"反序列化参数失败: {e}")
            return False
    
class StableDiffusion(nndeploy.dag.Node):
    """
    基于nndeploy框架的Diffusers Pipeline节点
    
    该节点封装了Hugging Face Diffusers库的文本到图像生成功能，
    支持Stable Diffusion等多种扩散模型的推理。
    """
    def __init__(self, name: str, inputs: list[nndeploy.dag.Edge] = [], outputs: list[nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key("nndeploy.diffusers.StableDiffusion")
        self.set_desc("Diffusers Pipeline for text-to-image generation")
        self.set_input_type(str)  # 输入文本提示词
        self.set_input_type(str)  # 输入反向文本提示词
        self.set_output_type(np.ndarray)  # 输出ndarray图像
        
        # 初始化参数
        self.param_ = FromPretrainedParam()
        
        # 设置默认设备为CUDA
        self.set_device_type(nndeploy.base.DeviceType(nndeploy.base.DeviceTypeCode.cuda, 0))
        self.height = 512
        self.width = 512
        self.num_inference_steps = 20
        self.guidance_scale = 8.0
        self.num_images_per_prompt = 1
        slef.is_
        self.seed = None
        self.latents = None
        self.ip_adapter_image = None
        self.ip_adapter_image_embeds = None
        # self.output_type = "numpy"
        self.cross_attention_kwargs = None
        self.guidance_rescale = 0.0
        
        # 管道实例
        self.pipeline = None
        
    def init(self) -> bool:        
        try:
            try:
                # 首先尝试本地加载
                self.pipeline = DiffusionPipeline.from_pretrained(
                    self.param_.pretrained_model_name_or_path,
                    local_files_only=True,
                    use_safetensors=True,
                    **load_kwargs
                )
                print("成功从本地加载模型")
            except Exception as local_error:
                print(f"本地加载失败: {local_error}")
                print("尝试从远程下载...")
                
                # 远程下载
                self.pipeline = DiffusionPipeline.from_pretrained(
                    self.param_.pretrained_model_name_or_path,
                    resume_download=True,
                    force_download=False,
                    **load_kwargs
                )
                print("成功从远程下载并加载模型")
            
            # 根据设备类型移动管道
            device_type = self.get_device_type()
            if device_type.code_ == nndeploy.base.DeviceTypeCode.cuda:
                device_id = device_type.device_id_
                target_device = f"cuda:{device_id}"
                print(f"将管道移动到设备: {target_device}")
                self.pipeline.to(target_device)
            elif device_type.code_ == nndeploy.base.DeviceTypeCode.cpu:
                print("将管道移动到CPU")
                self.pipeline.to("cpu")
            else:
                print("使用默认CPU设备")
                self.pipeline.to("cpu")
            
            print("Diffusers管道初始化完成")
            return True
            
        except Exception as e:
            print(f"初始化Diffusers管道失败: {e}")
            return False
    
    def run(self) -> bool:
        """执行文本到图像生成"""
        try:
            # 获取输入文本提示词
            input_edge = self.get_input(0)
            prompt = input_edge.get(self)
            
            if not prompt or not isinstance(prompt, str):
                print("错误: 输入提示词为空或格式不正确")
                return False
            
            print(f"生成图像，提示词: {prompt}")
            
            # 执行推理
            result = self.pipeline(
                prompt=prompt,
                num_inference_steps=50,  # 可以作为参数配置
                guidance_scale=7.5,      # 可以作为参数配置
                height=512,              # 可以作为参数配置
                width=512,               # 可以作为参数配置
                return_dict=True
            )
            
            # 获取生成的图像
            generated_image = result.images[0]
            
            # 输出到输出边
            output_edge = self.get_output(0)
            output_edge.set(generated_image)
            
            print("图像生成完成")
            return True
            
        except Exception as e:
            print(f"图像生成失败: {e}")
            return False
    
    def serialize(self) -> str:
        """序列化节点参数"""
        try:
            # 添加必需参数
            self.add_required_param("param")
            
            # 获取基类序列化结果
            base_json = super().serialize()
            json_obj = json.loads(base_json)
            
            # 添加自定义参数
            json_obj["param"] = json.loads(self.param_.serialize())
            
            return json.dumps(json_obj, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"序列化失败: {e}")
            return "{}"
    
    def deserialize(self, json_str: str) -> bool:
        """反序列化节点参数"""
        try:
            json_obj = json.loads(json_str)
            
            # 反序列化自定义参数
            if "param" in json_obj:
                param_json = json.dumps(json_obj["param"])
                if not self.param_.deserialize(param_json):
                    print("参数反序列化失败")
                    return False
            
            # 调用基类反序列化
            return super().deserialize(json_str)
            
        except Exception as e:
            print(f"反序列化失败: {e}")
            return False


class DiffusersPipelineCreator(nndeploy.dag.NodeCreator):
    """Diffusers Pipeline节点创建器"""
    
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        """创建Diffusers Pipeline节点实例"""
        self.node = StableDiffusion(name, inputs, outputs)
        return self.node


# 注册节点创建器
diffusers_pipeline_creator = DiffusersPipelineCreator()
nndeploy.dag.register_node("nndeploy.diffusers.StableDiffusion", diffusers_pipeline_creator)

# 保持向后兼容的别名
Pipeline = StableDiffusion
