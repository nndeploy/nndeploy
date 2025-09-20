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
from diffusers import AutoPipelineForText2Image
from diffusers import AutoPipelineForImage2Image
from diffusers import AutoPipelineForInpainting

from diffusers.quantizers import PipelineQuantizationConfig

from typing import List, Optional, Tuple, Union
import numpy as np
# from PIL import Image
from PIL import Image, ImageFilter, ImageOps

supported_text2image_models = [
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "kandinsky-community/kandinsky-2-2-decoder",
]

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
        self.add_dropdown_param("pretrained_model_name_or_path", supported_text2image_models)
        base_json = super().serialize()
        json_obj = json.loads(base_json)
        json_obj["pretrained_model_name_or_path"] = self.pretrained_model_name_or_path
        json_obj["torch_dtype"] = self.torch_dtype
        json_obj["mirror"] = self.mirror
        json_obj["low_cpu_mem_usage"] = self.low_cpu_mem_usage
        return json.dumps(json_obj, ensure_ascii=False, indent=2)
    
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
                
            # 调用基类反序列化
            return super().deserialize(json_str)
        except Exception as e:
            print(f"反序列化参数失败: {e}")
            return nndeploy.base.Status.error()
        
class QuantizationParam(nndeploy.base.Param):
    def __init__(self):
        super().__init__()
        self.is_quantized = False
        self.quant_backend = "bitsandbytes_4bit"
        self.quant_kwargs = {}
        self.components_to_quantize = ["transformer", "text_encoder"]
        self.quant_mapping = {}
        
    def serialize(self) -> str:
        """序列化参数为JSON字符串"""
        param_dict = {
            "is_quantized": self.is_quantized,
            "quant_backend": self.quant_backend,
            "quant_kwargs": self.quant_kwargs,
            "components_to_quantize": self.components_to_quantize,
            "quant_mapping": self.quant_mapping,
        }
        return json.dumps(param_dict, ensure_ascii=False, indent=2)
    
    def deserialize(self, json_str: str) -> nndeploy.base.Status:
        """从JSON字符串反序列化参数"""
        try:
            param_dict = json.loads(json_str)
            
            # 量化相关参数
            self.is_quantized = param_dict.get("is_quantized", False)
            self.quant_backend = param_dict.get("quant_backend", None)
            self.quant_kwargs = param_dict.get("quant_kwargs", {})
            self.components_to_quantize = param_dict.get("components_to_quantize", [])
            self.quant_mapping = param_dict.get("quant_mapping", {})
            
            return nndeploy.base.Status.ok()
        except Exception as e:
            print(f"反序列化量化参数失败: {e}")
            return nndeploy.base.Status.error()

    
class Text2Image(nndeploy.dag.Node):
    """
    基于nndeploy框架的Diffusers Pipeline节点
    
    该节点封装了Hugging Face Diffusers库的文本到图像生成功能，
    支持Stable Diffusion等多种扩散模型的推理。
    """
    def __init__(self, name: str, inputs: list[nndeploy.dag.Edge] = [], outputs: list[nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key("nndeploy.diffusion.Text2Image")
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
        # 低内存
        self.enable_model_cpu_offload = True
        # 内存高效注意力
        self.enable_xformers_memory_efficient_attention = False
        # 是否启用LoRA
        self.is_lora = False
        # LoRA权重
        self.lora_weights = ""
        # 量化
        self.quantization_param = QuantizationParam()
        
        # 管道实例
        self.pipeline = None
        
    def init(self):        
        try:
            quantization_config = None
            if self.quantization_param.is_quantized:
                quantization_config = PipelineQuantizationConfig(
                    quant_backend=self.quantization_param.quant_backend,
                    quant_kwargs=self.quantization_param.quant_kwargs,
                    components_to_quantize=self.quantization_param.components_to_quantize,
                    quant_mapping=self.quantization_param.quant_mapping
                )
            try:
                # 首先尝试本地加载
                self.pipeline = AutoPipelineForText2Image.from_pretrained(
                    pretrained_model_or_path=self.param.pretrained_model_name_or_path,
                    local_files_only=True,
                    torch_dtype=get_torch_dtype(self.param.torch_dtype),
                    low_cpu_mem_usage=self.param.low_cpu_mem_usage,
                    quantization_config=quantization_config
                )
            except Exception as local_error:                
                self.pipeline = AutoPipelineForText2Image.from_pretrained(
                    pretrained_model_or_path=self.param.pretrained_model_name_or_path,
                    torch_dtype=get_torch_dtype(self.param.torch_dtype),
                    mirror=self.param.mirror,
                    low_cpu_mem_usage=self.param.low_cpu_mem_usage,
                    quantization_config=quantization_config
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
                
            if self.is_lora:
                self.pipeline.load_lora_weights(self.lora_weights)
                
            if self.enable_model_cpu_offload:
                self.pipeline.enable_model_cpu_offload()
            if self.enable_xformers_memory_efficient_attention:
                try:
                    import xformers
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    print("已启用XFormers内存高效注意力")
                except ImportError:
                    print("警告: 未安装xformers，无法启用内存高效注意力。请参考 https://github.com/facebookresearch/xformers 安装xformers")
                except Exception as e:
                    print(f"启用XFormers内存高效注意力失败: {e}")
                
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
            json_obj["enable_model_cpu_offload"] = self.enable_model_cpu_offload
            json_obj["enable_xformers_memory_efficient_attention"] = self.enable_xformers_memory_efficient_attention
            json_obj["is_lora"] = self.is_lora
            json_obj["lora_weights"] = self.lora_weights
            json_obj["quantization_param"] = json.loads(self.quantization_param.serialize())
            
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
            if "enable_model_cpu_offload" in json_obj:
                self.enable_model_cpu_offload = json_obj["enable_model_cpu_offload"]
            if "enable_xformers_memory_efficient_attention" in json_obj:
                self.enable_xformers_memory_efficient_attention = json_obj["enable_xformers_memory_efficient_attention"]
            if "is_lora" in json_obj:
                self.is_lora = json_obj["is_lora"]
            if "lora_weights" in json_obj:
                self.lora_weights = json_obj["lora_weights"]
            if "quantization_param" in json_obj:
                quantization_param_json = json.dumps(json_obj["quantization_param"])
                self.quantization_param.deserialize(quantization_param_json)
            
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
nndeploy.dag.register_node("nndeploy.diffusion.Text2Image", text2image_creator)


class Image2Image(nndeploy.dag.Node):
    """
    基于nndeploy框架的Diffusers Pipeline图生图节点
    
    该节点封装了Hugging Face Diffusers库的图像到图像生成功能，
    支持Stable Diffusion等多种扩散模型的图像转换推理。
    """
    def __init__(self, name: str, inputs: list[nndeploy.dag.Edge] = [], outputs: list[nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key("nndeploy.diffusion.Image2Image")
        self.set_desc("Diffusers Pipeline for image-to-image generation")
        self.set_input_type(str, "Input text prompt")  # Input text prompt
        self.set_input_type(str, "Input negative text prompt")  # Input negative text prompt
        self.set_input_type(Image, "Input source image")  # Input source image
        self.set_output_type(Image, "Output PIL image")  # Output PIL image
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
        self.strength = 0.5  # 图生图特有参数，控制对原图的修改程度
        self.is_random = True  # 是否随机生成噪声
        self.generator_seed = 42  # 随机种子，None表示随机
        self.enable_model_cpu_offload = False  # 是否启用模型CPU卸载
        self.enable_xformers_memory_efficient_attention = False  # 是否启用XFormers内存高效注意力
        self.timesteps: List[int] = None,
        self.sigmas: List[float] = None,
        
        # 管道实例
        self.pipeline = None
        
    def init(self):        
        try:
            try:
                # 首先尝试本地加载
                self.pipeline = AutoPipelineForImage2Image.from_pretrained(
                    pretrained_model_or_path=self.param.pretrained_model_name_or_path,
                    local_files_only=True,
                    torch_dtype=get_torch_dtype(self.param.torch_dtype),
                    low_cpu_mem_usage=self.param.low_cpu_mem_usage
                )
            except Exception as local_error:                
                self.pipeline = AutoPipelineForImage2Image.from_pretrained(
                    pretrained_model_or_path=self.param.pretrained_model_name_or_path,
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
                
            if self.enable_model_cpu_offload:
                self.pipeline.enable_model_cpu_offload()
            if self.enable_xformers_memory_efficient_attention:
                try:
                    import xformers
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    print("已启用XFormers内存高效注意力")
                except ImportError:
                    print("警告: 未安装xformers，无法启用内存高效注意力。请参考 https://github.com/facebookresearch/xformers 安装xformers")
                except Exception as e:
                    print(f"启用XFormers内存高效注意力失败: {e}")
                
            return nndeploy.base.Status.ok()
            
        except Exception as e:
            print(f"初始化Diffusers图生图管道失败: {e}")
            return nndeploy.base.Status.error()
    
    def run(self) -> nndeploy.base.Status:
        """执行图像到图像生成"""
        try:
            # 获取输入文本提示词
            input_edge = self.get_input(0)
            prompt = input_edge.get(self)
            
            input_edge = self.get_input(1)
            negative_prompt = input_edge.get(self)
            
            input_edge = self.get_input(2)
            source_image = input_edge.get(self)
                        
            timesteps=None
            if self.timesteps != [None]:
                print(f"timesteps: {self.timesteps}")
                timesteps = self.timesteps
            sigmas = None
            if self.sigmas != [None]:
                print(f"sigmas: {self.sigmas}")
                sigmas = self.sigmas
            
            # 根据source_image类型确定num_images_per_prompt
            num_images_per_prompt = 1
            if source_image is not None:
                if isinstance(source_image, (list, tuple)):
                    num_images_per_prompt = len(source_image)
                elif isinstance(source_image, torch.Tensor) and len(source_image.shape) >= 4:
                    # 对于numpy数组或torch张量，假设第一个维度是batch维度
                    num_images_per_prompt = source_image.shape[0]
                else:
                    # 单个图像
                    num_images_per_prompt = 1
                    
            print(f"num_images_per_prompt: {num_images_per_prompt}")
            print(f"source_image: {source_image}")
            print("pipeline", self.pipeline)
            
            device_type = self.get_device_type()
            device = "cpu"
            if device_type.code_ == nndeploy.base.DeviceTypeCode.cuda:
                device_id = device_type.device_id_
                device = f"cuda:{device_id}"
            generator = None
            if not self.is_random:
                generator = torch.Generator(device=device)
                generator.manual_seed(self.generator_seed) 
                
            print("generator", generator)
                       
            result = self.pipeline(
                prompt=prompt,
                image=source_image,
                strength=self.strength,
                num_inference_steps=self.num_inference_steps,
                timesteps=timesteps,
                sigmas=sigmas,
                guidance_scale=self.guidance_scale,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                eta=self.eta,
                generator=generator,
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
            print(f"图生图推理失败: {e}")
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
            json_obj["strength"] = self.strength
            json_obj["timesteps"] = self.timesteps if self.timesteps is not None else List[int]
            json_obj["sigmas"] = self.sigmas if self.sigmas is not None else List[float]
            json_obj["is_random"] = self.is_random
            json_obj["generator_seed"] = self.generator_seed
            json_obj["enable_model_cpu_offload"] = self.enable_model_cpu_offload
            json_obj["enable_xformers_memory_efficient_attention"] = self.enable_xformers_memory_efficient_attention
            
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
            if "strength" in json_obj:
                self.strength = json_obj["strength"]
            if "timesteps" in json_obj:
                self.timesteps = json_obj["timesteps"]
            if "sigmas" in json_obj:
                self.sigmas = json_obj["sigmas"]
            if "is_random" in json_obj:
                self.is_random = json_obj["is_random"]
            if "generator_seed" in json_obj:
                self.generator_seed = json_obj["generator_seed"]
            if "enable_model_cpu_offload" in json_obj:
                self.enable_model_cpu_offload = json_obj["enable_model_cpu_offload"]
            if "enable_xformers_memory_efficient_attention" in json_obj:
                self.enable_xformers_memory_efficient_attention = json_obj["enable_xformers_memory_efficient_attention"]
            
            # 调用基类反序列化
            return super().deserialize(json_str)
            
        except Exception as e:
            print(f"反序列化失败: {e}")
            return nndeploy.base.Status.ok()


class Image2ImageCreator(nndeploy.dag.NodeCreator):
    """Diffusers Pipeline图生图节点创建器"""
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        """创建Diffusers Pipeline图生图节点实例"""
        self.node = Image2Image(name, inputs, outputs)
        return self.node


# 注册节点创建器
image2image_creator = Image2ImageCreator()
nndeploy.dag.register_node("nndeploy.diffusion.Image2Image", image2image_creator)


class Inpainting(nndeploy.dag.Node):
    """
    基于nndeploy框架的Diffusers Pipeline图像修复节点
    
    该节点封装了Hugging Face Diffusers库的图像修复功能，
    支持Stable Diffusion等多种扩散模型的图像修复推理。
    """
    def __init__(self, name: str, inputs: list[nndeploy.dag.Edge] = [], outputs: list[nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key("nndeploy.diffusion.Inpainting")
        self.set_desc("Diffusers Pipeline for image inpainting")
        self.set_input_type(str, "Input text prompt")  # Input text prompt
        self.set_input_type(str, "Input negative text prompt")  # Input negative text prompt
        self.set_input_type(Image, "Input source image")  # Input source image
        self.set_input_type(Image, "Input mask image")  # Input mask image
        self.set_output_type(Image, "Output PIL image")  # Output PIL image
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
        self.strength = 0.8  # 修复强度，控制对原图的修改程度
        self.timesteps: List[int] = None,
        self.sigmas: List[float] = None,
        self.is_random = True  # 是否随机生成噪声
        self.generator_seed = 42  # 随机种子，None表示随机
        self.enable_model_cpu_offload = False  # 是否启用模型CPU卸载
        self.enable_xformers_memory_efficient_attention = False  # 是否启用XFormers内存高效注意力
        
        # 管道实例
        self.pipeline = None
        
    def init(self):        
        try:
            try:
                self.pipeline = AutoPipelineForInpainting.from_pretrained(
                    pretrained_model_or_path=self.param.pretrained_model_name_or_path,
                    local_files_only=True,
                    torch_dtype=get_torch_dtype(self.param.torch_dtype),
                    low_cpu_mem_usage=self.param.low_cpu_mem_usage
                )
            except Exception as local_error:                
                self.pipeline = AutoPipelineForInpainting.from_pretrained(
                    pretrained_model_or_path=self.param.pretrained_model_name_or_path,
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
                
            if self.enable_model_cpu_offload:
                self.pipeline.enable_model_cpu_offload()
            if self.enable_xformers_memory_efficient_attention:
                try:
                    import xformers
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    print("已启用XFormers内存高效注意力")
                except ImportError:
                    print("警告: 未安装xformers，无法启用内存高效注意力。请参考 https://github.com/facebookresearch/xformers 安装xformers")
                except Exception as e:
                    print(f"启用XFormers内存高效注意力失败: {e}")
                
            return nndeploy.base.Status.ok()
            
        except Exception as e:
            print(f"初始化Diffusers图像修复管道失败: {e}")
            return nndeploy.base.Status.error()
    
    def run(self) -> nndeploy.base.Status:
        """执行图像修复"""
        try:
            # 获取输入文本提示词
            input_edge = self.get_input(0)
            prompt = input_edge.get(self)
            
            input_edge = self.get_input(1)
            negative_prompt = input_edge.get(self)
            
            input_edge = self.get_input(2)
            source_image = input_edge.get(self)
            
            input_edge = self.get_input(3)
            mask_image = input_edge.get(self)
                        
            timesteps=None
            if self.timesteps != [None]:
                print(f"timesteps: {self.timesteps}")
                timesteps = self.timesteps
            sigmas = None
            if self.sigmas != [None]:
                print(f"sigmas: {self.sigmas}")
                sigmas = self.sigmas
            
            # 根据source_image类型确定num_images_per_prompt
            num_images_per_prompt = 1
            if source_image is not None:
                if isinstance(source_image, (list, tuple)):
                    num_images_per_prompt = len(source_image)
                elif isinstance(source_image, torch.Tensor) and len(source_image.shape) >= 4:
                    num_images_per_prompt = source_image.shape[0]
                    
            print(f"num_images_per_prompt: {num_images_per_prompt}")
            print(f"source_image: {source_image}")
            print("pipeline", self.pipeline)
            
            device_type = self.get_device_type()
            device = "cpu"
            if device_type.code_ == nndeploy.base.DeviceTypeCode.cuda:
                device_id = device_type.device_id_
                device = f"cuda:{device_id}"
            generator = None
            if not self.is_random:
                generator = torch.Generator(device=device)
                generator.manual_seed(self.generator_seed)
                
            print("generator", generator)
            
            # 执行图像修复推理
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=source_image,
                mask_image=mask_image,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                eta=self.eta,
                guidance_rescale=self.guidance_rescale,
                strength=self.strength,
                timesteps=timesteps,
                sigmas=sigmas,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator
            )
            
            # result = self.pipeline(
            #     prompt=prompt,
            #     negative_prompt=negative_prompt,
            #     image=source_image,
            #     mask_image=mask_image
            # )
            
            # 输出到输出边
            min_len = min(len(result.images), len(self.get_all_output()))
            for i in range(min_len):
                output_edge = self.get_output(i)
                generated_image = result.images[i]
                output_edge.set(generated_image)
            
            return nndeploy.base.Status.ok()
            
        except Exception as e:
            print(f"图像修复推理失败: {e}")
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
            json_obj["strength"] = self.strength
            json_obj["timesteps"] = self.timesteps if self.timesteps is not None else List[int]
            json_obj["sigmas"] = self.sigmas if self.sigmas is not None else List[float]
            json_obj["is_random"] = self.is_random
            json_obj["generator_seed"] = self.generator_seed
            json_obj["enable_model_cpu_offload"] = self.enable_model_cpu_offload
            json_obj["enable_xformers_memory_efficient_attention"] = self.enable_xformers_memory_efficient_attention
            
             # 添加自定义参数
            json_obj["param"] = json.loads(self.param.serialize())
            
            return json.dumps(json_obj, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"序列化失败: {e}")
            return super().serialize()
    
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
            if "strength" in json_obj:
                self.strength = json_obj["strength"]
            if "timesteps" in json_obj:
                self.timesteps = json_obj["timesteps"]
            if "sigmas" in json_obj:
                self.sigmas = json_obj["sigmas"]
            if "is_random" in json_obj:
                self.is_random = json_obj["is_random"]
            if "generator_seed" in json_obj:
                self.generator_seed = json_obj["generator_seed"]
            if "enable_model_cpu_offload" in json_obj:
                self.enable_model_cpu_offload = json_obj["enable_model_cpu_offload"]
            if "enable_xformers_memory_efficient_attention" in json_obj:
                self.enable_xformers_memory_efficient_attention = json_obj["enable_xformers_memory_efficient_attention"]
            
            # 调用基类反序列化
            return super().deserialize(json_str)
            
        except Exception as e:
            print(f"反序列化失败: {e}")
            return nndeploy.base.Status.ok()

class InpaintingCreator(nndeploy.dag.NodeCreator):
    """Diffusers Pipeline图像修复节点创建器"""
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        """创建Diffusers Pipeline图像修复节点实例"""
        self.node = Inpainting(name, inputs, outputs)
        return self.node

# 注册节点创建器
inpainting_creator = InpaintingCreator()
nndeploy.dag.register_node("nndeploy.diffusion.Inpainting", inpainting_creator)

