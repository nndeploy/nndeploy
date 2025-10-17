import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.device
import nndeploy.dag
from nndeploy.base import get_torch_dtype

import json
import torch
import numpy as np
from typing import Optional, Tuple, Union

from PIL import Image

class LatentNoise(nndeploy.dag.Node):
    """
    潜在空间噪声生成输入节点
    
    该节点用于生成扩散模型所需的初始噪声张量，
    所有参数都内置在节点中，无需外部输入。
    """
    def __init__(self, name: str, inputs: list[nndeploy.dag.Edge] = [], outputs: list[nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key("nndeploy.diffusion.LatentNoise")
        self.set_desc("Generate random noise tensor for diffusion models")
        
        # 输出：噪声张量
        self.set_output_type(torch.Tensor, "Output latent noise tensor")
        
        # 内置参数
        self.height = 512
        self.width = 512
        self.batch_size = 1
        self.num_channels_latents = 4  # Stable Diffusion的潜在空间通道数
        self.is_random = True  # 是否随机生成噪声
        self.generator_seed = 42  # 随机种子，None表示随机
        self.noise_scale = 1.0  # 噪声缩放因子
        self.dtype = "float16"  # 数据类型
        
        # 设置默认设备为CUDA
        self.set_device_type(nndeploy.base.DeviceType(nndeploy.base.DeviceTypeCode.cuda, 0))
        
    def init(self):
        """初始化节点"""
        try:
            return nndeploy.base.Status.ok()
        except Exception as e:
            print(f"初始化潜在空间噪声节点失败: {e}")
            return nndeploy.base.Status.error()
    
    def run(self) -> nndeploy.base.Status:
        """生成潜在空间噪声"""
        try:
            # 计算潜在空间尺寸（通常是图像尺寸的1/8）
            latent_height = self.height // 8
            latent_width = self.width // 8
            
            # 确定设备
            device_type = self.get_device_type()
            if device_type.code_ == nndeploy.base.DeviceTypeCode.cuda:
                device_id = device_type.device_id_
                device = f"cuda:{device_id}"
            else:
                device = "cpu"
            
            # 设置随机种子
            generator = None
            if not self.is_random:
                generator = torch.Generator(device=device)
                generator.manual_seed(self.generator_seed)
            
            # 生成噪声张量
            noise_shape = (
                self.batch_size,
                self.num_channels_latents,
                latent_height,
                latent_width
            )
            
            dtype = get_torch_dtype(self.dtype)
            
            # 生成标准高斯噪声
            latent_noise = torch.randn(
                noise_shape,
                generator=generator,
                device=device,
                dtype=dtype
            )
            
            # 应用噪声缩放
            latent_noise = latent_noise * self.noise_scale
            
            # 输出噪声张量
            output_edge = self.get_output(0)
            output_edge.set(latent_noise)
            
            return nndeploy.base.Status.ok()
            
        except Exception as e:
            print(f"生成潜在空间噪声失败: {e}")
            return nndeploy.base.Status.error()
    
    def serialize(self) -> str:
        """序列化节点参数"""
        try:
            # 获取基类序列化结果
            base_json = super().serialize()
            json_obj = json.loads(base_json)
            
            # 添加内置参数
            json_obj["height"] = self.height
            json_obj["width"] = self.width
            json_obj["batch_size"] = self.batch_size
            json_obj["num_channels_latents"] = self.num_channels_latents
            json_obj["is_random"] = self.is_random
            json_obj["generator_seed"] = self.generator_seed
            json_obj["noise_scale"] = self.noise_scale
            json_obj["dtype"] = self.dtype
            
            return json.dumps(json_obj, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"序列化潜在空间噪声节点失败: {e}")
            return "{}"
    
    def deserialize(self, json_str: str) -> nndeploy.base.Status:
        """反序列化节点参数"""
        try:
            # 调用基类反序列化
            super().deserialize(json_str)
            
            # 解析JSON
            json_obj = json.loads(json_str)
            
            # 反序列化内置参数
            self.height = json_obj.get("height", 512)
            self.width = json_obj.get("width", 512)
            self.batch_size = json_obj.get("batch_size", 1)
            self.num_channels_latents = json_obj.get("num_channels_latents", 4)
            self.is_random = json_obj.get("is_random", True)
            self.generator_seed = json_obj.get("generator_seed", 42)
            self.noise_scale = json_obj.get("noise_scale", 1.0)
            self.dtype = json_obj.get("dtype", "float16")
            
            return nndeploy.base.Status.ok()
        except Exception as e:
            return nndeploy.base.Status.error()
          
class LatentNoiseCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = LatentNoise(name, inputs, outputs)
        return self.node

latent_noise_creator = LatentNoiseCreator()
nndeploy.dag.register_node("nndeploy.diffusion.LatentNoise", latent_noise_creator)

class LatentEmpty(nndeploy.dag.Node):
    """
    空潜在空间输入节点
    
    该节点用于生成全零的潜在空间张量，
    所有参数都内置在节点中。
    """
    def __init__(self, name: str, inputs: list[nndeploy.dag.Edge] = [], outputs: list[nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key("nndeploy.diffusion.LatentEmpty")
        self.set_desc("Generate empty (zero) latent tensor")
        
        # 输出：空潜在空间张量
        self.set_output_type(torch.Tensor, "Output empty latent tensor")
        
        # 内置参数
        self.height = 512
        self.width = 512
        self.batch_size = 1
        self.num_channels_latents = 4
        self.dtype = "float16"
        
        # 设置默认设备为CUDA
        self.set_device_type(nndeploy.base.DeviceType(nndeploy.base.DeviceTypeCode.cuda, 0))
    
    def init(self):
        """初始化节点"""
        try:
            return nndeploy.base.Status.ok()
        except Exception as e:
            print(f"初始化空潜在空间节点失败: {e}")
            return nndeploy.base.Status.error()
    
    def run(self) -> nndeploy.base.Status:
        """生成空潜在空间张量"""
        try:
            # 计算潜在空间尺寸
            latent_height = self.height // 8
            latent_width = self.width // 8
            
            # 确定设备
            device_type = self.get_device_type()
            if device_type.code_ == nndeploy.base.DeviceTypeCode.cuda:
                device_id = device_type.device_id_
                device = f"cuda:{device_id}"
            else:
                device = "cpu"
            
            # 生成空张量
            empty_shape = (
                self.batch_size,
                self.num_channels_latents,
                latent_height,
                latent_width
            )
            
            dtype = get_torch_dtype(self.dtype)
            
            # 生成全零张量
            empty_latent = torch.zeros(
                empty_shape,
                device=device,
                dtype=dtype
            )
            
            # 输出空张量
            output_edge = self.get_output(0)
            output_edge.set(empty_latent)
            
            return nndeploy.base.Status.ok()
            
        except Exception as e:
            print(f"生成空潜在空间张量失败: {e}")
            return nndeploy.base.Status.error()
    
    def serialize(self) -> str:
        """序列化节点参数"""
        try:
            base_json = super().serialize()
            json_obj = json.loads(base_json)
            
            json_obj["height"] = self.height
            json_obj["width"] = self.width
            json_obj["batch_size"] = self.batch_size
            json_obj["num_channels_latents"] = self.num_channels_latents
            json_obj["dtype"] = self.dtype
            
            return json.dumps(json_obj, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"序列化空潜在空间节点失败: {e}")
            return "{}"
    
    def deserialize(self, json_str: str) -> nndeploy.base.Status:
        """反序列化节点参数"""
        try:
            super().deserialize(json_str)
            
            json_obj = json.loads(json_str)
            
            self.height = json_obj.get("height", 512)
            self.width = json_obj.get("width", 512)
            self.batch_size = json_obj.get("batch_size", 1)
            self.num_channels_latents = json_obj.get("num_channels_latents", 4)
            self.dtype = json_obj.get("dtype", "float16")
            
            return nndeploy.base.Status.ok()
        except Exception as e:
            return nndeploy.base.Status.error()

class LatentEmptyCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = LatentEmpty(name, inputs, outputs)
        return self.node

latent_empty_creator = LatentEmptyCreator()
nndeploy.dag.register_node("nndeploy.diffusion.LatentEmpty", latent_empty_creator)

class LatentFromImage(nndeploy.dag.Node):
    """
    从图像生成潜在空间输入节点
    
    该节点用于将输入图像编码为潜在空间表示，
    编码参数都内置在节点中。
    """
    def __init__(self, name: str, inputs: list[nndeploy.dag.Edge] = [], outputs: list[nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key("nndeploy.diffusion.LatentFromImage")
        self.set_desc("Encode image to latent space representation")
        
        # 输入：PIL图像
        
        self.set_input_type(Image.Image, "Input PIL image")
        
        # 输出：潜在空间张量
        self.set_output_type(torch.Tensor, "Output latent tensor")
        
        # 内置参数
        self.vae_scale_factor = 0.18215  # VAE缩放因子
        self.dtype = "float16"
        
        # 设置默认设备为CUDA
        self.set_device_type(nndeploy.base.DeviceType(nndeploy.base.DeviceTypeCode.cuda, 0))
    
    def init(self):
        """初始化节点"""
        try:
            return nndeploy.base.Status.ok()
        except Exception as e:
            print(f"初始化图像到潜在空间节点失败: {e}")
            return nndeploy.base.Status.error()
    
    def run(self) -> nndeploy.base.Status:
        """将图像编码为潜在空间"""
        try:
            # 获取输入PIL图像
            input_edge = self.get_input(0)
            pil_image = input_edge.get(self)
            
            # 确定设备
            device_type = self.get_device_type()
            if device_type.code_ == nndeploy.base.DeviceTypeCode.cuda:
                device_id = device_type.device_id_
                device = f"cuda:{device_id}"
            else:
                device = "cpu"
            
            # 将PIL图像转换为张量
            import torchvision.transforms as transforms
            
            # 定义转换管道
            transform = transforms.Compose([
                transforms.ToTensor(),  # 转换为张量并归一化到[0,1]
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1,1]
            ])
            
            # 转换PIL图像为张量
            image_tensor = transform(pil_image).unsqueeze(0)  # 添加批次维度
            
            # 确保图像在正确的设备上
            if image_tensor.device != torch.device(device):
                image_tensor = image_tensor.to(device)
            
            # 确保数据类型正确
            dtype = get_torch_dtype(self.dtype)
            if image_tensor.dtype != dtype:
                image_tensor = image_tensor.to(dtype)
            
            # 简化的潜在空间编码（实际应用中需要VAE编码器）
            # 这里使用下采样模拟潜在空间编码
            import torch.nn.functional as F
            
            # 下采样到1/8尺寸
            latent = F.avg_pool2d(image_tensor, kernel_size=8, stride=8)
            
            # 应用VAE缩放因子
            latent = latent * self.vae_scale_factor
            
            # 输出潜在空间张量
            output_edge = self.get_output(0)
            output_edge.set(latent)
            
            return nndeploy.base.Status.ok()
            
        except Exception as e:
            print(f"图像编码为潜在空间失败: {e}")
            return nndeploy.base.Status.error()
    
    def serialize(self) -> str:
        """序列化节点参数"""
        try:
            base_json = super().serialize()
            json_obj = json.loads(base_json)
            
            json_obj["vae_scale_factor"] = self.vae_scale_factor
            json_obj["dtype"] = self.dtype
            
            return json.dumps(json_obj, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"序列化图像到潜在空间节点失败: {e}")
            return "{}"
    
    def deserialize(self, json_str: str) -> nndeploy.base.Status:
        """反序列化节点参数"""
        try:
            super().deserialize(json_str)
            
            json_obj = json.loads(json_str)
            
            self.vae_scale_factor = json_obj.get("vae_scale_factor", 0.18215)
            self.dtype = json_obj.get("dtype", "float16")
            
            return nndeploy.base.Status.ok()
        except Exception as e:
            return nndeploy.base.Status.error()

class LatentFromImageCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = LatentFromImage(name, inputs, outputs)
        return self.node

latent_from_image_creator = LatentFromImageCreator()
nndeploy.dag.register_node("nndeploy.diffusion.LatentFromImage", latent_from_image_creator)

class LatentBatch(nndeploy.dag.Node):
    """
    潜在空间批处理输入节点
    
    该节点用于将多个潜在空间张量组合成批次，
    批处理参数都内置在节点中。
    """
    def __init__(self, name: str, inputs: list[nndeploy.dag.Edge] = [], outputs: list[nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key("nndeploy.diffusion.LatentBatch")
        self.set_desc("Batch multiple latent tensors together")
        
        # 输入：多个潜在空间张量
        self.set_input_type(torch.Tensor, "Input latent tensor 1")
        self.set_input_type(torch.Tensor, "Input latent tensor 2")  
        self.set_dynamic_input(True)
        
        # 输出：批处理后的潜在空间张量
        self.set_output_type(torch.Tensor, "Output batched latent tensor")
        
        # 内置参数
        self.max_batch_size = 4  # 最大批次大小
        self.dtype = "float16"
        
        # 设置默认设备为CUDA
        self.set_device_type(nndeploy.base.DeviceType(nndeploy.base.DeviceTypeCode.cuda, 0))
    
    def init(self):
        """初始化节点"""
        try:
            return nndeploy.base.Status.ok()
        except Exception as e:
            print(f"初始化潜在空间批处理节点失败: {e}")
            return nndeploy.base.Status.error()
    
    def run(self) -> nndeploy.base.Status:
        """执行潜在空间批处理"""
        try:
            # 收集所有有效的输入张量
            latent_tensors = []
            
            for i in len(self.get_all_input()):
                input_edge = self.get_input(i)
                latent = input_edge.get(self)
                if latent is not None:
                    latent_tensors.append(latent)
            
            if not latent_tensors:
                print("没有有效的潜在空间张量输入")
                return nndeploy.base.Status.error()
            
            # 确定设备
            device_type = self.get_device_type()
            if device_type.code_ == nndeploy.base.DeviceTypeCode.cuda:
                device_id = device_type.device_id_
                device = f"cuda:{device_id}"
            else:
                device = "cpu"
            
            # 确保所有张量在同一设备上
            dtype = get_torch_dtype(self.dtype)
            for i, tensor in enumerate(latent_tensors):
                if tensor.device != torch.device(device):
                    latent_tensors[i] = tensor.to(device)
                if tensor.dtype != dtype:
                    latent_tensors[i] = latent_tensors[i].to(dtype)
            
            # 沿批次维度拼接张量
            batched_latent = torch.cat(latent_tensors, dim=0)
            
            # 输出批处理结果
            output_edge = self.get_output(0)
            output_edge.set(batched_latent)
            
            return nndeploy.base.Status.ok()
            
        except Exception as e:
            print(f"潜在空间批处理失败: {e}")
            return nndeploy.base.Status.error()
    
    def serialize(self) -> str:
        """序列化节点参数"""
        try:
            base_json = super().serialize()
            json_obj = json.loads(base_json)
            
            json_obj["dtype"] = self.dtype
            
            return json.dumps(json_obj, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"序列化潜在空间批处理节点失败: {e}")
            return "{}"
    
    def deserialize(self, json_str: str) -> nndeploy.base.Status:
        """反序列化节点参数"""
        try:
            super().deserialize(json_str)
            
            json_obj = json.loads(json_str)
            
            self.dtype = json_obj.get("dtype", "float16")
            
            return nndeploy.base.Status.ok()
        except Exception as e:
            return nndeploy.base.Status.error()

class LatentBatchCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        self.node = LatentBatch(name, inputs, outputs)
        return self.node

latent_batch_creator = LatentBatchCreator()
nndeploy.dag.register_node("nndeploy.diffusion.LatentBatch", latent_batch_creator)