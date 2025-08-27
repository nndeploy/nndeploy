# import torch
# import time
# from diffusers import DiffusionPipeline

# from typing import Dict, Any, List, Optional
# import numpy as np

# import nndeploy._nndeploy_internal as _C

# import nndeploy.base
# import nndeploy.device
# import nndeploy.dag

# diffuser_model_ids = []

# def add_diffuser_model_id(model_id: str):
#     diffuser_model_ids.append(model_id)
    
# def remove_diffuser_model_id(model_id: str):
#     diffuser_model_ids.remove(model_id)

# class DiffusionPipelineNode(Node):
#     """基于nndeploy节点封装的DiffusionPipeline"""
#     def __init__(self, name, inputs: [nndeploy.dag.Edge] = [], outputs: [nndeploy.dag.Edge] = []):
#         super().__init__(name, inputs, outputs)
#         self.set_key(type(self).__name__)
#         self.set_input_type(str)
#         self.set_output_type(np.ndarray)
        
#         self.diffuser_model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
#         self.data_type_ = nndeploy.base.DataType.kDataTypeFloat16
#         self.device_type_ = nndeploy.base.DeviceType.kDeviceTypeCodeCuda
#         self.pipeline = None
        
#     def init(self) -> bool:
#         """初始化Diffusion Pipeline"""
#         try:
#             print(f"正在加载模型: {self.model_id}")
#             start_time = time.time()
            
#             self.pipeline = DiffusionPipeline.from_pretrained(
#                 self.model_id, 
#                 torch_dtype=self.torch_dtype
#             )
            
#             # 检查设备类型并移动到相应设备
#             device_type = self.get_device_type()
#             if device_type == DeviceType.kDeviceTypeCodeCuda:
#                 self.pipeline.to("cuda")
#             elif device_type == DeviceType.kDeviceTypeCpu:
#                 self.pipeline.to("cpu")
            
#             end_time = time.time()
#             print(f"模型加载完成，耗时: {end_time - start_time:.2f} 秒")
#             return True
            
#         except Exception as e:
#             print(f"模型加载失败: {e}")
#             return False
    
#     def run(self) -> bool:
#         """执行推理"""
#         try:
#             # 获取输入提示词
#             input_tensor = self.get_input("prompt")
#             if input_tensor is None:
#                 print("未找到输入提示词")
#                 return False
            
#             prompt = input_tensor.get_data_as_string()
            
#             print(f"开始生成图像，提示词: {prompt}")
#             start_time = time.time()
            
#             # 执行推理
#             result = self.pipeline(prompt)
#             images = result.images
            
#             end_time = time.time()
#             print(f"图像生成完成，耗时: {end_time - start_time:.2f} 秒")
            
#             # 将结果设置到输出张量
#             if images and len(images) > 0:
#                 output_tensor = self.get_output("image")
#                 if output_tensor is not None:
#                     # 这里需要根据具体的图像格式进行转换
#                     # 暂时存储PIL图像对象的引用
#                     output_tensor.set_data(images[0])
#                     return True
            
#             return False
            
#         except Exception as e:
#             print(f"推理执行失败: {e}")
#             return False


# def create_diffusion_pipeline_node(name: str = "diffusion_pipeline",
#                                   model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
#                                   torch_dtype=torch.float16) -> DiffusionPipelineNode:
#     """创建Diffusion Pipeline节点的工厂函数"""
#     return DiffusionPipelineNode(name, model_id, torch_dtype)
