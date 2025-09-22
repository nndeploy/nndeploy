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
from PIL import Image, ImageFilter, ImageOps

from .diffusers_info.pretrain_model_paths import get_text2image_pipelines_pretrained_model_paths
from .diffusers_info.pretrain_model_paths import get_image2image_pipelines_pretrained_model_paths
from .diffusers_info.pretrain_model_paths import get_inpainting_pipelines_pretrained_model_paths

from diffusers.utils import logging

logging.set_verbosity_info()

def get_scheduler(pipeline, scheduler, scheduler_kwargs):
    if scheduler == "default":
        return pipeline.scheduler

class Text2Image(nndeploy.dag.Node):
    """
    Diffusers Pipeline node based on the nndeploy framework.

    This node wraps the Hugging Face Diffusers library for text-to-image generation,
    supporting inference for Stable Diffusion and other diffusion models.
    """
    def __init__(self, name: str, inputs: list[nndeploy.dag.Edge] = [], outputs: list[nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key("nndeploy.diffusion.Text2Image")
        self.set_desc("Diffusers Pipeline for text-to-image generation")
        self.set_input_type(str, "Input text prompt")
        self.set_input_type(str, "Input negative text prompt")
        self.set_input_type(torch.Tensor, "Input latent")
        self.set_output_type(Image, "Output PIL image")
        self.set_dynamic_output(True)
        
        # Default device is CUDA
        self.set_device_type(nndeploy.base.DeviceType(nndeploy.base.DeviceTypeCode.cuda, 0))
        # Model loading parameters
        self.pretrained_model_name_or_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        self.torch_dtype = "float16"
        self.use_safetensors = True
        
        # Runtime parameters
        self.num_inference_steps = 50
        self.guidance_scale = 8.0
        self.guidance_rescale = 0.0
        self.scheduler = "default"
        self.scheduler_kwargs = {"key": "value"}
        
        # Memory optimization
        self.enable_model_cpu_offload = True
        self.enable_sequential_cpu_offload = False
        self.enable_xformers_memory_efficient_attention = False
        
        # Pipeline instance
        self.pipeline = None
        
    def init(self):        
        try:
            try:
                # Try to load locally first
                self.pipeline = AutoPipelineForText2Image.from_pretrained(
                    pretrained_model_or_path=self.pretrained_model_name_or_path,
                    local_files_only=True,
                    torch_dtype=get_torch_dtype(self.torch_dtype),
                    use_safetensors=self.use_safetensors,
                )
            except Exception as local_error:                
                self.pipeline = AutoPipelineForText2Image.from_pretrained(
                    pretrained_model_or_path=self.pretrained_model_name_or_path,
                    torch_dtype=get_torch_dtype(self.torch_dtype),
                    use_safetensors=self.use_safetensors,
                )
                
            print("Text2Image pipeline:", self.pipeline)
            
            # Memory optimization
            if self.enable_sequential_cpu_offload:
                # self.pipeline.reset_device_map()
                try:
                    self.pipeline.enable_sequential_cpu_offload()
                    self.enable_model_cpu_offload = False 
                except ImportError as e:
                    print("Accelerate library needs to be upgraded: pip install accelerate>=0.14.0")
                    self.enable_model_cpu_offload = True  
                except RuntimeError as e:
                    print("No available accelerator device found")
                    self.enable_model_cpu_offload = True  
                except ValueError as e:
                    print(f"Configuration conflict: {e}")
                    self.enable_model_cpu_offload = True         
            
            # Move pipeline to the appropriate device
            if not self.enable_sequential_cpu_offload:
                device_type = self.get_device_type()
                if device_type.code_ == nndeploy.base.DeviceTypeCode.cuda:
                    device_id = device_type.device_id_
                    target_device = f"cuda:{device_id}"
                    self.pipeline.to(target_device)
                elif device_type.code_ == nndeploy.base.DeviceTypeCode.cpu:
                    self.pipeline.to("cpu")
                
            # More memory optimization    
            if self.enable_model_cpu_offload:
                self.pipeline.enable_model_cpu_offload()
            if self.enable_xformers_memory_efficient_attention:
                try:
                    import xformers
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    print("XFormers memory efficient attention enabled.")
                except ImportError:
                    print("Warning: xformers is not installed. Cannot enable memory efficient attention. Please refer to https://github.com/facebookresearch/xformers for installation.")
                except Exception as e:
                    print(f"Failed to enable XFormers memory efficient attention: {e}")
                
            return nndeploy.base.Status.ok()   
        except Exception as e:
            print(f"Failed to initialize Diffusers pipeline: {e}")
            return nndeploy.base.Status.error()
    
    def run(self) -> nndeploy.base.Status:
        """Run text-to-image generation"""
        try:
            if self.scheduler != "default":
                self.pipeline.scheduler = get_scheduler(self.pipeline, self.scheduler, self.scheduler_kwargs)
            
            # Get input text prompt
            input_edge = self.get_input(0)
            prompt = input_edge.get(self)
            
            input_edge = self.get_input(1)
            negative_prompt = input_edge.get(self)
            
            input_edge = self.get_input(2)
            latent = input_edge.get(self)
            
            num_images_per_prompt = latent.shape[0]
            
            # 检查pipeline是否支持guidance_rescale参数
            import inspect
            pipeline_call_signature = inspect.signature(self.pipeline.__call__)
            supports_guidance_rescale = 'guidance_rescale' in pipeline_call_signature.parameters
            if supports_guidance_rescale:
                # print("Pipeline supports guidance_rescale parameter")
                result = self.pipeline(
                    prompt=prompt,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    latents=latent,
                    guidance_rescale=self.guidance_rescale
                )
            else:
                # print("Pipeline does not support guidance_rescale parameter")
                result = self.pipeline(
                    prompt=prompt,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    latents=latent,
                )
                            
            # Set output to output edges
            min_len = min(len(result.images), len(self.get_all_output()))
            for i in range(min_len):
                output_edge = self.get_output(i)
                generated_image = result.images[i]
                output_edge.set(generated_image)
            
            return nndeploy.base.Status.ok()            
        except Exception as e:
            return nndeploy.base.Status.error()
    
    def serialize(self) -> str:
        """Serialize node parameters"""
        try:
            # Add required parameters
            self.add_required_param("pretrained_model_name_or_path")
            self.add_dropdown_param("pretrained_model_name_or_path", get_text2image_pipelines_pretrained_model_paths())
            # Get base class serialization
            base_json = super().serialize()
            json_obj = json.loads(base_json)
            
            # Serialize parameters
            # Model loading
            json_obj["pretrained_model_name_or_path"] = self.pretrained_model_name_or_path
            json_obj["torch_dtype"] = self.torch_dtype
            json_obj["use_safetensors"] = self.use_safetensors
            # Runtime
            json_obj["num_inference_steps"] = self.num_inference_steps
            json_obj["guidance_scale"] = self.guidance_scale
            json_obj["guidance_rescale"] = self.guidance_rescale
            json_obj["scheduler"] = self.scheduler
            json_obj["scheduler_kwargs"] = self.scheduler_kwargs
            # Memory optimization
            json_obj["enable_model_cpu_offload"] = self.enable_model_cpu_offload
            json_obj["enable_sequential_cpu_offload"] = self.enable_sequential_cpu_offload
            json_obj["enable_xformers_memory_efficient_attention"] = self.enable_xformers_memory_efficient_attention
            
            return json.dumps(json_obj, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"Serialization failed: {e}")
            return "{}"
    
    def deserialize(self, json_str: str) -> nndeploy.base.Status:
        """Deserialize node parameters"""
        try:
            json_obj = json.loads(json_str)
            
            # Deserialize parameters, with default values for missing keys
            self.pretrained_model_name_or_path = json_obj.get("pretrained_model_name_or_path", "")
            self.torch_dtype = json_obj.get("torch_dtype", "float16")
            self.use_safetensors = json_obj.get("use_safetensors", True)
            self.num_inference_steps = json_obj.get("num_inference_steps", 50)
            self.guidance_scale = json_obj.get("guidance_scale", 8.0)
            self.guidance_rescale = json_obj.get("guidance_rescale", 0.0)
            self.scheduler = json_obj.get("scheduler", "default")
            self.scheduler_kwargs = json_obj.get("scheduler_kwargs", {"key": "value"})
            self.enable_model_cpu_offload = json_obj.get("enable_model_cpu_offload", True)
            self.enable_sequential_cpu_offload = json_obj.get("enable_sequential_cpu_offload", False)
            self.enable_xformers_memory_efficient_attention = json_obj.get("enable_xformers_memory_efficient_attention", False)
            
            # Call base class deserialization
            return super().deserialize(json_str)
            
        except Exception as e:
            print(f"Deserialization failed: {e}")
            return nndeploy.base.Status.ok()

class Text2ImageCreator(nndeploy.dag.NodeCreator):
    """Diffusers Pipeline node creator"""
    def __init__(self):
        super().__init__()
        
    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        """Create Diffusers Pipeline node instance"""
        self.node = Text2Image(name, inputs, outputs)
        return self.node

# Register node creator
text2image_creator = Text2ImageCreator()
nndeploy.dag.register_node("nndeploy.diffusion.Text2Image", text2image_creator)


class Image2Image(nndeploy.dag.Node):
    """
    Diffusers Pipeline node for image-to-image generation based on the nndeploy framework.

    This node wraps the image-to-image generation functionality of the Hugging Face Diffusers library,
    supporting Stable Diffusion and other diffusion models for image transformation inference.
    """
    def __init__(self, name: str, inputs: list[nndeploy.dag.Edge] = [], outputs: list[nndeploy.dag.Edge] = []):
        super().__init__(name, inputs, outputs)
        self.set_key("nndeploy.diffusion.Image2Image")
        self.set_desc("Diffusers Pipeline for image-to-image generation")
        self.set_input_type(str, "Input text prompt")
        self.set_input_type(str, "Input negative text prompt")
        self.set_input_type(Image, "Input source image")
        self.set_output_type(Image, "Output PIL image")
        self.set_dynamic_output(True)

        # Set default device to CUDA
        self.set_device_type(nndeploy.base.DeviceType(nndeploy.base.DeviceTypeCode.cuda, 0))

        # Parameters
        self.pretrained_model_name_or_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        self.torch_dtype = "float16"
        self.use_safetensors = True
        self.num_inference_steps = 50
        self.guidance_scale = 8.0
        self.guidance_rescale = 0.0
        self.scheduler = "default"
        self.scheduler_kwargs = {"key": "value"}
        self.strength = 0.8  # Controls the degree of modification to the original image
        self.is_random = True  # Whether to generate noise randomly
        self.generator_seed = 42  # Random seed, None means random
        self.enable_model_cpu_offload = False
        self.enable_sequential_cpu_offload = False
        self.enable_xformers_memory_efficient_attention = False

        # Pipeline instance
        self.pipeline = None

    def init(self):
        try:
            try:
                # Try to load locally first
                self.pipeline = AutoPipelineForImage2Image.from_pretrained(
                    pretrained_model_or_path=self.pretrained_model_name_or_path,
                    local_files_only=True,
                    torch_dtype=get_torch_dtype(self.torch_dtype),
                    use_safetensors=self.use_safetensors
                )
            except Exception as local_error:
                self.pipeline = AutoPipelineForImage2Image.from_pretrained(
                    pretrained_model_or_path=self.pretrained_model_name_or_path,
                    torch_dtype=get_torch_dtype(self.torch_dtype),
                    use_safetensors=self.use_safetensors
                )
                
            print("Image2Image pipeline:", self.pipeline)

            # Memory optimization
            if self.enable_sequential_cpu_offload:
                self.pipeline.reset_device_map()
                try:
                    self.pipeline.enable_sequential_cpu_offload()
                    self.enable_model_cpu_offload = False
                except ImportError:
                    print("Accelerate library needs to be upgraded: pip install accelerate>=0.14.0")
                    self.enable_model_cpu_offload = True
                except RuntimeError:
                    print("No available accelerator device found")
                    self.enable_model_cpu_offload = True
                except ValueError as e:
                    print(f"Configuration conflict: {e}")
                    self.enable_model_cpu_offload = True

            # Move pipeline to the appropriate device
            if not self.enable_sequential_cpu_offload:
                device_type = self.get_device_type()
                if device_type.code_ == nndeploy.base.DeviceTypeCode.cuda:
                    device_id = device_type.device_id_
                    target_device = f"cuda:{device_id}"
                    self.pipeline.to(target_device)
                elif device_type.code_ == nndeploy.base.DeviceTypeCode.cpu:
                    self.pipeline.to("cpu")

            # More memory optimization
            if self.enable_model_cpu_offload:
                self.pipeline.enable_model_cpu_offload()
            if self.enable_xformers_memory_efficient_attention:
                try:
                    import xformers
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    print("XFormers memory efficient attention enabled.")
                except ImportError:
                    print("Warning: xformers is not installed. Cannot enable memory efficient attention. Please refer to https://github.com/facebookresearch/xformers for installation.")
                except Exception as e:
                    print(f"Failed to enable XFormers memory efficient attention: {e}")

            return nndeploy.base.Status.ok()
        except Exception as e:
            print(f"Failed to initialize Diffusers image-to-image pipeline: {e}")
            return nndeploy.base.Status.error()

    def run(self) -> nndeploy.base.Status:
        """Run image-to-image generation"""
        try:
            if self.scheduler != "default":
                self.pipeline.scheduler = get_scheduler(self.pipeline, self.scheduler, self.scheduler_kwargs)
            # Get input text prompt
            input_edge = self.get_input(0)
            prompt = input_edge.get(self)

            input_edge = self.get_input(1)
            negative_prompt = input_edge.get(self)

            input_edge = self.get_input(2)
            source_image = input_edge.get(self)

            # Determine num_images_per_prompt based on source_image type
            num_images_per_prompt = 1
            if source_image is not None:
                if isinstance(source_image, (list, tuple)):
                    num_images_per_prompt = len(source_image)
                elif isinstance(source_image, torch.Tensor) and len(source_image.shape) >= 4:
                    # For torch.Tensor, assume the first dimension is batch
                    num_images_per_prompt = source_image.shape[0]

            generator = None
            if not self.is_random:
                device_type = self.get_device_type()
                device = "cpu"
                if device_type.code_ == nndeploy.base.DeviceTypeCode.cuda:
                    device_id = device_type.device_id_
                    device = f"cuda:{device_id}"
                generator = torch.Generator(device=device)
                generator.manual_seed(self.generator_seed)

            import inspect
            pipeline_call_signature = inspect.signature(self.pipeline.__call__)
            supports_guidance_rescale = 'guidance_rescale' in pipeline_call_signature.parameters
            if supports_guidance_rescale:
                result = self.pipeline(
                    prompt=prompt,
                    image=source_image,
                    strength=self.strength,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator,
                    guidance_rescale=self.guidance_rescale
                )
            else:
                result = self.pipeline(
                    prompt=prompt,
                    image=source_image,
                    strength=self.strength,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator,
                )

            # Set output to output edges
            min_len = min(len(result.images), len(self.get_all_output()))
            for i in range(min_len):
                output_edge = self.get_output(i)
                generated_image = result.images[i]
                output_edge.set(generated_image)

            return nndeploy.base.Status.ok()
        except Exception as e:
            print(f"Image-to-image inference failed: {e}")
            return nndeploy.base.Status.error()

    def serialize(self) -> str:
        """Serialize node parameters"""
        try:
            # Add required parameters
            self.add_required_param("pretrained_model_name_or_path")
            self.add_dropdown_param("pretrained_model_name_or_path", get_image2image_pipelines_pretrained_model_paths())
            # Get base class serialization
            base_json = super().serialize()
            json_obj = json.loads(base_json)

            # Serialize inference parameters
            json_obj["pretrained_model_name_or_path"] = self.pretrained_model_name_or_path
            json_obj["torch_dtype"] = self.torch_dtype
            json_obj["use_safetensors"] = self.use_safetensors
            json_obj["num_inference_steps"] = self.num_inference_steps
            json_obj["guidance_scale"] = self.guidance_scale
            json_obj["guidance_rescale"] = self.guidance_rescale
            json_obj["scheduler"] = self.scheduler
            json_obj["scheduler_kwargs"] = self.scheduler_kwargs
            json_obj["strength"] = self.strength
            json_obj["is_random"] = self.is_random
            json_obj["generator_seed"] = self.generator_seed
            json_obj["enable_model_cpu_offload"] = self.enable_model_cpu_offload
            json_obj["enable_sequential_cpu_offload"] = self.enable_sequential_cpu_offload
            json_obj["enable_xformers_memory_efficient_attention"] = self.enable_xformers_memory_efficient_attention

            return json.dumps(json_obj, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Serialization failed: {e}")
            return "{}"

    def deserialize(self, json_str: str) -> nndeploy.base.Status:
        """Deserialize node parameters"""
        try:
            json_obj = json.loads(json_str)

            self.pretrained_model_name_or_path = json_obj.get("pretrained_model_name_or_path", "")
            self.torch_dtype = json_obj.get("torch_dtype", "float16")
            self.use_safetensors = json_obj.get("use_safetensors", True)
            self.num_inference_steps = json_obj.get("num_inference_steps", 50)
            self.guidance_scale = json_obj.get("guidance_scale", 8.0)
            self.guidance_rescale = json_obj.get("guidance_rescale", 0.0)
            self.scheduler = json_obj.get("scheduler", "default")
            self.scheduler_kwargs = json_obj.get("scheduler_kwargs", {"key": "value"})
            self.strength = json_obj.get("strength", 0.8)
            self.is_random = json_obj.get("is_random", True)
            self.generator_seed = json_obj.get("generator_seed", 42)
            self.enable_model_cpu_offload = json_obj.get("enable_model_cpu_offload", False)
            self.enable_sequential_cpu_offload = json_obj.get("enable_sequential_cpu_offload", False)
            self.enable_xformers_memory_efficient_attention = json_obj.get("enable_xformers_memory_efficient_attention", False)

            return super().deserialize(json_str)

        except Exception as e:
            print(f"Deserialization failed: {e}")
            return nndeploy.base.Status.ok()


class Image2ImageCreator(nndeploy.dag.NodeCreator):
    """Diffusers Pipeline image-to-image node creator"""
    def __init__(self):
        super().__init__()

    def create_node(self, name: str, inputs: list[nndeploy.dag.Edge], outputs: list[nndeploy.dag.Edge]):
        """Create Diffusers Pipeline image-to-image node instance"""
        self.node = Image2Image(name, inputs, outputs)
        return self.node

# Register node creator
image2image_creator = Image2ImageCreator()
nndeploy.dag.register_node("nndeploy.diffusion.Image2Image", image2image_creator)


class Inpainting(nndeploy.dag.Node):
    """
    Diffusers Pipeline node for image inpainting based on the nndeploy framework.

    This node wraps the image inpainting functionality of the Hugging Face Diffusers library,
    supporting Stable Diffusion and other diffusion models for image inpainting inference.
    """

    def __init__(self, name: str, inputs: Optional[List[nndeploy.dag.Edge]] = None, outputs: Optional[List[nndeploy.dag.Edge]] = None):
        super().__init__(name, inputs, outputs)
        self.set_key("nndeploy.diffusion.Inpainting")
        self.set_desc("Diffusers Pipeline for image inpainting")
        self.set_input_type(str, "Input text prompt")
        self.set_input_type(str, "Input negative text prompt")
        self.set_input_type(Image, "Input source image")
        self.set_input_type(Image, "Input mask image")
        self.set_output_type(Image, "Output PIL image")
        self.set_dynamic_output(True)

        # Set default device to CUDA
        self.set_device_type(nndeploy.base.DeviceType(nndeploy.base.DeviceTypeCode.cuda, 0))

        # Parameters
        self.pretrained_model_name_or_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        self.torch_dtype = "float16"
        self.use_safetensors = True
        self.num_inference_steps = 50
        self.guidance_scale = 8.0
        self.guidance_rescale = 0.0
        self.scheduler = "default"
        self.scheduler_kwargs = {"key": "value"}
        self.strength = 0.8  # Inpainting strength, controls the degree of modification to the original image
        self.is_random = True  # Whether to generate noise randomly
        self.generator_seed = 42  # Random seed, None means random
        self.enable_model_cpu_offload = False  # Whether to enable model CPU offload
        self.enable_sequential_cpu_offload = False
        self.enable_xformers_memory_efficient_attention = False  # Whether to enable XFormers memory efficient attention

        # Pipeline instance
        self.pipeline = None

    def init(self):
        try:
            # Load model
            try:
                self.pipeline = AutoPipelineForInpainting.from_pretrained(
                    pretrained_model_or_path=self.pretrained_model_name_or_path,
                    local_files_only=True,
                    torch_dtype=get_torch_dtype(self.torch_dtype),
                    use_safetensors=self.use_safetensors,
                )
            except Exception as local_error:
                self.pipeline = AutoPipelineForInpainting.from_pretrained(
                    pretrained_model_or_path=self.pretrained_model_name_or_path,
                    torch_dtype=get_torch_dtype(self.torch_dtype),
                    use_safetensors=self.use_safetensors,
                )
            
            print("Inpainting pipeline:", self.pipeline)

            # Memory optimization
            if self.enable_sequential_cpu_offload and hasattr(self.pipeline, "enable_sequential_cpu_offload"):
                self.pipeline.reset_device_map()
                try:
                    self.pipeline.enable_sequential_cpu_offload()
                    self.enable_model_cpu_offload = False 
                except ImportError as e:
                    print("Accelerate library needs to be upgraded: pip install accelerate>=0.14.0")
                    self.enable_model_cpu_offload = True  
                except RuntimeError as e:
                    print("No available accelerator device found")
                    self.enable_model_cpu_offload = True  
                except ValueError as e:
                    print(f"Configuration conflict: {e}")
                    self.enable_model_cpu_offload = True         
            
            # Move pipeline to the appropriate device
            if not self.enable_sequential_cpu_offload:
                device_type = self.get_device_type()
                if device_type.code_ == nndeploy.base.DeviceTypeCode.cuda:
                    device_id = device_type.device_id_
                    target_device = f"cuda:{device_id}"
                    self.pipeline.to(target_device)
                elif device_type.code_ == nndeploy.base.DeviceTypeCode.cpu:
                    self.pipeline.to("cpu")
                
            # More memory optimization    
            if self.enable_model_cpu_offload and hasattr(self.pipeline, "enable_model_cpu_offload"):
                self.pipeline.enable_model_cpu_offload()
            if self.enable_xformers_memory_efficient_attention and hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
                try:
                    import xformers
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    print("XFormers memory efficient attention enabled.")
                except ImportError:
                    print("Warning: xformers is not installed. Cannot enable memory efficient attention. Please refer to https://github.com/facebookresearch/xformers for installation.")
                except Exception as e:
                    print(f"Failed to enable XFormers memory efficient attention: {e}")
                
            return nndeploy.base.Status.ok()  
        except Exception as e:
            print(f"Failed to initialize Diffusers inpainting pipeline: {e}")
            return nndeploy.base.Status.error()

    def run(self) -> nndeploy.base.Status:
        """Run image inpainting"""
        try:
            # Get inputs
            prompt = self.get_input(0).get(self) if self.get_input(0) else ""
            negative_prompt = self.get_input(1).get(self) if self.get_input(1) else None
            source_image = self.get_input(2).get(self) if self.get_input(2) else None
            mask_image = self.get_input(3).get(self) if self.get_input(3) else None

            # Check input validity
            if not prompt or source_image is None or mask_image is None:
                print("Missing input: prompt/source_image/mask_image cannot be None or empty")
                return nndeploy.base.Status.error()

            # Handle num_images_per_prompt
            num_images_per_prompt = 1
            if isinstance(source_image, (list, tuple)):
                num_images_per_prompt = len(source_image)
            elif isinstance(source_image, torch.Tensor) and len(source_image.shape) >= 4:
                num_images_per_prompt = source_image.shape[0]

            # Generator
            generator = None
            if not self.is_random:
                device_type = self.get_device_type()
                device = "cpu"
                if device_type.code_ == nndeploy.base.DeviceTypeCode.cuda:
                    device_id = device_type.device_id_
                    device = f"cuda:{device_id}"
                generator = torch.Generator(device=device)
                generator.manual_seed(self.generator_seed)

            # Inference
            import inspect
            pipeline_call_signature = inspect.signature(self.pipeline.__call__)
            supports_guidance_rescale = 'guidance_rescale' in pipeline_call_signature.parameters
            if supports_guidance_rescale:
                result = self.pipeline(
                    prompt=prompt,
                    image=source_image,
                    mask_image=mask_image,
                    strength=self.strength,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator,
                    guidance_rescale=self.guidance_rescale
                )
            else:
                result = self.pipeline(
                    prompt=prompt,
                    image=source_image,
                    mask_image=mask_image,
                    strength=self.strength,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator,
                )

            # Set output to output edges
            min_len = min(len(result.images), len(self.get_all_output()))
            for i in range(min_len):
                output_edge = self.get_output(i)
                generated_image = result.images[i]
                output_edge.set(generated_image)

            return nndeploy.base.Status.ok()
        except Exception as e:
            print(f"Inpainting inference failed: {e}")
            return nndeploy.base.Status.error()

    def serialize(self) -> str:
        """Serialize node parameters"""
        try:
            # Add required parameters
            self.add_required_param("pretrained_model_name_or_path")
            self.add_dropdown_param("pretrained_model_name_or_path", get_inpainting_pipelines_pretrained_model_paths())
            # Get base class serialization
            base_json = super().serialize()
            json_obj = json.loads(base_json)

            # Serialize inference parameters
            json_obj["pretrained_model_name_or_path"] = self.pretrained_model_name_or_path
            json_obj["torch_dtype"] = self.torch_dtype
            json_obj["use_safetensors"] = self.use_safetensors
            json_obj["num_inference_steps"] = self.num_inference_steps
            json_obj["guidance_scale"] = self.guidance_scale
            json_obj["guidance_rescale"] = self.guidance_rescale
            json_obj["scheduler"] = self.scheduler
            json_obj["scheduler_kwargs"] = self.scheduler_kwargs
            json_obj["strength"] = self.strength
            json_obj["is_random"] = self.is_random
            json_obj["generator_seed"] = self.generator_seed
            json_obj["enable_model_cpu_offload"] = self.enable_model_cpu_offload
            json_obj["enable_sequential_cpu_offload"] = self.enable_sequential_cpu_offload
            json_obj["enable_xformers_memory_efficient_attention"] = self.enable_xformers_memory_efficient_attention

            return json.dumps(json_obj, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Serialization failed: {e}")
            return super().serialize()

    def deserialize(self, json_str: str) -> nndeploy.base.Status:
        """Deserialize node parameters"""
        try:
            json_obj = json.loads(json_str)

            self.pretrained_model_name_or_path = json_obj.get("pretrained_model_name_or_path", "")
            self.torch_dtype = json_obj.get("torch_dtype", "float16")
            self.use_safetensors = json_obj.get("use_safetensors", True)
            self.num_inference_steps = json_obj.get("num_inference_steps", 50)
            self.guidance_scale = json_obj.get("guidance_scale", 8.0)
            self.guidance_rescale = json_obj.get("guidance_rescale", 0.0)
            self.scheduler = json_obj.get("scheduler", "default")
            self.scheduler_kwargs = json_obj.get("scheduler_kwargs", {"key": "value"})
            self.strength = json_obj.get("strength", 0.8)
            self.is_random = json_obj.get("is_random", True)
            self.generator_seed = json_obj.get("generator_seed", 42)
            self.enable_model_cpu_offload = json_obj.get("enable_model_cpu_offload", False)
            self.enable_sequential_cpu_offload = json_obj.get("enable_sequential_cpu_offload", False)
            self.enable_xformers_memory_efficient_attention = json_obj.get("enable_xformers_memory_efficient_attention", False)

            return super().deserialize(json_str)

        except Exception as e:
            print(f"Deserialization failed: {e}")
            return nndeploy.base.Status.ok()

class InpaintingCreator(nndeploy.dag.NodeCreator):
    """Diffusers Pipeline image inpainting node creator"""
    def __init__(self):
        super().__init__()

    def create_node(self, name: str, inputs: List[nndeploy.dag.Edge], outputs: List[nndeploy.dag.Edge]):
        """Create Diffusers Pipeline image inpainting node instance"""
        self.node = Inpainting(name, inputs, outputs)
        return self.node

# Register node creator
inpainting_creator = InpaintingCreator()
nndeploy.dag.register_node("nndeploy.diffusion.Inpainting", inpainting_creator)
