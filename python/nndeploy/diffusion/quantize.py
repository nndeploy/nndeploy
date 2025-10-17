
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

class QuantizationParam(nndeploy.base.Param):
    def __init__(self):
        super().__init__()
        self.is_quantized = False
        self.quant_backend = "bitsandbytes_4bit"
        self.quant_kwargs = {}
        self.components_to_quantize = ["transformer", "text_encoder"]
        self.quant_mapping = {}
        
    def serialize(self) -> str:
        """Serialize parameters to JSON string"""
        param_dict = {
            "is_quantized": self.is_quantized,
            "quant_backend": self.quant_backend,
            "quant_kwargs": self.quant_kwargs,
            "components_to_quantize": self.components_to_quantize,
            "quant_mapping": self.quant_mapping,
        }
        return json.dumps(param_dict, ensure_ascii=False, indent=2)
    
    def deserialize(self, json_str: str) -> nndeploy.base.Status:
        """Deserialize parameters from JSON string"""
        try:
            param_dict = json.loads(json_str)
            
            self.is_quantized = param_dict.get("is_quantized", False)
            self.quant_backend = param_dict.get("quant_backend", None)
            self.quant_kwargs = param_dict.get("quant_kwargs", {})
            self.components_to_quantize = param_dict.get("components_to_quantize", [])
            self.quant_mapping = param_dict.get("quant_mapping", {})
            
            return nndeploy.base.Status.ok()
        except Exception as e:
            print(f"Failed to deserialize quantization parameters: {e}")
            return nndeploy.base.Status.error()