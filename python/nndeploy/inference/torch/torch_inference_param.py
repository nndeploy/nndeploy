
import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.device
import nndeploy.ir
import nndeploy.op
from ..inference_param import InferenceParam, InferenceParamCreator, register_inference_param_creator

import torch

class TorchInferenceParam(nndeploy.inference.InferenceParam):
    def __init__(self, inference_type):
        super().__init__(inference_type)
        self.model_type = nndeploy.base.ModelType.TorchPth
        self.is_path = False        
        self._nn_model : torch.nn.Module = None
        self._is_compile = False
        
    def set_model_value(self, value, index=-1):
        super().set_model_value(value, index)
        if isinstance(value, str):
            if self.is_path and self.model_type == nndeploy.base.ModelType.TorchScript:
                self._nn_model = torch.jit.load(value)
            else:
                self._nn_model = torch.jit.load(io.BytesIO(value))
        else:
            raise ValueError("Invalid model value")
          
    @property
    def nn_model(self):
        return self._nn_model
      
    @nn_model.setter
    def nn_model(self, value):
        self._nn_model = value
        
    def set(self, value : Union[torch.nn.Module, bool, dict]):
        if isinstance(value, dict):
            super().set(value)
        elif isinstance(value, bool):
            self._is_compile = value
        else:
            self._nn_model = value
    
    def get(self, key : str = None):
        if key is None:
            return self._nn_model
        else:
            return super().get(key)


class TorchInferenceParamCreator(InferenceParamCreator):
    def __init__(self):
        super().__init__()

    def create_inference_param(self, type: nndeploy.base.InferenceType):
        return TorchInferenceParam(type)


register_inference_param_creator(nndeploy.base.InferenceType.Torch, TorchInferenceParamCreator())