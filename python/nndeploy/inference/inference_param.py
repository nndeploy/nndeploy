
import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.device
import nndeploy.ir
import nndeploy.op


class InferenceParam(_C.inference.InferenceParam):
    def __init__(self, inference_type):
        super().__init__(inference_type)
        self._default_dic = {}

    def set_inference_type(self, inference_type: nndeploy.base.InferenceType):
        super().set_inference_type(inference_type)

    def get_inference_type(self):
        return super().get_inference_type()
        
    def set_model_type(self, model_type: nndeploy.base.ModelType):
        super().set_model_type(model_type)
        
    def get_model_type(self):
        return super().get_model_type()
        
    def set_is_path(self, is_path: bool):
        super().set_is_path(is_path)
        
    def get_is_path(self):
        return super().get_is_path()
        
    def set_model_value(self, model_value, index=-1):
        if isinstance(model_value, list):
            super().set_model_value(model_value)
        else:
            super().set_model_value(model_value, index)
            
    def get_model_value(self):
        return super().get_model_value()
        
    def set_input_num(self, input_num: int):
        super().set_input_num(input_num)
        
    def get_input_num(self):
        return super().get_input_num()
        
    def set_input_name(self, input_name, index=-1):
        if isinstance(input_name, list):
            super().set_input_name(input_name)
        else:
            super().set_input_name(input_name, index)
            
    def get_input_name(self):
        return super().get_input_name()
        
    def set_input_shape(self, input_shape, index=-1):
        if isinstance(input_shape[0], list):
            super().set_input_shape(input_shape)
        else:
            super().set_input_shape(input_shape, index)
            
    def get_input_shape(self):
        return super().get_input_shape()
        
    def set_output_num(self, output_num: int):
        super().set_output_num(output_num)
        
    def get_output_num(self):
        return super().get_output_num()
        
    def set_output_name(self, output_name, index=-1):
        if isinstance(output_name, list):
            super().set_output_name(output_name)
        else:
            super().set_output_name(output_name, index)
            
    def get_output_name(self):
        return super().get_output_name()
        
    def set_encrypt_type(self, encrypt_type: nndeploy.base.EncryptType):
        super().set_encrypt_type(encrypt_type)
        
    def get_encrypt_type(self):
        return super().get_encrypt_type()
        
    def set_license(self, license: str):
        super().set_license(license)
        
    def get_license(self):
        return super().get_license()
        
    def set_device_type(self, device_type: nndeploy.base.DeviceType):
        super().set_device_type(device_type)
        
    def get_device_type(self):
        return super().get_device_type()
        
    def set_num_thread(self, num_thread: int):
        super().set_num_thread(num_thread)
        
    def get_num_thread(self):
        return super().get_num_thread()
        
    def set_gpu_tune_kernel(self, gpu_tune_kernel: int):
        super().set_gpu_tune_kernel(gpu_tune_kernel)
        
    def get_gpu_tune_kernel(self):
        return super().get_gpu_tune_kernel()
        
    def set_share_memory_mode(self, share_memory_mode: nndeploy.base.ShareMemoryType):
        super().set_share_memory_mode(share_memory_mode)
        
    def get_share_memory_mode(self):
        return super().get_share_memory_mode()
        
    def set_precision_type(self, precision_type: nndeploy.base.PrecisionType):
        super().set_precision_type(precision_type)
        
    def get_precision_type(self):
        return super().get_precision_type()
        
    def set_power_type(self, power_type: nndeploy.base.PowerType):
        super().set_power_type(power_type)
        
    def get_power_type(self):
        return super().get_power_type()
        
    def set_is_dynamic_shape(self, is_dynamic_shape: bool):
        super().set_is_dynamic_shape(is_dynamic_shape)
        
    def get_is_dynamic_shape(self):
        return super().get_is_dynamic_shape()
        
    def set_min_shape(self, min_shape: dict):
        super().set_min_shape(min_shape)
        
    def get_min_shape(self):
        return super().get_min_shape()
        
    def set_opt_shape(self, opt_shape: dict):
        super().set_opt_shape(opt_shape)
        
    def get_opt_shape(self):
        return super().get_opt_shape()
        
    def set_max_shape(self, max_shape: dict):
        super().set_max_shape(max_shape)
        
    def get_max_shape(self):
        return super().get_max_shape()
        
    def set_cache_path(self, cache_path: list):
        super().set_cache_path(cache_path)
        
    def get_cache_path(self):
        return super().get_cache_path()
        
    def set_library_path(self, library_path, index=-1):
        if isinstance(library_path, list):
            super().set_library_path(library_path)
        else:
            super().set_library_path(library_path, index)
            
    def get_library_path(self):
        return super().get_library_path()
    

    def __str__(self):
        return str(self._default_dic)

    def set(self, dic : dict):
        for k, v in dic.items():
            self._default_dic[k] = v   

    def get(self, key: str):
        if key in self._default_dic:
            return self._default_dic[key]
        else:
            print(f"Unsupported key: {key}")
            return None


class InferenceParamCreator(_C.inference.InferenceParamCreator):
    def __init__(self):
        super().__init__()

    def create_inference_param(self, type: nndeploy.base.InferenceType):
        # 必须实现
        raise NotImplementedError("base class InferenceParamCreator does not implement create_inference_param method")


def register_inference_param_creator(type: nndeploy.base.InferenceType, creator: InferenceParamCreator):
    _C.inference.register_inference_param_creator(type, creator)


def create_inference_param(type: nndeploy.base.InferenceType):
    return _C.inference.create_inference_param(type)

