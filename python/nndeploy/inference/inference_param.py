
import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.device
import nndeploy.ir
import nndeploy.op


class InferenceParam(_C.inference.InferenceParam):
    def __init__(self, inference_type):
        super().__init__(inference_type)
        self._default_dic = {}

    @property
    def inference_type(self):
        return self.inference_type_

    @inference_type.setter
    def inference_type(self, value):
        self.inference_type_ = value

    @property
    def model_type(self):
        return self.model_type_

    @model_type.setter
    def model_type(self, value):
        self.model_type_ = value

    @property
    def is_path(self):
        return self.is_path_

    @is_path.setter
    def is_path(self, value):
        self.is_path_ = value

    @property
    def model_value(self):
        return self.model_value_

    @model_value.setter
    def model_value(self, value):
        self.model_value_ = value

    @property
    def encrypt_type(self):
        return self.encrypt_type_

    @encrypt_type.setter
    def encrypt_type(self, value):
        self.encrypt_type_ = value

    @property
    def license(self):
        return self.license_

    @license.setter
    def license(self, value):
        self.license_ = value

    @property
    def device_type(self):
        return self.device_type_

    @device_type.setter
    def device_type(self, value):
        self.device_type_ = value

    @property
    def num_thread(self):
        return self.num_thread_

    @num_thread.setter
    def num_thread(self, value):
        self.num_thread_ = value

    @property
    def gpu_tune_kernel(self):
        return self.gpu_tune_kernel_

    @gpu_tune_kernel.setter
    def gpu_tune_kernel(self, value):
        self.gpu_tune_kernel_ = value

    @property
    def share_memory_mode(self):
        return self.share_memory_mode_

    @share_memory_mode.setter
    def share_memory_mode(self, value):
        self.share_memory_mode_ = value

    @property
    def precision_type(self):
        return self.precision_type_

    @precision_type.setter
    def precision_type(self, value):
        self.precision_type_ = value

    @property
    def power_type(self):
        return self.power_type_

    @power_type.setter
    def power_type(self, value):
        self.power_type_ = value

    @property
    def is_dynamic_shape(self):
        return self.is_dynamic_shape_

    @is_dynamic_shape.setter
    def is_dynamic_shape(self, value):
        self.is_dynamic_shape_ = value

    @property
    def min_shape(self):
        return self.min_shape_

    @min_shape.setter
    def min_shape(self, value):
        self.min_shape_ = value

    @property
    def opt_shape(self):
        return self.opt_shape_

    @opt_shape.setter
    def opt_shape(self, value):
        self.opt_shape_ = value

    @property
    def max_shape(self):
        return self.max_shape_

    @max_shape.setter
    def max_shape(self, value):
        self.max_shape_ = value

    @property
    def cache_path(self):
        return self.cache_path_

    @cache_path.setter
    def cache_path(self, value):
        self.cache_path_ = value

    @property
    def library_path(self):
        return self.library_path_

    @library_path.setter
    def library_path(self, value):
        self.library_path_ = value

    def __str__(self):
        return str(self._default_dic)

    def set(self, dic : dict):
        for k, v in dic.items():
            if k in self._default_dic:
                self._default_dic[k] = v   
            else:
                print(f"Unsupported key: {k}")

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

