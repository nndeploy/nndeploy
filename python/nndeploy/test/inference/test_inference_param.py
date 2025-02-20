
import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.device
import nndeploy.ir
import nndeploy.op


# python3 nndeploy/test/inference/test_inference_param.py


from nndeploy.inference import InferenceParam, InferenceParamCreator, register_inference_param_creator, create_inference_param


class MyInferenceParam(InferenceParam):
    def __init__(self, inference_type: nndeploy.base.InferenceType):
        super().__init__(inference_type)
        self._default_dic = {}
        self.test = "test"

    def __str__(self):
        return f"inference_type: {self.inference_type}, model_type: {self.model_type}, is_path: {self.is_path}, model_value: {self.model_value}, encrypt_type: {self.encrypt_type}, license: {self.license}, device_type: {self.device_type}, num_thread: {self.num_thread}, gpu_tune_kernel: {self.gpu_tune_kernel}, share_memory_mode: {self.share_memory_mode}, precision_type: {self.precision_type}, power_type: {self.power_type}, is_dynamic_shape: {self.is_dynamic_shape}, min_shape: {self.min_shape}, opt_shape: {self.opt_shape}, max_shape: {self.max_shape}, cache_path: {self.cache_path}, library_path: {self.library_path}, _default_dic: {self._default_dic}, test: {self.test}"
    

class MyInferenceParamCreator(_C.inference.InferenceParamCreator):
    def __init__(self):
        super().__init__()
    def create_inference_param(self, inference_type: nndeploy.base.InferenceType) -> _C.ir.Interpret:
        print(inference_type)
        if inference_type == nndeploy.base.InferenceType.NotSupport:
            param = MyInferenceParam(inference_type)
            return param
        else:
            return MyInferenceParam(inference_type)



if __name__ == "__main__":
    print("InferenceParam")
    inference_param = InferenceParam(nndeploy.base.InferenceType.NotSupport)
    print(inference_param)
    
    creator = MyInferenceParamCreator()
    _C.inference.register_inference_param_creator(nndeploy.base.InferenceType.NotSupport, creator)
    test_inference_param = InferenceParam(nndeploy.base.InferenceType.NotSupport)
    print(type(test_inference_param))

    default_inference_param = _C.inference.create_inference_param(nndeploy.base.InferenceType.NotSupport)
    print(type(default_inference_param))
    print((default_inference_param))
