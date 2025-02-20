
import nndeploy._nndeploy_internal as _C


import nndeploy.base
import nndeploy.device
import nndeploy.ir
import nndeploy.op
import nndeploy.inference

from nndeploy.inference import Inference, InferenceCreator, register_inference_creator, create_inference


# python3 nndeploy/test/inference/test_inference.py


class MyInference(_C.inference.Inference):
    def __init__(self, type):
        super().__init__(type)


class MyInferenceCreator(_C.inference.InferenceCreator):
    def __init__(self):
        super().__init__()

    def create_inference(self, type):
        if type == nndeploy.base.InferenceType.NotSupport:
            return MyInference(type)
        else:
            return super().create_inference(type)
        
    
if __name__ == "__main__":
    creator = MyInferenceCreator()
    register_inference_creator(nndeploy.base.InferenceType.NotSupport,  creator)
    inference = create_inference(nndeploy.base.InferenceType.NotSupport)
    print(inference)
    param = inference.get_param()
    print(param)

    inference_param = _C.inference.create_inference_param(nndeploy.base.InferenceType.AscendCL)
    inference = create_inference(nndeploy.base.InferenceType.AscendCL)
    print(inference)
    # param = inference.get_param_cpp()
    # param = _C.inference.InferenceParam.cast(param)
    # 强转为InferenceParam
    # param = _C.inference.InferenceParam(param)
    inference_param.model_type = nndeploy.base.ModelType.Onnx
    inference.set_param(inference_param)
    # inference.init()
    # inference.run()
    # output_tensor = inference.get_output_tensor_after_run("output", nndeploy.base.DeviceType.AscendCL, False)
    # print(output_tensor)
    
    print(inference_param)

