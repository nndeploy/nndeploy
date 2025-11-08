
try:
    from .inference_param import InferenceParam, InferenceParamCreator, register_inference_param_creator, create_inference_param
    from .inference import Inference, InferenceCreator, register_inference_creator, create_inference
except:
    pass