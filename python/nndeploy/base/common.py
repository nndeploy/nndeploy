import nndeploy._nndeploy_internal as _C

device_name_to_code = {
    "cpu": _C.base.DeviceTypeCode.cpu,
    "cuda": _C.base.DeviceTypeCode.cuda,
    "arm": _C.base.DeviceTypeCode.arm,
    "x86": _C.base.DeviceTypeCode.x86,
    "ascendcl": _C.base.DeviceTypeCode.ascendcl,
    "opencl": _C.base.DeviceTypeCode.opencl,
    "opengl": _C.base.DeviceTypeCode.opengl,
    "metal": _C.base.DeviceTypeCode.metal,
    "vulkan": _C.base.DeviceTypeCode.vulkan,
    "applenpu": _C.base.DeviceTypeCode.applenpu,
}

class DeviceType(_C.base.DeviceType):
    def __init__(self, device_name = "cpu", device_id =0):
        super().__init__()
        self.code_ = device_name_to_code[device_name]
        self.device_id_ = device_id


def model_type_from_name(model_type_name: str) -> _C.base.ModelType:
    if model_type_name == "default":
        return _C.base.ModelType.kModelTypeDefault
    elif model_type_name == "openvino":
        return _C.base.ModelType.kModelTypeOpenVino
    elif model_type_name == "tensorrt":
        return _C.base.ModelType.kModelTypeTensorRt
    elif model_type_name == "coreml":
        return _C.base.ModelType.kModelTypeCoreML
    elif model_type_name == "tflite":
        return _C.base.ModelType.kModelTypeTfLite
    elif model_type_name == "onnx":
        return _C.base.ModelType.kModelTypeOnnx
    elif model_type_name == "ascendcl":
        return _C.base.ModelType.kModelTypeAscendCL
    elif model_type_name == "ncnn":
        return _C.base.ModelType.kModelTypeNcnn
    elif model_type_name == "tnn":
        return _C.base.ModelType.kModelTypeTnn
    elif model_type_name == "mnn":
        return _C.base.ModelType.kModelTypeMnn
    elif model_type_name == "paddlelite":
        return _C.base.ModelType.kModelTypePaddleLite
    elif model_type_name == "rknn":
        return _C.base.ModelType.kModelTypeRknn
    elif model_type_name == "tvm":
        return _C.base.ModelType.kModelTypeTvm
    elif model_type_name == "aitemplate":
        return _C.base.ModelType.kModelTypeAITemplate
    elif model_type_name == "snpe":
        return _C.base.ModelType.kModelTypeSnpe
    elif model_type_name == "qnn":
        return _C.base.ModelType.kModelTypeQnn
    elif model_type_name == "sophon":
        return _C.base.ModelType.kModelTypeSophon
    elif model_type_name == "torchscript":
        return _C.base.ModelType.kModelTypeTorchScript
    elif model_type_name == "torchpth":
        return _C.base.ModelType.kModelTypeTorchPth
    elif model_type_name == "hdf5":
        return _C.base.ModelType.kModelTypeHdf5
    elif model_type_name == "safetensors":
        return _C.base.ModelType.kModelTypeSafetensors
    elif model_type_name == "neuropilot":
        return _C.base.ModelType.kModelTypeNeuroPilot
    else:
        raise ValueError(f"not support model type: {model_type_name}")