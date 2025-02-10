
import json
import nndeploy._nndeploy_internal as _C

from enum import Enum
import numpy as np
import json


name_to_data_type_code = {
    "Uint": _C.base.DataTypeCode.Uint,
    "Int": _C.base.DataTypeCode.Int,
    "Fp": _C.base.DataTypeCode.Fp,
    "BFp": _C.base.DataTypeCode.BFp,
    "OpaqueHandle": _C.base.DataTypeCode.OpaqueHandle,
    "NotSupport": _C.base.DataTypeCode.NotSupport,
}


data_type_code_to_name = {v: k for k, v in name_to_data_type_code.items()}


class DataTypeCode(Enum):
    Uint = _C.base.DataTypeCode.Uint
    Int = _C.base.DataTypeCode.Int
    Fp = _C.base.DataTypeCode.Fp
    BFp = _C.base.DataTypeCode.BFp
    OpaqueHandle = _C.base.DataTypeCode.OpaqueHandle
    NotSupport = _C.base.DataTypeCode.NotSupport

    @classmethod
    def from_data_type_code(cls, data_type_code: _C.base.DataTypeCode) -> _C.base.DataTypeCode:
        return cls(data_type_code)
    
    @classmethod
    def from_name(cls, name: str) -> _C.base.DataTypeCode:
        if name not in name_to_data_type_code:
            raise ValueError(f"Unsupported data type code: {name}")
        else:
            return cls(name_to_data_type_code[name])   


class DataType(_C.base.DataType):
    def __init__(self, data_type_code: DataTypeCode, bits: int, lanes: int = 1):
        super().__init__(data_type_code.value, bits, lanes)

    @classmethod
    def from_data_type(cls, data_type: _C.base.DataType):
        return cls(DataTypeCode.from_data_type_code(data_type.code_), data_type.bits_, data_type.lanes_)
    
    @classmethod
    def from_data_type_code(cls, data_type_code: _C.base.DataTypeCode, bits: int, lanes: int = 1):
        return cls(DataTypeCode.from_data_type_code(data_type_code), bits, lanes)
    
    @classmethod
    def from_numpy_dtype(cls, numpy_dtype: np.dtype):
        if numpy_dtype == np.uint8:
            return cls(DataTypeCode.Uint, 8, 1)
        elif numpy_dtype == np.uint16:
            return cls(DataTypeCode.Uint, 16, 1)
        elif numpy_dtype == np.uint32:
            return cls(DataTypeCode.Uint, 32, 1)
        elif numpy_dtype == np.uint64:
            return cls(DataTypeCode.Uint, 64, 1)
        elif numpy_dtype == np.int8:
            return cls(DataTypeCode.Int, 8, 1)
        elif numpy_dtype == np.int16:
            return cls(DataTypeCode.Int, 16, 1)
        elif     numpy_dtype == np.int32:
            return cls(DataTypeCode.Int, 32, 1)
        elif numpy_dtype == np.int64:
            return cls(DataTypeCode.Int, 64, 1)
        elif numpy_dtype == np.float16:
            return cls(DataTypeCode.Fp, 16, 1)
        elif numpy_dtype == np.float32:
            return cls(DataTypeCode.Fp, 32, 1)
        elif numpy_dtype == np.float64:
            return cls(DataTypeCode.Fp, 64, 1)
        # elif numpy_dtype == bfloat16:
        #     return cls(DataTypeCode.BFp, 16, 1)
        else:
            raise ValueError(f"Unsupported numpy dtype: {numpy_dtype}")
    
    @classmethod
    def from_name(cls, name: str):
        if name == "uint8":
            return cls(DataTypeCode.Uint, 8, 1)
        elif name == "uint16":
            return cls(DataTypeCode.Uint, 16, 1)
        elif name == "uint32":
            return cls(DataTypeCode.Uint, 32, 1)
        elif name == "uint64":
            return _C.base.DataType(_C.base.DataTypeCode.Uint, 64)
        elif name == "int8":
            return cls(DataTypeCode.Int, 8, 1)
        elif name == "int16":
            return cls(DataTypeCode.Int, 16, 1)
        elif name == "int32":
            return cls(DataTypeCode.Int, 32, 1)
        elif name == "int64":
            return cls(DataTypeCode.Int, 64, 1)
        elif name == "float16":
            return _C.base.DataType(_C.base.DataTypeCode.Fp, 16)
        elif name == "float32":
            return cls(DataTypeCode.Fp, 32, 1)
        elif name == "float64":
            return cls(DataTypeCode.Fp, 64, 1)
        elif name == "bfloat16":
            return cls(DataTypeCode.BFp, 16, 1)
        else:
            raise ValueError(f"Unsupported data type: {name}")

    def get_numpy_dtype(self):
        if self.code_ == _C.base.DataTypeCode.Uint:
            if self.bits_ == 8:
                return np.uint8
            elif self.bits_ == 16:
                return np.uint16
            elif self.bits_ == 32:
                return np.uint32
            elif self.bits_ == 64:
                return np.uint64
        elif self.code_ == _C.base.DataTypeCode.Int:
            if self.bits_ == 8:
                return np.int8
            elif self.bits_ == 16:
                return np.int16
            elif self.bits_ == 32:
                return np.int32
            elif self.bits_ == 64:
                return np.int64
        elif self.code_ == _C.base.DataTypeCode.Fp:
            if self.bits_ == 16:
                return np.float16
            elif self.bits_ == 32:
                return np.float32
            elif self.bits_ == 64:
                return np.float64
        # elif self.code_ == _C.base.DataTypeCode.BFp:
        #     if self.bits_ == 16:
        #         return bfloat16
        else:
            raise ValueError(f"Unsupported DataType: {self}")
    
    def get_data_type_code(self):
        return self.code_
    
    def get_bits(self):
        return self.bits_
    
    def get_lanes(self):
        return self.lanes_

    def get_bytes(self):
        return self.get_size()
    
    def get_name(self):
        if self.code_ == _C.base.DataTypeCode.Uint:
            if self.bits_ == 8:
                return "uint8"
            elif self.bits_ == 16:
                return "uint16"
            elif self.bits_ == 32:
                return "uint32"
            elif self.bits_ == 64:
                return "uint64"
        elif self.code_ == _C.base.DataTypeCode.Int:
            if self.bits_ == 8:
                return "int8"
            elif self.bits_ == 16:
                return "int16"
            elif self.bits_ == 32:
                return "int32"
            elif self.bits_ == 64:
                return "int64"  
        elif self.code_ == _C.base.DataTypeCode.Fp:
            if self.bits_ == 16:
                return "float16"
            elif self.bits_ == 32:
                return "float32"
            elif self.bits_ == 64:
                return "float64"
        elif self.code_ == _C.base.DataTypeCode.BFp:
            if self.bits_ == 16:
                return "bfloat16"
        else:
            raise ValueError(f"Unsupported DataType: {self}")


name_to_device_type_code = {
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
    "rocm": _C.base.DeviceTypeCode.rocm,
    "sycl": _C.base.DeviceTypeCode.sycl,
    "hexagon": _C.base.DeviceTypeCode.hexagon,
    "mtkvpu": _C.base.DeviceTypeCode.mtkvpu,
    "rknpu": _C.base.DeviceTypeCode.rknpu,
    "qualcomnpu": _C.base.DeviceTypeCode.qualcomnpu,
    "mtknpu": _C.base.DeviceTypeCode.mtknpu,
    "sophonnpu": _C.base.DeviceTypeCode.sophonnpu,
    "riscv": _C.base.DeviceTypeCode.riscv,
    "notsupport": _C.base.DeviceTypeCode.notsupport,
}


device_type_code_to_name = {v: k for k, v in name_to_device_type_code.items()}


class DeviceTypeCode(Enum):
    cpu = _C.base.DeviceTypeCode.cpu
    cuda = _C.base.DeviceTypeCode.cuda
    arm = _C.base.DeviceTypeCode.arm
    x86 = _C.base.DeviceTypeCode.x86
    ascendcl = _C.base.DeviceTypeCode.ascendcl
    opencl = _C.base.DeviceTypeCode.opencl
    opengl = _C.base.DeviceTypeCode.opengl
    metal = _C.base.DeviceTypeCode.metal
    vulkan = _C.base.DeviceTypeCode.vulkan
    applenpu = _C.base.DeviceTypeCode.applenpu
    rocm = _C.base.DeviceTypeCode.rocm
    sycl = _C.base.DeviceTypeCode.sycl
    hexagon = _C.base.DeviceTypeCode.hexagon
    mtkvpu = _C.base.DeviceTypeCode.mtkvpu
    rknpu = _C.base.DeviceTypeCode.rknpu
    qualcomnpu = _C.base.DeviceTypeCode.qualcomnpu
    mtknpu = _C.base.DeviceTypeCode.mtknpu
    sophonnpu = _C.base.DeviceTypeCode.sophonnpu
    riscv = _C.base.DeviceTypeCode.riscv
    notsupport = _C.base.DeviceTypeCode.notsupport

    @classmethod
    def from_device_type_code(cls, device_type_code: _C.base.DeviceTypeCode):
        return cls(device_type_code)
    
    @classmethod
    def from_name(cls, device_name: str):
        return cls(name_to_device_type_code[device_name])


class DeviceType(_C.base.DeviceType):
    def __init__(self, device_name = "cpu", device_id =0):
        code = name_to_device_type_code[device_name]
        super().__init__(code, device_id)

    @classmethod
    def from_device_type(cls, device_type: _C.base.DeviceType):
        if device_type.code_ not in device_type_code_to_name:
            raise ValueError(f"Unsupported device type code: {device_type.code_}")
        device_name = device_type_code_to_name[device_type.code_]
        return cls(device_name, device_type.device_id_)
    
    @classmethod  
    def from_device_type_code(cls, device_type_code: DeviceTypeCode, device_id: int = 0):
        if device_type_code.value not in device_type_code_to_name:
            raise ValueError(f"Unsupported device type code: {device_type_code}")
        device_name = device_type_code_to_name[device_type_code.value]
        return cls(device_name, device_id)
    
    @classmethod  
    def from_device_type_code_v0(cls, device_type_code: _C.base.DeviceTypeCode, device_id: int = 0):
        if device_type_code not in device_type_code_to_name:
            raise ValueError(f"Unsupported device type code: {device_type_code}")
        device_name = device_type_code_to_name[device_type_code]
        return cls(device_name, device_id)
    
    def get_device_type_code(self):
        return DeviceTypeCode.from_device_type_code(self.code_)
    
    def get_device_type_code_v0(self):
        return self.code_
    
    def get_device_id(self):
        return self.device_id_
    
    def get_device_name(self):
        return device_type_code_to_name[self.code_]


name_to_data_format = {
    "N": _C.base.DataFormat.N,
    "NC": _C.base.DataFormat.NC,
    "NCL": _C.base.DataFormat.NCL,
    "S1D": _C.base.DataFormat.S1D,
    "NCHW": _C.base.DataFormat.NCHW,
    "NHWC": _C.base.DataFormat.NHWC,
    "OIHW": _C.base.DataFormat.OIHW,
    "NC4HW": _C.base.DataFormat.NC4HW,
    "NC8HW": _C.base.DataFormat.NC8HW,
    "NCDHW": _C.base.DataFormat.NCDHW,
    "NDHWC": _C.base.DataFormat.NDHWC,
    "Auto": _C.base.DataFormat.Auto,
    "NotSupport": _C.base.DataFormat.NotSupport,
}


data_format_to_name = {v: k for k, v in name_to_data_format.items()}


class DataFormat(Enum):
    N = _C.base.DataFormat.N
    NC = _C.base.DataFormat.NC
    NCL = _C.base.DataFormat.NCL
    S1D = _C.base.DataFormat.S1D
    NCHW = _C.base.DataFormat.NCHW
    NHWC = _C.base.DataFormat.NHWC
    OIHW = _C.base.DataFormat.OIHW
    NC4HW = _C.base.DataFormat.NC4HW
    NC8HW = _C.base.DataFormat.NC8HW
    NCDHW = _C.base.DataFormat.NCDHW
    NDHWC = _C.base.DataFormat.NDHWC
    Auto = _C.base.DataFormat.Auto
    NotSupport = _C.base.DataFormat.NotSupport

    @classmethod
    def from_data_format(cls, data_format: _C.base.DataFormat):
        return cls(data_format)
    
    @classmethod
    def from_name(cls, data_format_name: str):
        if data_format_name not in name_to_data_format:
            raise ValueError(f"Unsupported data format name: {data_format_name}")
        return cls(name_to_data_format[data_format_name])


name_to_precision_type = {
    "BFp16": _C.base.PrecisionType.BFp16,
    "Fp16": _C.base.PrecisionType.Fp16,
    "Fp32": _C.base.PrecisionType.Fp32,
    "Fp64": _C.base.PrecisionType.Fp64,
    "NotSupport": _C.base.PrecisionType.NotSupport,
}


precision_type_to_name = {v: k for k, v in name_to_precision_type.items()}


class PrecisionType(Enum):
    BFp16 = _C.base.PrecisionType.BFp16
    Fp16 = _C.base.PrecisionType.Fp16
    Fp32 = _C.base.PrecisionType.Fp32
    Fp64 = _C.base.PrecisionType.Fp64
    NotSupport = _C.base.PrecisionType.NotSupport

    @classmethod
    def from_precision_type(cls, precision_type: _C.base.PrecisionType):
        return cls(precision_type)
    
    @classmethod
    def from_name(cls, precision_type_name: str):
        if precision_type_name not in name_to_precision_type:
            raise ValueError(f"Unsupported precision type name: {precision_type_name}")
        return cls(name_to_precision_type[precision_type_name])
    

name_to_power_type = {
    "High": _C.base.PowerType.High,
    "Normal": _C.base.PowerType.Normal,
    "Low": _C.base.PowerType.Low,
    "NotSupport": _C.base.PowerType.NotSupport,
}


power_type_to_name = {v: k for k, v in name_to_power_type.items()}


class PowerType(Enum):
    High = _C.base.PowerType.High
    Normal = _C.base.PowerType.Normal
    Low = _C.base.PowerType.Low
    NotSupport = _C.base.PowerType.NotSupport

    @classmethod
    def from_power_type(cls, power_type: _C.base.PowerType):
        return cls(power_type)
    
    @classmethod
    def from_name(cls, power_type_name: str):
        if power_type_name not in name_to_power_type:
            raise ValueError(f"Unsupported power type name: {power_type_name}")
        return cls(name_to_power_type[power_type_name])
    

name_to_share_memory_type = {
    "NoShare": _C.base.ShareMemoryType.NoShare,
    "ShareFromExternal": _C.base.ShareMemoryType.ShareFromExternal,
    "NotSupport": _C.base.ShareMemoryType.NotSupport,
}


share_memory_type_to_name = {v: k for k, v in name_to_share_memory_type.items()}


class ShareMemoryType(Enum):
    NoShare = _C.base.ShareMemoryType.NoShare
    ShareFromExternal = _C.base.ShareMemoryType.ShareFromExternal
    NotSupport = _C.base.ShareMemoryType.NotSupport

    @classmethod
    def from_share_memory_type(cls, share_memory_type: _C.base.ShareMemoryType):
        return cls(share_memory_type)
    
    @classmethod
    def from_name(cls, share_memory_type_name: str):
        if share_memory_type_name not in name_to_share_memory_type:
            raise ValueError(f"Unsupported share memory type name: {share_memory_type_name}")
        return cls(name_to_share_memory_type[share_memory_type_name])


name_to_memory_type = {
    "None": _C.base.MemoryType.kMemoryTypeNone,
    "Allocate": _C.base.MemoryType.Allocate,
    "External": _C.base.MemoryType.External,
    "Mapped": _C.base.MemoryType.Mapped,
}


memory_type_to_name = {v: k for k, v in name_to_memory_type.items()}


class MemoryType(Enum):
    kMemoryTypeNone = _C.base.MemoryType.kMemoryTypeNone
    Allocate = _C.base.MemoryType.Allocate
    External = _C.base.MemoryType.External
    Mapped = _C.base.MemoryType.Mapped

    @classmethod
    def from_memory_type(cls, memory_type: _C.base.MemoryType):
        return cls(memory_type)
    
    @classmethod
    def from_name(cls, memory_type_name: str):
        if memory_type_name not in name_to_memory_type:
            raise ValueError(f"Unsupported memory type name: {memory_type_name}")
        return cls(name_to_memory_type[memory_type_name])
    
    
name_to_memory_pool_type = {
    "Embed": _C.base.MemoryPoolType.Embed,
    "Unity": _C.base.MemoryPoolType.Unity,
    "ChunkIndepend": _C.base.MemoryPoolType.ChunkIndepend,
}


memory_pool_type_to_name = {v: k for k, v in name_to_memory_pool_type.items()}


class MemoryPoolType(Enum):
    Embed = _C.base.MemoryPoolType.Embed
    Unity = _C.base.MemoryPoolType.Unity
    ChunkIndepend = _C.base.MemoryPoolType.ChunkIndepend

    @classmethod
    def from_memory_pool_type(cls, memory_pool_type: _C.base.MemoryPoolType):
        return cls(memory_pool_type)
    
    @classmethod
    def from_name(cls, memory_pool_type_name: str):
        if memory_pool_type_name not in name_to_memory_pool_type:
            raise ValueError(f"Unsupported memory pool type name: {memory_pool_type_name}")
        return cls(name_to_memory_pool_type[memory_pool_type_name])
    

name_to_tensor_type = {
    "Default": _C.base.TensorType.Default,
}


tensor_type_to_name = {v: k for k, v in name_to_tensor_type.items()}


class TensorType(Enum):
    Default = _C.base.TensorType.Default

    @classmethod
    def from_tensor_type(cls, tensor_type: _C.base.TensorType):
        return cls(tensor_type)
    
    @classmethod
    def from_name(cls, tensor_type_name: str):
        if tensor_type_name not in name_to_tensor_type:
            raise ValueError(f"Unsupported tensor type name: {tensor_type_name}")
        return cls(name_to_tensor_type[tensor_type_name])
    
    
name_to_forward_op_type = {
    "Default": _C.base.ForwardOpType.Default,
    "OneDnn": _C.base.ForwardOpType.OneDnn,
    "XnnPack": _C.base.ForwardOpType.XnnPack,
    "QnnPack": _C.base.ForwardOpType.QnnPack,
    "Cudnn": _C.base.ForwardOpType.Cudnn,
    "AclOp": _C.base.ForwardOpType.AclOp,
    "NotSupport": _C.base.ForwardOpType.NotSupport,
}


forward_op_type_to_name = {v: k for k, v in name_to_forward_op_type.items()}


class ForwardOpType(Enum):
    Default = _C.base.ForwardOpType.Default
    OneDnn = _C.base.ForwardOpType.OneDnn
    XnnPack = _C.base.ForwardOpType.XnnPack
    QnnPack = _C.base.ForwardOpType.QnnPack
    Cudnn = _C.base.ForwardOpType.Cudnn
    AclOp = _C.base.ForwardOpType.AclOp
    NotSupport = _C.base.ForwardOpType.NotSupport

    @classmethod
    def from_forward_op_type(cls, forward_op_type: _C.base.ForwardOpType):
        return cls(forward_op_type)
    
    @classmethod
    def from_name(cls, forward_op_type_name: str):
        if forward_op_type_name not in name_to_forward_op_type:
            raise ValueError(f"Unsupported forward op type name: {forward_op_type_name}")
        return cls(name_to_forward_op_type[forward_op_type_name])
    

name_to_inference_opt_level = {
    "Zero": _C.base.InferenceOpt.Level0,
    "One": _C.base.InferenceOpt.Level1,
    "Auto": _C.base.InferenceOpt.LevelAuto,
}


inference_opt_level_to_name = {v: k for k, v in name_to_inference_opt_level.items()}


class InferenceOptLevel(Enum):
    Level0 = _C.base.InferenceOpt.Level0
    Level1 = _C.base.InferenceOpt.Level1
    LevelAuto = _C.base.InferenceOpt.LevelAuto

    @classmethod
    def from_inference_opt(cls, inference_opt_level: _C.base.InferenceOpt):
        return cls(inference_opt_level)
    
    @classmethod
    def from_name(cls, inference_opt_level_name: str):
        if inference_opt_level_name not in name_to_inference_opt_level:
            raise ValueError(f"Unsupported inference optimization level name: {inference_opt_level_name}")
        return cls(name_to_inference_opt_level[inference_opt_level_name])
    

name_to_model_type = {
    "Default": _C.base.ModelType.Default,
    "OpenVino": _C.base.ModelType.OpenVino,
    "TensorRt": _C.base.ModelType.TensorRt,
    "CoreML": _C.base.ModelType.CoreML,
    "TfLite": _C.base.ModelType.TfLite,
    "Onnx": _C.base.ModelType.Onnx,
    "AscendCL": _C.base.ModelType.AscendCL,
    "Ncnn": _C.base.ModelType.Ncnn,
    "Tnn": _C.base.ModelType.Tnn,
    "Mnn": _C.base.ModelType.Mnn,
    "PaddleLite": _C.base.ModelType.PaddleLite,
    "Rknn": _C.base.ModelType.Rknn,
    "Tvm": _C.base.ModelType.Tvm,
    "AITemplate": _C.base.ModelType.AITemplate,
    "Snpe": _C.base.ModelType.Snpe,
    "Qnn": _C.base.ModelType.Qnn,
    "Sophon": _C.base.ModelType.Sophon,
    "TorchScript": _C.base.ModelType.TorchScript,
    "TorchPth": _C.base.ModelType.TorchPth,
    "Hdf5": _C.base.ModelType.Hdf5,
    "Safetensors": _C.base.ModelType.Safetensors,
    "NeuroPilot": _C.base.ModelType.NeuroPilot,
    "NotSupport": _C.base.ModelType.NotSupport,
}


model_type_to_name = {v: k for k, v in name_to_model_type.items()}


class ModelType(Enum):
    Default = _C.base.ModelType.Default
    OpenVino = _C.base.ModelType.OpenVino
    TensorRt = _C.base.ModelType.TensorRt
    CoreML = _C.base.ModelType.CoreML
    TfLite = _C.base.ModelType.TfLite
    Onnx = _C.base.ModelType.Onnx
    AscendCL = _C.base.ModelType.AscendCL
    Ncnn = _C.base.ModelType.Ncnn
    Tnn = _C.base.ModelType.Tnn
    Mnn = _C.base.ModelType.Mnn
    PaddleLite = _C.base.ModelType.PaddleLite
    Rknn = _C.base.ModelType.Rknn
    Tvm = _C.base.ModelType.Tvm
    AITemplate = _C.base.ModelType.AITemplate
    Snpe = _C.base.ModelType.Snpe
    Qnn = _C.base.ModelType.Qnn
    Sophon = _C.base.ModelType.Sophon
    TorchScript = _C.base.ModelType.TorchScript
    TorchPth = _C.base.ModelType.TorchPth
    Hdf5 = _C.base.ModelType.Hdf5
    Safetensors = _C.base.ModelType.Safetensors
    NeuroPilot = _C.base.ModelType.NeuroPilot
    NotSupport = _C.base.ModelType.NotSupport

    @classmethod
    def from_model_type(cls, model_type: _C.base.ModelType):
        return cls(model_type)
    
    @classmethod
    def from_name(cls, model_type_name: str):
        if model_type_name not in name_to_model_type:
            raise ValueError(f"Unsupported model type name: {model_type_name}")
        return cls(name_to_model_type[model_type_name])
    
    
name_to_inference_type = {
    "Default": _C.base.InferenceType.Default,
    "OpenVino": _C.base.InferenceType.OpenVino,
    "TensorRt": _C.base.InferenceType.TensorRt,
    "CoreML": _C.base.InferenceType.CoreML,
    "TfLite": _C.base.InferenceType.TfLite,
    "OnnxRuntime": _C.base.InferenceType.OnnxRuntime,
    "AscendCL": _C.base.InferenceType.AscendCL,
    "Ncnn": _C.base.InferenceType.Ncnn,
    "Tnn": _C.base.InferenceType.Tnn,
    "Mnn": _C.base.InferenceType.Mnn,
    "PaddleLite": _C.base.InferenceType.PaddleLite,
    "Rknn": _C.base.InferenceType.Rknn,
    "Tvm": _C.base.InferenceType.Tvm,
    "AITemplate": _C.base.InferenceType.AITemplate,
    "Snpe": _C.base.InferenceType.Snpe,
    "Qnn": _C.base.InferenceType.Qnn,
    "Sophon": _C.base.InferenceType.Sophon,
    "Torch": _C.base.InferenceType.Torch,
    "TensorFlow": _C.base.InferenceType.TensorFlow,
    "NeuroPilot": _C.base.InferenceType.NeuroPilot,
    "NotSupport": _C.base.InferenceType.NotSupport,
}


inference_type_to_name = {v: k for k, v in name_to_inference_type.items()}


class InferenceType(Enum):
    Default = _C.base.InferenceType.Default
    OpenVino = _C.base.InferenceType.OpenVino
    TensorRt = _C.base.InferenceType.TensorRt
    CoreML = _C.base.InferenceType.CoreML
    TfLite = _C.base.InferenceType.TfLite
    OnnxRuntime = _C.base.InferenceType.OnnxRuntime
    AscendCL = _C.base.InferenceType.AscendCL
    Ncnn = _C.base.InferenceType.Ncnn
    Tnn = _C.base.InferenceType.Tnn
    Mnn = _C.base.InferenceType.Mnn
    PaddleLite = _C.base.InferenceType.PaddleLite
    Rknn = _C.base.InferenceType.Rknn
    Tvm = _C.base.InferenceType.Tvm
    AITemplate = _C.base.InferenceType.AITemplate
    Snpe = _C.base.InferenceType.Snpe
    Qnn = _C.base.InferenceType.Qnn
    Sophon = _C.base.InferenceType.Sophon
    Torch = _C.base.InferenceType.Torch
    TensorFlow = _C.base.InferenceType.TensorFlow
    NeuroPilot = _C.base.InferenceType.NeuroPilot
    NotSupport = _C.base.InferenceType.NotSupport

    @classmethod
    def from_inference_type(cls, inference_type: _C.base.InferenceType):
        return cls(inference_type)
    
    @classmethod
    def from_name(cls, inference_type_name: str):
        if inference_type_name not in name_to_inference_type:
            raise ValueError(f"Unsupported inference type name: {inference_type_name}")
        return cls(name_to_inference_type[inference_type_name])
    

name_to_encrypt_type = {
    "kEncryptTypeNone": _C.base.EncryptType.kEncryptTypeNone,
    "Base64": _C.base.EncryptType.Base64,
}


encrypt_type_to_name = {v: k for k, v in name_to_encrypt_type.items()}


class EncryptType(Enum):
    kEncryptTypeNone = _C.base.EncryptType.kEncryptTypeNone
    Base64 = _C.base.EncryptType.Base64

    @classmethod
    def from_encrypt_type(cls, encrypt_type: _C.base.EncryptType):
        return cls(encrypt_type)
    
    @classmethod
    def from_name(cls, encrypt_type_name: str):
        if encrypt_type_name not in name_to_encrypt_type:
            raise ValueError(f"Unsupported encrypt type name: {encrypt_type_name}")
        return cls(name_to_encrypt_type[encrypt_type_name])
    

name_to_codec_type = {
    "kCodecTypeNone": _C.base.CodecType.kCodecTypeNone,
    "OpenCV": _C.base.CodecType.OpenCV,
    "FFmpeg": _C.base.CodecType.FFmpeg,
    "Stb": _C.base.CodecType.Stb,
}


codec_type_to_name = {v: k for k, v in name_to_codec_type.items()}


class CodecType(Enum):
    kCodecTypeNone = _C.base.CodecType.kCodecTypeNone
    OpenCV = _C.base.CodecType.OpenCV
    FFmpeg = _C.base.CodecType.FFmpeg
    Stb = _C.base.CodecType.Stb

    @classmethod
    def from_codec_type(cls, codec_type: _C.base.CodecType):
        return cls(codec_type)
    
    @classmethod
    def from_name(cls, codec_type_name: str):
        if codec_type_name not in name_to_codec_type:
            raise ValueError(f"Unsupported codec type name: {codec_type_name}")
        return cls(name_to_codec_type[codec_type_name])
    

name_to_codec_flag = {
    "Image": _C.base.CodecFlag.Image,
    "Images": _C.base.CodecFlag.Images,
    "Video": _C.base.CodecFlag.Video,
    "Camera": _C.base.CodecFlag.Camera,
    "Other": _C.base.CodecFlag.Other,
}


codec_flag_to_name = {v: k for k, v in name_to_codec_flag.items()}


class CodecFlag(Enum):
    Image = _C.base.CodecFlag.Image
    Images = _C.base.CodecFlag.Images
    Video = _C.base.CodecFlag.Video  
    Camera = _C.base.CodecFlag.Camera
    Other = _C.base.CodecFlag.Other

    @classmethod
    def from_codec_flag(cls, codec_flag: _C.base.CodecFlag):
        return cls(codec_flag)
    
    @classmethod 
    def from_name(cls, codec_flag_name: str):
        if codec_flag_name not in name_to_codec_flag:
            raise ValueError(f"Unsupported codec flag name: {codec_flag_name}")
        return cls(name_to_codec_flag[codec_flag_name])
    
    
name_to_parallel_type = {
    "kParallelTypeNone": _C.base.ParallelType.kParallelTypeNone,
    "Sequential": _C.base.ParallelType.Sequential,
    "Task": _C.base.ParallelType.Task,
    "Pipeline": _C.base.ParallelType.Pipeline,
}

parallel_type_to_name = {v: k for k, v in name_to_parallel_type.items()}


class ParallelType(Enum):
    kParallelTypeNone = _C.base.ParallelType.kParallelTypeNone
    Sequential = _C.base.ParallelType.Sequential
    Task = _C.base.ParallelType.Task
    Pipeline = _C.base.ParallelType.Pipeline

    @classmethod
    def from_parallel_type(cls, parallel_type: _C.base.ParallelType):
        return cls(parallel_type)
    
    @classmethod
    def from_name(cls, parallel_type_name: str):
        if parallel_type_name not in name_to_parallel_type:
            raise ValueError(f"Unsupported parallel type name: {parallel_type_name}")
        return cls(name_to_parallel_type[parallel_type_name])
    
    
name_to_edge_type = {
    "Fixed": _C.base.EdgeType.Fixed,
    "Pipeline": _C.base.EdgeType.Pipeline,
}


edge_type_to_name = {v: k for k, v in name_to_edge_type.items()}


class EdgeType(Enum):
    Fixed = _C.base.EdgeType.Fixed
    Pipeline = _C.base.EdgeType.Pipeline

    @classmethod
    def from_edge_type(cls, edge_type: _C.base.EdgeType):
        return cls(edge_type)
    
    @classmethod
    def from_name(cls, edge_type_name: str):
        if edge_type_name not in name_to_edge_type:
            raise ValueError(f"Unsupported edge type name: {edge_type_name}")
        return cls(name_to_edge_type[edge_type_name])

    
name_to_edge_update_flag = {
    "Complete": _C.base.EdgeUpdateFlag.Complete,
    "Terminate": _C.base.EdgeUpdateFlag.Terminate,
    "Error": _C.base.EdgeUpdateFlag.Error,
}


edge_update_flag_to_name = {v: k for k, v in name_to_edge_update_flag.items()}

class EdgeUpdateFlag(Enum):
    Complete = _C.base.EdgeUpdateFlag.Complete
    Terminate = _C.base.EdgeUpdateFlag.Terminate
    Error = _C.base.EdgeUpdateFlag.Error

    @classmethod
    def from_edge_update_flag(cls, edge_update_flag: _C.base.EdgeUpdateFlag):
        return cls(edge_update_flag)
    
    @classmethod
    def from_name(cls, edge_update_flag_name: str):
        if edge_update_flag_name not in name_to_edge_update_flag:
            raise ValueError(f"Unsupported edge update flag name: {edge_update_flag_name}")
        return cls(name_to_edge_update_flag[edge_update_flag_name])
    

name_to_node_color_type = {
    "White": _C.base.NodeColorType.White,
    "Gray": _C.base.NodeColorType.Gray,
    "Black": _C.base.NodeColorType.Black,
}


node_color_type_to_name = {v: k for k, v in name_to_node_color_type.items()}


class NodeColorType(Enum):
    White = _C.base.NodeColorType.White
    Gray = _C.base.NodeColorType.Gray
    Black = _C.base.NodeColorType.Black

    @classmethod
    def from_node_color_type(cls, node_color_type: _C.base.NodeColorType):
        return cls(node_color_type)
    
    @classmethod
    def from_name(cls, node_color_type_name: str):
        if node_color_type_name not in name_to_node_color_type:
            raise ValueError(f"Unsupported node color type name: {node_color_type_name}")
        return cls(name_to_node_color_type[node_color_type_name])
    

name_to_topo_sort_type = {
    "BFS": _C.base.TopoSortType.BFS,
    "DFS": _C.base.TopoSortType.DFS,
}


topo_sort_type_to_name = {v: k for k, v in name_to_topo_sort_type.items()}


class TopoSortType(Enum):
    BFS = _C.base.TopoSortType.BFS
    DFS = _C.base.TopoSortType.DFS

    @classmethod
    def from_topo_sort_type(cls, topo_sort_type: _C.base.TopoSortType):
        return cls(topo_sort_type)
    
    @classmethod
    def from_name(cls, topo_sort_type_name: str):
        if topo_sort_type_name not in name_to_topo_sort_type:
            raise ValueError(f"Unsupported topo sort type name: {topo_sort_type_name}")
        return cls(name_to_topo_sort_type[topo_sort_type_name])
    
    
name_to_status_code =  {
    "Ok": _C.base.StatusCode.Ok,
    "ErrorUnknown": _C.base.StatusCode.ErrorUnknown,
    "ErrorOutOfMemory": _C.base.StatusCode.ErrorOutOfMemory,
    "ErrorNotSupport": _C.base.StatusCode.ErrorNotSupport,
    "ErrorNotImplement": _C.base.StatusCode.ErrorNotImplement,
    "ErrorInvalidValue": _C.base.StatusCode.ErrorInvalidValue,
    "ErrorInvalidParam": _C.base.StatusCode.ErrorInvalidParam,
    "ErrorNullParam": _C.base.StatusCode.ErrorNullParam,
    "ErrorThreadPool": _C.base.StatusCode.ErrorThreadPool,
    "ErrorIO": _C.base.StatusCode.ErrorIO,
    "ErrorDeviceCpu": _C.base.StatusCode.ErrorDeviceCpu,
    "ErrorDeviceArm": _C.base.StatusCode.ErrorDeviceArm,
    "ErrorDeviceX86": _C.base.StatusCode.ErrorDeviceX86,
    "ErrorDeviceRiscV": _C.base.StatusCode.ErrorDeviceRiscV,
    "ErrorDeviceCuda": _C.base.StatusCode.ErrorDeviceCuda,
    "ErrorDeviceRocm": _C.base.StatusCode.ErrorDeviceRocm,
    "ErrorDeviceSyCL": _C.base.StatusCode.ErrorDeviceSyCL,
    "ErrorDeviceOpenCL": _C.base.StatusCode.ErrorDeviceOpenCL,
    "ErrorDeviceOpenGL": _C.base.StatusCode.ErrorDeviceOpenGL,
    "ErrorDeviceMetal": _C.base.StatusCode.ErrorDeviceMetal,
    "ErrorDeviceVulkan": _C.base.StatusCode.ErrorDeviceVulkan,
    "ErrorDeviceHexagon": _C.base.StatusCode.ErrorDeviceHexagon,
    "ErrorDeviceMtkVpu": _C.base.StatusCode.ErrorDeviceMtkVpu,
    "ErrorDeviceAscendCL": _C.base.StatusCode.ErrorDeviceAscendCL,
    "ErrorDeviceAppleNpu": _C.base.StatusCode.ErrorDeviceAppleNpu,
    "ErrorDeviceRkNpu": _C.base.StatusCode.ErrorDeviceRkNpu,
    "ErrorDeviceQualcommNpu": _C.base.StatusCode.ErrorDeviceQualcommNpu,
    "ErrorDeviceMtkNpu": _C.base.StatusCode.ErrorDeviceMtkNpu,
    "ErrorDeviceSophonNpu": _C.base.StatusCode.ErrorDeviceSophonNpu,
    "ErrorOpAscendCL": _C.base.StatusCode.ErrorOpAscendCL,
    "ErrorInferenceDefault": _C.base.StatusCode.ErrorInferenceDefault,
    "ErrorInferenceOpenVino": _C.base.StatusCode.ErrorInferenceOpenVino,
    "ErrorInferenceTensorRt": _C.base.StatusCode.ErrorInferenceTensorRt,
    "ErrorInferenceCoreML": _C.base.StatusCode.ErrorInferenceCoreML,
    "ErrorInferenceTfLite": _C.base.StatusCode.ErrorInferenceTfLite,
    "ErrorInferenceOnnxRuntime": _C.base.StatusCode.ErrorInferenceOnnxRuntime,
    "ErrorInferenceAscendCL": _C.base.StatusCode.ErrorInferenceAscendCL,
    "ErrorInferenceNcnn": _C.base.StatusCode.ErrorInferenceNcnn,
    "ErrorInferenceTnn": _C.base.StatusCode.ErrorInferenceTnn,
    "ErrorInferenceMnn": _C.base.StatusCode.ErrorInferenceMnn,
    "ErrorInferencePaddleLite": _C.base.StatusCode.ErrorInferencePaddleLite,
    "ErrorInferenceRknn": _C.base.StatusCode.ErrorInferenceRknn,
    "ErrorInferenceTvm": _C.base.StatusCode.ErrorInferenceTvm,
    "ErrorInferenceAITemplate": _C.base.StatusCode.ErrorInferenceAITemplate,
    "ErrorInferenceSnpe": _C.base.StatusCode.ErrorInferenceSnpe,
    "ErrorInferenceQnn": _C.base.StatusCode.ErrorInferenceQnn,
    "ErrorInferenceSophon": _C.base.StatusCode.ErrorInferenceSophon,
    "ErrorInferenceTorch": _C.base.StatusCode.ErrorInferenceTorch,
    "ErrorInferenceTensorFlow": _C.base.StatusCode.ErrorInferenceTensorFlow,
    "ErrorInferenceNeuroPilot": _C.base.StatusCode.ErrorInferenceNeuroPilot,
    "ErrorDag": _C.base.StatusCode.ErrorDag,
}


status_code_to_name = {v: k for k, v in name_to_status_code.items()}


class StatusCode(Enum):
    """
    default member:
        value: _C.base.StatusCode
        name: str
    """ 
    Ok = _C.base.StatusCode.Ok
    ErrorUnknown = _C.base.StatusCode.ErrorUnknown
    ErrorOutOfMemory = _C.base.StatusCode.ErrorOutOfMemory
    ErrorNotSupport = _C.base.StatusCode.ErrorNotSupport
    ErrorNotImplement = _C.base.StatusCode.ErrorNotImplement
    ErrorInvalidValue = _C.base.StatusCode.ErrorInvalidValue
    ErrorInvalidParam = _C.base.StatusCode.ErrorInvalidParam
    ErrorNullParam = _C.base.StatusCode.ErrorNullParam
    ErrorThreadPool = _C.base.StatusCode.ErrorThreadPool
    ErrorIO = _C.base.StatusCode.ErrorIO
    ErrorDeviceCpu = _C.base.StatusCode.ErrorDeviceCpu
    ErrorDeviceArm = _C.base.StatusCode.ErrorDeviceArm
    ErrorDeviceX86 = _C.base.StatusCode.ErrorDeviceX86
    ErrorDeviceRiscV = _C.base.StatusCode.ErrorDeviceRiscV
    ErrorDeviceCuda = _C.base.StatusCode.ErrorDeviceCuda
    ErrorDeviceRocm = _C.base.StatusCode.ErrorDeviceRocm
    ErrorDeviceSyCL = _C.base.StatusCode.ErrorDeviceSyCL
    ErrorDeviceOpenCL = _C.base.StatusCode.ErrorDeviceOpenCL
    ErrorDeviceOpenGL = _C.base.StatusCode.ErrorDeviceOpenGL
    ErrorDeviceMetal = _C.base.StatusCode.ErrorDeviceMetal
    ErrorDeviceVulkan = _C.base.StatusCode.ErrorDeviceVulkan
    ErrorDeviceHexagon = _C.base.StatusCode.ErrorDeviceHexagon
    ErrorDeviceMtkVpu = _C.base.StatusCode.ErrorDeviceMtkVpu
    ErrorDeviceAscendCL = _C.base.StatusCode.ErrorDeviceAscendCL
    ErrorDeviceAppleNpu = _C.base.StatusCode.ErrorDeviceAppleNpu
    ErrorDeviceRkNpu = _C.base.StatusCode.ErrorDeviceRkNpu
    ErrorDeviceQualcommNpu = _C.base.StatusCode.ErrorDeviceQualcommNpu
    ErrorDeviceMtkNpu = _C.base.StatusCode.ErrorDeviceMtkNpu
    ErrorDeviceSophonNpu = _C.base.StatusCode.ErrorDeviceSophonNpu
    ErrorOpAscendCL = _C.base.StatusCode.ErrorOpAscendCL
    ErrorInferenceDefault = _C.base.StatusCode.ErrorInferenceDefault
    ErrorInferenceOpenVino = _C.base.StatusCode.ErrorInferenceOpenVino
    ErrorInferenceTensorRt = _C.base.StatusCode.ErrorInferenceTensorRt
    ErrorInferenceCoreML = _C.base.StatusCode.ErrorInferenceCoreML
    ErrorInferenceTfLite = _C.base.StatusCode.ErrorInferenceTfLite
    ErrorInferenceOnnxRuntime = _C.base.StatusCode.ErrorInferenceOnnxRuntime
    ErrorInferenceAscendCL = _C.base.StatusCode.ErrorInferenceAscendCL
    ErrorInferenceNcnn = _C.base.StatusCode.ErrorInferenceNcnn
    ErrorInferenceTnn = _C.base.StatusCode.ErrorInferenceTnn
    ErrorInferenceMnn = _C.base.StatusCode.ErrorInferenceMnn
    ErrorInferencePaddleLite = _C.base.StatusCode.ErrorInferencePaddleLite
    ErrorInferenceRknn = _C.base.StatusCode.ErrorInferenceRknn
    ErrorInferenceTvm = _C.base.StatusCode.ErrorInferenceTvm
    ErrorInferenceAITemplate = _C.base.StatusCode.ErrorInferenceAITemplate
    ErrorInferenceSnpe = _C.base.StatusCode.ErrorInferenceSnpe
    ErrorInferenceQnn = _C.base.StatusCode.ErrorInferenceQnn
    ErrorInferenceSophon = _C.base.StatusCode.ErrorInferenceSophon
    ErrorInferenceTorch = _C.base.StatusCode.ErrorInferenceTorch
    ErrorInferenceTensorFlow = _C.base.StatusCode.ErrorInferenceTensorFlow
    ErrorInferenceNeuroPilot = _C.base.StatusCode.ErrorInferenceNeuroPilot
    ErrorDag = _C.base.StatusCode.ErrorDag

    @classmethod
    def from_status_code(cls, status_code: _C.base.StatusCode):
        return cls(status_code)
    
    @classmethod
    def from_name(cls, status_code_name: str):
        if status_code_name not in name_to_status_code:
            return cls(StatusCode.ErrorUnknown)
        else:
            return cls(name_to_status_code[status_code_name])
    
class Status(_C.base.Status):
    def __init__(self, status_code: StatusCode):
        super().__init__(status_code.value)

    @classmethod
    def from_status(cls, status: _C.base.Status):
        return cls(status.get_code())
    
    @classmethod
    def from_status_code(cls, status_code: _C.base.StatusCode):
        return cls(status_code)
    
    @classmethod
    def from_name(cls, status_name: str):
        if status_name not in name_to_status_code:
            return cls(StatusCode.ErrorUnknown)
        else:
            return cls(name_to_status_code[status_name])

    def get_code_name(self):
        return status_code_to_name[self.get_code()]
    

name_to_pixel_type = {
    "GRAY": _C.base.PixelType.GRAY,
    "RGB": _C.base.PixelType.RGB,
    "BGR": _C.base.PixelType.BGR,
    "RGBA": _C.base.PixelType.RGBA,
    "BGRA": _C.base.PixelType.BGRA,
    "NotSupport": _C.base.PixelType.NotSupport,
}

pixel_type_to_name = {v: k for k, v in name_to_pixel_type.items()}

class PixelType(Enum):
    GRAY = _C.base.PixelType.GRAY
    RGB = _C.base.PixelType.RGB
    BGR = _C.base.PixelType.BGR
    RGBA = _C.base.PixelType.RGBA
    BGRA = _C.base.PixelType.BGRA
    NotSupport = _C.base.PixelType.NotSupport

    @classmethod
    def from_pixel_type(cls, pixel_type: _C.base.PixelType):
        return cls(pixel_type)
    
    @classmethod
    def from_name(cls, pixel_type_name: str):
        if pixel_type_name not in name_to_pixel_type:
            raise ValueError(f"Unsupported pixel type name: {pixel_type_name}")
        return cls(name_to_pixel_type[pixel_type_name])


name_to_cvt_color_type = {
    "RGB2GRAY": _C.base.CvtColorType.RGB2GRAY,
    "BGR2GRAY": _C.base.CvtColorType.BGR2GRAY,
    "RGBA2GRAY": _C.base.CvtColorType.RGBA2GRAY,
    "BGRA2GRAY": _C.base.CvtColorType.BGRA2GRAY,
    "GRAY2RGB": _C.base.CvtColorType.GRAY2RGB,
    "BGR2RGB": _C.base.CvtColorType.BGR2RGB,
    "RGBA2RGB": _C.base.CvtColorType.RGBA2RGB,
    "BGRA2RGB": _C.base.CvtColorType.BGRA2RGB,
    "GRAY2BGR": _C.base.CvtColorType.GRAY2BGR,
    "RGB2BGR": _C.base.CvtColorType.RGB2BGR,
    "RGBA2BGR": _C.base.CvtColorType.RGBA2BGR,
    "BGRA2BGR": _C.base.CvtColorType.BGRA2BGR,
    "GRAY2RGBA": _C.base.CvtColorType.GRAY2RGBA,
    "RGB2RGBA": _C.base.CvtColorType.RGB2RGBA,
    "BGR2RGBA": _C.base.CvtColorType.BGR2RGBA,
    "BGRA2RGBA": _C.base.CvtColorType.BGRA2RGBA,
    "GRAY2BGRA": _C.base.CvtColorType.GRAY2BGRA,
    "RGB2BGRA": _C.base.CvtColorType.RGB2BGRA,
    "BGR2BGRA": _C.base.CvtColorType.BGR2BGRA,
    "RGBA2BGRA": _C.base.CvtColorType.RGBA2BGRA,
    "NotSupport": _C.base.CvtColorType.NotSupport,
}

cvt_color_type_to_name = {v: k for k, v in name_to_cvt_color_type.items()}

class CvtColorType(Enum):
    RGB2GRAY = _C.base.CvtColorType.RGB2GRAY
    BGR2GRAY = _C.base.CvtColorType.BGR2GRAY
    RGBA2GRAY = _C.base.CvtColorType.RGBA2GRAY
    BGRA2GRAY = _C.base.CvtColorType.BGRA2GRAY
    GRAY2RGB = _C.base.CvtColorType.GRAY2RGB
    BGR2RGB = _C.base.CvtColorType.BGR2RGB
    RGBA2RGB = _C.base.CvtColorType.RGBA2RGB
    BGRA2RGB = _C.base.CvtColorType.BGRA2RGB
    GRAY2BGR = _C.base.CvtColorType.GRAY2BGR
    RGB2BGR = _C.base.CvtColorType.RGB2BGR
    RGBA2BGR = _C.base.CvtColorType.RGBA2BGR
    BGRA2BGR = _C.base.CvtColorType.BGRA2BGR
    GRAY2RGBA = _C.base.CvtColorType.GRAY2RGBA
    RGB2RGBA = _C.base.CvtColorType.RGB2RGBA
    BGR2RGBA = _C.base.CvtColorType.BGR2RGBA
    BGRA2RGBA = _C.base.CvtColorType.BGRA2RGBA
    GRAY2BGRA = _C.base.CvtColorType.GRAY2BGRA
    RGB2BGRA = _C.base.CvtColorType.RGB2BGRA
    BGR2BGRA = _C.base.CvtColorType.BGR2BGRA
    RGBA2BGRA = _C.base.CvtColorType.RGBA2BGRA
    NotSupport = _C.base.CvtColorType.NotSupport

    @classmethod
    def from_cvt_color_type(cls, cvt_color_type: _C.base.CvtColorType):
        return cls(cvt_color_type)
    
    @classmethod
    def from_name(cls, cvt_color_type_name: str):
        if cvt_color_type_name not in name_to_cvt_color_type:
            raise ValueError(f"Unsupported cvt color type name: {cvt_color_type_name}")
        return cls(name_to_cvt_color_type[cvt_color_type_name])

name_to_interp_type = {
    "Nearst": _C.base.InterpType.Nearst,
    "Linear": _C.base.InterpType.Linear,
    "Cubic": _C.base.InterpType.Cubic,
    "Arer": _C.base.InterpType.Arer,
    "NotSupport": _C.base.InterpType.NotSupport,
}

interp_type_to_name = {v: k for k, v in name_to_interp_type.items()}

class InterpType(Enum):
    Nearst = _C.base.InterpType.Nearst
    Linear = _C.base.InterpType.Linear
    Cubic = _C.base.InterpType.Cubic
    Arer = _C.base.InterpType.Arer
    NotSupport = _C.base.InterpType.NotSupport

    @classmethod
    def from_interp_type(cls, interp_type: _C.base.InterpType):
        return cls(interp_type)
    
    @classmethod
    def from_name(cls, interp_type_name: str):
        if interp_type_name not in name_to_interp_type:
            raise ValueError(f"Unsupported interp type name: {interp_type_name}")
        return cls(name_to_interp_type[interp_type_name])


name_to_border_type = {
    "Constant": _C.base.BorderType.Constant,
    "Reflect": _C.base.BorderType.Reflect,
    "Edge": _C.base.BorderType.Edge,
    "NotSupport": _C.base.BorderType.NotSupport,
}

border_type_to_name = {v: k for k, v in name_to_border_type.items()}

class BorderType(Enum):
    Constant = _C.base.BorderType.Constant
    Reflect = _C.base.BorderType.Reflect
    Edge = _C.base.BorderType.Edge
    NotSupport = _C.base.BorderType.NotSupport

    @classmethod
    def from_border_type(cls, border_type: _C.base.BorderType):
        return cls(border_type)
    
    @classmethod
    def from_name(cls, border_type_name: str):
        if border_type_name not in name_to_border_type:
            raise ValueError(f"Unsupported border type name: {border_type_name}")
        return cls(name_to_border_type[border_type_name])


class TimeProfiler:
    def __init__(self):
        self._profiler = _C.base.TimeProfiler()
    
    def reset(self):
        self._profiler.reset()
    
    def start(self, key: str):
        self._profiler.start(key)
    
    def end(self, key: str):
        self._profiler.end(key)
    
    def print(self, title: str = ""):
        self._profiler.print(title)
    
    def print_index(self, title: str, index: int):
        self._profiler.print_index(title, index)
    
    def print_remove_warmup(self, title: str, warmup_times: int):
        self._profiler.print_remove_warmup(title, warmup_times)


def time_profiler_reset():
    _C.base.time_profiler_reset()

def time_point_start(key: str):
    _C.base.time_point_start(key)

def time_point_end(key: str):
    _C.base.time_point_end(key)

def time_profiler_print(title: str = ""):
    _C.base.time_profiler_print(title)

def time_profiler_print_index(title: str, index: int):
    _C.base.time_profiler_print_index(title, index)

def time_profiler_print_remove_warmup(title: str, warmup_times: int):
    _C.base.time_profiler_print_remove_warmup(title, warmup_times)


class Param(_C.base.Param):
    def __init__(self):
        super().__init__()
        self._default_dic = {}

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
    
    def serialize(self, value):
        if isinstance(value, dict):
            self._default_dic = value
        elif isinstance(value, str):
            try:
                self._default_dic = json.loads(value)
            except json.JSONDecodeError:
                print(f"Failed to deserialize string: {value}")
        else:
            raise ValueError(f"Unsupported value type: {type(value)}")
        
    def deserialize(self, value):
        if isinstance(value, dict):
            value.update(self._default_dic)
        elif isinstance(value, str):
            try:
                with open(value, 'w') as f:
                    json.dump(self._default_dic, f, indent=4)
            except IOError:
                print(f"Failed to write file: {value}")
        else:
            raise ValueError(f"Unsupported value type: {type(value)}")
        
    def get_default_dict(self):
        return self._default_dic
