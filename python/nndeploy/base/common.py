
import nndeploy._nndeploy_internal as _C

from enum import Enum
from typing import Union
import numpy as np
import json


all_type_enum = []


name_to_data_type_code = {
    "Uint": _C.base.DataTypeCode.Uint,
    "Int": _C.base.DataTypeCode.Int,
    "Fp": _C.base.DataTypeCode.Fp,
    "BFp": _C.base.DataTypeCode.BFp,
    "OpaqueHandle": _C.base.DataTypeCode.OpaqueHandle,
    "NotSupport": _C.base.DataTypeCode.NotSupport,
}


data_type_code_to_name = {v: k for k, v in name_to_data_type_code.items()}


class DataTypeCode(_C.base.DataTypeCode):
    Uint = _C.base.DataTypeCode.Uint
    Int = _C.base.DataTypeCode.Int
    Fp = _C.base.DataTypeCode.Fp
    BFp = _C.base.DataTypeCode.BFp
    OpaqueHandle = _C.base.DataTypeCode.OpaqueHandle
    NotSupport = _C.base.DataTypeCode.NotSupport
    
    @classmethod
    def from_name(cls, name: str):
        if name not in name_to_data_type_code:
            raise ValueError(f"Unsupported data type code: {name}")
        else:   
            return cls(name_to_data_type_code[name])         


class DataType(_C.base.DataType):
    def __init__(self, *args, **kwargs):
        """
        Constructs a DataType object.

        The constructor can be called in the following ways:
        1. DataType(numpy_dtype): Constructs a DataType from a numpy dtype.
        2. DataType(name): Constructs a DataType from a string name (e.g., "float32").
        3. DataType(data_type_code, bits=32, lanes=1): Constructs a DataType from a DataTypeCode enum value, with optional bits and lanes.
        4. DataType(data_type_code, bits, lanes=1): Constructs a DataType from a DataTypeCode enum value and bits, with optional lanes.
        5. DataType(data_type_code, bits, lanes): Constructs a DataType from a DataTypeCode enum value, bits, and lanes.
        6. DataType(type): Constructs a DataType from a Python type (e.g., np.float32).
        """
        if len(args) == 1 and isinstance(args[0], np.dtype):
            data_type = DataType.from_numpy_dtype(args[0])
            data_type_code = data_type.get_data_type_code()
            bits = data_type.get_bits()
            lanes = data_type.get_lanes()
        elif len(args) == 1 and isinstance(args[0], str):
            data_type = DataType.from_name(args[0])
            data_type_code = data_type.get_data_type_code()
            bits = data_type.get_bits()
            lanes = data_type.get_lanes()
        elif len(args) == 1 and isinstance(args[0], _C.base.DataTypeCode):
            data_type_code = args[0]
            bits = kwargs.get("bits", 32)
            lanes = kwargs.get("lanes", 1)
        elif len(args) == 2 and isinstance(args[0], _C.base.DataTypeCode) and isinstance(args[1], int):
            data_type_code = args[0]
            bits = args[1]
            lanes = kwargs.get("lanes", 1)
        elif len(args) == 3 and isinstance(args[0], _C.base.DataTypeCode) and isinstance(args[1], int) and isinstance(args[2], int):
            data_type_code = args[0]
            bits = args[1]
            lanes = args[2]
        elif len(args) == 1 and isinstance(args[0], DataTypeCode):
            data_type_code = args[0]
            bits = kwargs.get("bits", 32)
            lanes = kwargs.get("lanes", 1)
        elif len(args) == 2 and isinstance(args[0], DataTypeCode) and isinstance(args[1], int):
            data_type_code = args[0]
            bits = args[1]
            lanes = kwargs.get("lanes", 1)
        elif len(args) == 3 and isinstance(args[0], DataTypeCode) and isinstance(args[1], int) and isinstance(args[2], int):
            data_type_code = args[0]
            bits = args[1]
            lanes = args[2]
        elif len(args) == 1 and isinstance(args[0], type):
            data_type = DataType.from_numpy_dtype(args[0])
            data_type_code = data_type.get_data_type_code()
            bits = data_type.get_bits()
            lanes = data_type.get_lanes()
        else:
            raise ValueError("Invalid arguments for DataType constructor")
        super().__init__(data_type_code, bits, lanes)
    
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
        return self.size()
    
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


# {
#     "kDataTypeCodeUint8": [
#         "kDataTypeCodeUint8",
#         "kDataTypeCodeUint16",
#         "kDataTypeCodeUint32",
#         "kDataTypeCodeUint64",
#         "kDataTypeCodeInt8",
#         "kDataTypeCodeInt16",
#         "kDataTypeCodeInt32",
#         "kDataTypeCodeInt64",
#         "kDataTypeCodeFp16",
#         "kDataTypeCodeFp32",
#         "kDataTypeCodeFp64",
#         "kDataTypeCodeBFp16",
#         "kDataTypeCodeOpaqueHandle",
#         "kDataTypeCodeNotSupport"
#     ]
#     "kDataTypeCodeUint16": [
#         "kDataTypeCodeUint8",
#         "kDataTypeCodeUint16",
#         "kDataTypeCodeUint32",
#         "kDataTypeCodeUint64",
#         "kDataTypeCodeInt8",
#         "kDataTypeCodeInt16",
#         "kDataTypeCodeInt32",
#         "kDataTypeCodeInt64",
#         "kDataTypeCodeFp16",
#         "kDataTypeCodeFp32",
#         "kDataTypeCodeFp64",
#         "kDataTypeCodeBFp16",
#         "kDataTypeCodeOpaqueHandle",
#         "kDataTypeCodeNotSupport"
#     ]
#     ...
# }
def get_data_type_enum_json():
    enum_list = [
        "kDataTypeCodeUint8",
        "kDataTypeCodeUint16", 
        "kDataTypeCodeUint32",
        "kDataTypeCodeUint64",
        "kDataTypeCodeInt8",
        "kDataTypeCodeInt16", 
        "kDataTypeCodeInt32",
        "kDataTypeCodeInt64",
        "kDataTypeCodeFp16",
        "kDataTypeCodeFp32",
        "kDataTypeCodeFp64",
        "kDataTypeCodeBFp16",
        "kDataTypeCodeOpaqueHandle",
        "kDataTypeCodeNotSupport"
    ]
    data_type_enum = {}
    for single_enum in enum_list:
        data_type_enum[f"{single_enum}"] = enum_list
    return data_type_enum


all_type_enum.append(get_data_type_enum_json)


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


class DeviceTypeCode(_C.base.DeviceTypeCode):
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
    def from_name(cls, device_name: str):
        if device_name not in name_to_device_type_code:
            raise ValueError(f"Unsupported device type code: {device_name}")
        return cls(name_to_device_type_code[device_name])


class DeviceType(_C.base.DeviceType):
    def __init__(self, *args, **kwargs):
        """
        Constructs a DeviceType object.

        The constructor can be called in the following ways:
        1. DeviceType(device_name_and_id): Constructs a DeviceType from a string in the format "device_name:device_id(optional)" (e.g., "cuda:0").
        2. DeviceType(device_name, device_id): Constructs a DeviceType from a device_name and an integer device ID.
        3. DeviceType(device_type_code): Constructs a DeviceType from a DeviceTypeCode enum value. The device ID defaults to 0.
        4. DeviceType(device_type_code, device_id): Constructs a DeviceType from a DeviceTypeCode enum value and an integer device ID.
        5. DeviceType(): Constructs a DeviceType object with default values(cpu:0).
        """
        if len(args) == 1 and isinstance(args[0], str):
            device_name_and_id = args[0].split(":")
            device_name = device_name_and_id[0]
            if len(device_name_and_id) == 1:
                device_id = 0
            else:
                device_id = int(device_name_and_id[1])
            if device_name not in name_to_device_type_code:
                raise ValueError(f"Unsupported device type code: {device_name}")
            code = name_to_device_type_code[device_name]
            super().__init__(code, device_id)
        elif len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], int):
            device_name = args[0]
            if device_name not in name_to_device_type_code:
                raise ValueError(f"Unsupported device type code: {device_name}")
            code = name_to_device_type_code[device_name]
            device_id = args[1]
            super().__init__(code, device_id)
        elif len(args) == 1 and isinstance(args[0], _C.base.DeviceTypeCode):
            device_type_code = args[0]
            device_id = 0
            super().__init__(device_type_code, device_id)
        elif len(args) == 2 and isinstance(args[0], _C.base.DeviceTypeCode) and isinstance(args[1], int):
            device_type_code = args[0]
            device_id = args[1] 
            super().__init__(device_type_code, device_id)
        elif len(args) == 1 and isinstance(args[0], DeviceTypeCode):
            device_type_code = args[0]
            device_id = 0
            super().__init__(device_type_code, device_id)
        elif len(args) == 2 and isinstance(args[0], DeviceTypeCode) and isinstance(args[1], int):
            device_type_code = args[0]
            device_id = args[1] 
            super().__init__(device_type_code, device_id)
        elif len(args) == 0:
            super().__init__()
        else:
            raise ValueError("Invalid arguments for DeviceType constructor")
    
    def get_device_type_code(self):
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
    "NDCHW": _C.base.DataFormat.NDCHW,
    "Auto": _C.base.DataFormat.Auto,
    "NotSupport": _C.base.DataFormat.NotSupport,
}


data_format_to_name = {v: k for k, v in name_to_data_format.items()}


class DataFormat(_C.base.DataFormat):
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
    NDCHW = _C.base.DataFormat.NDCHW
    Auto = _C.base.DataFormat.Auto
    NotSupport = _C.base.DataFormat.NotSupport
   
    @classmethod
    def from_name(cls, data_format_name: str):
        if data_format_name not in name_to_data_format:
            raise ValueError(f"Unsupported data format name: {data_format_name}")
        return cls(name_to_data_format[data_format_name])


def get_data_format_enum_json():
    enum_list = []
    for data_format_name, data_format_code in name_to_data_format.items():
        data_format_str = _C.base.data_format_to_string(data_format_code)
        enum_list.append(data_format_str)
    data_format_enum = {}
    for single_enum in enum_list:
        data_format_enum[f"{single_enum}"] = enum_list
    return data_format_enum


all_type_enum.append(get_data_format_enum_json)


name_to_precision_type = {
    "BFp16": _C.base.PrecisionType.BFp16,
    "Fp16": _C.base.PrecisionType.Fp16,
    "Fp32": _C.base.PrecisionType.Fp32,
    "Fp64": _C.base.PrecisionType.Fp64,
    "NotSupport": _C.base.PrecisionType.NotSupport,
}


precision_type_to_name = {v: k for k, v in name_to_precision_type.items()}


def get_precision_type_enum_json():
    enum_list = []
    for precision_type_name, precision_type_code in name_to_precision_type.items():
        precision_type_str = _C.base.precision_type_to_string(precision_type_code)
        enum_list.append(precision_type_str)
    precision_type_enum = {}
    for single_enum in enum_list:
        precision_type_enum[f"{single_enum}"] = enum_list
    return precision_type_enum


all_type_enum.append(get_precision_type_enum_json)


class PrecisionType(_C.base.PrecisionType):
    BFp16 = _C.base.PrecisionType.BFp16
    Fp16 = _C.base.PrecisionType.Fp16
    Fp32 = _C.base.PrecisionType.Fp32
    Fp64 = _C.base.PrecisionType.Fp64
    NotSupport = _C.base.PrecisionType.NotSupport

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


def get_power_type_enum_json():
    enum_list = []
    for power_type_name, power_type_code in name_to_power_type.items():
        power_type_str = _C.base.power_type_to_string(power_type_code)
        enum_list.append(power_type_str)
    power_type_enum = {}
    for single_enum in enum_list:
        power_type_enum[f"{single_enum}"] = enum_list
    return power_type_enum


all_type_enum.append(get_power_type_enum_json)


class PowerType(_C.base.PowerType):
    High = _C.base.PowerType.High
    Normal = _C.base.PowerType.Normal
    Low = _C.base.PowerType.Low
    NotSupport = _C.base.PowerType.NotSupport
    
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


def get_share_memory_type_enum_json():
    enum_list = []
    for share_memory_type_name, share_memory_type_code in name_to_share_memory_type.items():
        share_memory_type_str = _C.base.share_memory_type_to_string(share_memory_type_code)
        enum_list.append(share_memory_type_str)
    share_memory_type_enum = {}
    for single_enum in enum_list:
        share_memory_type_enum[f"{single_enum}"] = enum_list
    return share_memory_type_enum


all_type_enum.append(get_share_memory_type_enum_json)


class ShareMemoryType(_C.base.ShareMemoryType):
    NoShare = _C.base.ShareMemoryType.NoShare
    ShareFromExternal = _C.base.ShareMemoryType.ShareFromExternal
    NotSupport = _C.base.ShareMemoryType.NotSupport
    
    @classmethod
    def from_name(cls, share_memory_type_name: str):
        if share_memory_type_name not in name_to_share_memory_type:
            raise ValueError(f"Unsupported share memory type name: {share_memory_type_name}")
        return cls(name_to_share_memory_type[share_memory_type_name])


name_to_memory_type = {
    "kMemoryTypeNone": _C.base.MemoryType.kMemoryTypeNone,
    "Allocate": _C.base.MemoryType.Allocate,
    "External": _C.base.MemoryType.External,
    "Mapped": _C.base.MemoryType.Mapped,
}


memory_type_to_name = {v: k for k, v in name_to_memory_type.items()}


def get_memory_type_enum_json():
    enum_list = []
    for memory_type_name, memory_type_code in name_to_memory_type.items():
        memory_type_str = _C.base.memory_type_to_string(memory_type_code)
        enum_list.append(memory_type_str)
    memory_type_enum = {}
    for single_enum in enum_list:
        memory_type_enum[f"{single_enum}"] = enum_list
    return memory_type_enum


all_type_enum.append(get_memory_type_enum_json)


class MemoryType(_C.base.MemoryType):
    kMemoryTypeNone = _C.base.MemoryType.kMemoryTypeNone
    Allocate = _C.base.MemoryType.Allocate
    External = _C.base.MemoryType.External
    Mapped = _C.base.MemoryType.Mapped
    
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


def get_memory_pool_type_enum_json():
    enum_list = []
    for memory_pool_type_name, memory_pool_type_code in name_to_memory_pool_type.items():
        memory_pool_type_str = _C.base.memory_pool_type_to_string(memory_pool_type_code)
        enum_list.append(memory_pool_type_str)
    memory_pool_type_enum = {}
    for single_enum in enum_list:
        memory_pool_type_enum[f"{single_enum}"] = enum_list
    return memory_pool_type_enum


all_type_enum.append(get_memory_pool_type_enum_json)


class MemoryPoolType(_C.base.MemoryPoolType):
    Embed = _C.base.MemoryPoolType.Embed
    Unity = _C.base.MemoryPoolType.Unity
    ChunkIndepend = _C.base.MemoryPoolType.ChunkIndepend
    
    @classmethod
    def from_name(cls, memory_pool_type_name: str):
        if memory_pool_type_name not in name_to_memory_pool_type:
            raise ValueError(f"Unsupported memory pool type name: {memory_pool_type_name}")
        return cls(name_to_memory_pool_type[memory_pool_type_name])
    

name_to_tensor_type = {
    "Default": _C.base.TensorType.Default,
    "Pipeline": _C.base.TensorType.Pipeline,
}


tensor_type_to_name = {v: k for k, v in name_to_tensor_type.items()}


def get_tensor_type_enum_json():
    enum_list = []
    for tensor_type_name, tensor_type_code in name_to_tensor_type.items():
        tensor_type_str = _C.base.tensor_type_to_string(tensor_type_code)
        enum_list.append(tensor_type_str)
    tensor_type_enum = {}
    for single_enum in enum_list:
        tensor_type_enum[f"{single_enum}"] = enum_list
    return tensor_type_enum


all_type_enum.append(get_tensor_type_enum_json)


class TensorType(_C.base.TensorType):
    Default = _C.base.TensorType.Default
    Pipeline = _C.base.TensorType.Pipeline

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


def get_forward_op_type_enum_json():
    enum_list = []
    for forward_op_type_name, forward_op_type_code in name_to_forward_op_type.items():
        forward_op_type_str = _C.base.forward_op_type_to_string(forward_op_type_code)
        enum_list.append(forward_op_type_str)
    forward_op_type_enum = {}
    for single_enum in enum_list:
        forward_op_type_enum[f"{single_enum}"] = enum_list
    return forward_op_type_enum


all_type_enum.append(get_forward_op_type_enum_json)


class ForwardOpType(_C.base.ForwardOpType):
    Default = _C.base.ForwardOpType.Default
    OneDnn = _C.base.ForwardOpType.OneDnn
    XnnPack = _C.base.ForwardOpType.XnnPack
    QnnPack = _C.base.ForwardOpType.QnnPack
    Cudnn = _C.base.ForwardOpType.Cudnn
    AclOp = _C.base.ForwardOpType.AclOp
    NotSupport = _C.base.ForwardOpType.NotSupport
    
    @classmethod
    def from_name(cls, forward_op_type_name: str):
        if forward_op_type_name not in name_to_forward_op_type:
            raise ValueError(f"Unsupported forward op type name: {forward_op_type_name}")
        return cls(name_to_forward_op_type[forward_op_type_name])
    

name_to_inference_opt_level = {
    "Level0": _C.base.InferenceOpt.Level0,
    "Level1": _C.base.InferenceOpt.Level1,
    "Auto": _C.base.InferenceOpt.LevelAuto,
}


inference_opt_level_to_name = {v: k for k, v in name_to_inference_opt_level.items()}


def get_inference_opt_level_enum_json():
    enum_list = []
    for inference_opt_level_name, inference_opt_level_code in name_to_inference_opt_level.items():
        inference_opt_level_str = _C.base.inference_opt_level_to_string(inference_opt_level_code)
        enum_list.append(inference_opt_level_str)
    inference_opt_level_enum = {}
    for single_enum in enum_list:
        inference_opt_level_enum[f"{single_enum}"] = enum_list
    return inference_opt_level_enum


all_type_enum.append(get_inference_opt_level_enum_json)


class InferenceOptLevel(_C.base.InferenceOpt):
    Level0 = _C.base.InferenceOpt.Level0
    Level1 = _C.base.InferenceOpt.Level1
    LevelAuto = _C.base.InferenceOpt.LevelAuto
    
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
    "GGUF": _C.base.ModelType.GGUF,
    "NotSupport": _C.base.ModelType.NotSupport,
}


model_type_to_name = {v: k for k, v in name_to_model_type.items()}


def get_model_type_enum_json():
    enum_list = []
    for model_type_name, model_type_code in name_to_model_type.items():
        model_type_str = _C.base.model_type_to_string(model_type_code)
        enum_list.append(model_type_str)
    model_type_enum = {}
    for single_enum in enum_list:
        model_type_enum[f"{single_enum}"] = enum_list
    return model_type_enum


all_type_enum.append(get_model_type_enum_json)


class ModelType(_C.base.ModelType):
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
    GGUF = _C.base.ModelType.GGUF
    NotSupport = _C.base.ModelType.NotSupport
    
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
    "Vllm": _C.base.InferenceType.Vllm,
    "SGLang": _C.base.InferenceType.SGLang,
    "Lmdeploy": _C.base.InferenceType.Lmdeploy,
    "LlamaCpp": _C.base.InferenceType.LlamaCpp,
    "LLM": _C.base.InferenceType.LLM,
    "XDit": _C.base.InferenceType.XDit,
    "OneDiff": _C.base.InferenceType.OneDiff,
    "Diffusers": _C.base.InferenceType.Diffusers,
    "Diff": _C.base.InferenceType.Diff,
    "NotSupport": _C.base.InferenceType.NotSupport,
}


inference_type_to_name = {v: k for k, v in name_to_inference_type.items()}


def get_inference_type_enum_json():
    enum_list = []
    for inference_type_name, inference_type_code in name_to_inference_type.items():
        inference_type_str = _C.base.inference_type_to_string(inference_type_code)
        enum_list.append(inference_type_str)
    inference_type_enum = {}
    for single_enum in enum_list:
        inference_type_enum[f"{single_enum}"] = enum_list
    return inference_type_enum


all_type_enum.append(get_inference_type_enum_json)


class InferenceType(_C.base.InferenceType):
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
    Vllm = _C.base.InferenceType.Vllm
    SGLang = _C.base.InferenceType.SGLang
    Lmdeploy = _C.base.InferenceType.Lmdeploy
    LlamaCpp = _C.base.InferenceType.LlamaCpp
    LLM = _C.base.InferenceType.LLM
    XDit = _C.base.InferenceType.XDit
    OneDiff = _C.base.InferenceType.OneDiff
    Diffusers = _C.base.InferenceType.Diffusers
    Diff = _C.base.InferenceType.Diff
    NotSupport = _C.base.InferenceType.NotSupport
    
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


def get_encrypt_type_enum_json():
    enum_list = []
    for encrypt_type_name, encrypt_type_code in name_to_encrypt_type.items():
        encrypt_type_str = _C.base.encrypt_type_to_string(encrypt_type_code)
        enum_list.append(encrypt_type_str)
    encrypt_type_enum = {}
    for single_enum in enum_list:
        encrypt_type_enum[f"{single_enum}"] = enum_list
    return encrypt_type_enum


all_type_enum.append(get_encrypt_type_enum_json)


class EncryptType(_C.base.EncryptType):
    kEncryptTypeNone = _C.base.EncryptType.kEncryptTypeNone
    Base64 = _C.base.EncryptType.Base64
    
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


def get_codec_type_enum_json():
    enum_list = []
    for codec_type_name, codec_type_code in name_to_codec_type.items():
        codec_type_str = _C.base.codec_type_to_string(codec_type_code)
        enum_list.append(codec_type_str)
    codec_type_enum = {}
    for single_enum in enum_list:
        codec_type_enum[f"{single_enum}"] = enum_list
    return codec_type_enum


all_type_enum.append(get_codec_type_enum_json)


class CodecType(_C.base.CodecType):
    kCodecTypeNone = _C.base.CodecType.kCodecTypeNone
    OpenCV = _C.base.CodecType.OpenCV
    FFmpeg = _C.base.CodecType.FFmpeg
    Stb = _C.base.CodecType.Stb
    
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


def get_codec_flag_enum_json():
    enum_list = []
    for codec_flag_name, codec_flag_code in name_to_codec_flag.items():
        codec_flag_str = _C.base.codec_flag_to_string(codec_flag_code)
        enum_list.append(codec_flag_str)
    codec_flag_enum = {}
    for single_enum in enum_list:
        codec_flag_enum[f"{single_enum}"] = enum_list
    return codec_flag_enum


all_type_enum.append(get_codec_flag_enum_json)


class CodecFlag(_C.base.CodecFlag):
    Image = _C.base.CodecFlag.Image
    Images = _C.base.CodecFlag.Images
    Video = _C.base.CodecFlag.Video  
    Camera = _C.base.CodecFlag.Camera
    Other = _C.base.CodecFlag.Other
    
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


def get_parallel_type_enum_json():
    enum_list = []
    for parallel_type_name, parallel_type_code in name_to_parallel_type.items():
        parallel_type_str = _C.base.parallel_type_to_string(parallel_type_code)
        enum_list.append(parallel_type_str)
    parallel_type_enum = {}
    for single_enum in enum_list:
        parallel_type_enum[f"{single_enum}"] = enum_list
    return parallel_type_enum


all_type_enum.append(get_parallel_type_enum_json)


class ParallelType(_C.base.ParallelType):
    kParallelTypeNone = _C.base.ParallelType.kParallelTypeNone
    Sequential = _C.base.ParallelType.Sequential
    Task = _C.base.ParallelType.Task
    Pipeline = _C.base.ParallelType.Pipeline
    
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


def get_edge_type_enum_json():
    enum_list = []
    for edge_type_name, edge_type_code in name_to_edge_type.items():
        edge_type_str = _C.base.edge_type_to_string(edge_type_code)
        enum_list.append(edge_type_str)
    edge_type_enum = {}
    for single_enum in enum_list:
        edge_type_enum[f"{single_enum}"] = enum_list
    return edge_type_enum


all_type_enum.append(get_edge_type_enum_json)


class EdgeType(_C.base.EdgeType):
    Fixed = _C.base.EdgeType.Fixed
    Pipeline = _C.base.EdgeType.Pipeline
    
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


def get_edge_update_flag_enum_json():
    enum_list = []
    for edge_update_flag_name, edge_update_flag_code in name_to_edge_update_flag.items():
        edge_update_flag_str = _C.base.edge_update_flag_to_string(edge_update_flag_code)
        enum_list.append(edge_update_flag_str)
    edge_update_flag_enum = {}
    for single_enum in enum_list:
        edge_update_flag_enum[f"{single_enum}"] = enum_list
    return edge_update_flag_enum


all_type_enum.append(get_edge_update_flag_enum_json)


class EdgeUpdateFlag(_C.base.EdgeUpdateFlag):
    Complete = _C.base.EdgeUpdateFlag.Complete
    Terminate = _C.base.EdgeUpdateFlag.Terminate
    Error = _C.base.EdgeUpdateFlag.Error
    
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


def get_node_color_type_enum_json():
    enum_list = []
    for node_color_type_name, node_color_type_code in name_to_node_color_type.items():
        node_color_type_str = _C.base.node_color_type_to_string(node_color_type_code)
        enum_list.append(node_color_type_str)
    node_color_type_enum = {}
    for single_enum in enum_list:
        node_color_type_enum[f"{single_enum}"] = enum_list
    return node_color_type_enum


all_type_enum.append(get_node_color_type_enum_json)


class NodeColorType(_C.base.NodeColorType):
    White = _C.base.NodeColorType.White
    Gray = _C.base.NodeColorType.Gray
    Black = _C.base.NodeColorType.Black

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


def get_topo_sort_type_enum_json():
    enum_list = []
    for topo_sort_type_name, topo_sort_type_code in name_to_topo_sort_type.items():
        topo_sort_type_str = _C.base.topo_sort_type_to_string(topo_sort_type_code)
        enum_list.append(topo_sort_type_str)
    topo_sort_type_enum = {}
    for single_enum in enum_list:
        topo_sort_type_enum[f"{single_enum}"] = enum_list
    return topo_sort_type_enum


all_type_enum.append(get_topo_sort_type_enum_json)


class TopoSortType(_C.base.TopoSortType):
    BFS = _C.base.TopoSortType.BFS
    DFS = _C.base.TopoSortType.DFS
    
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
    "ErrorInferenceVllm": _C.base.StatusCode.ErrorInferenceVllm,
    "ErrorInferenceSGLang": _C.base.StatusCode.ErrorInferenceSGLang,
    "ErrorInferenceLmdeploy": _C.base.StatusCode.ErrorInferenceLmdeploy,
    "ErrorInferenceLlamaCpp": _C.base.StatusCode.ErrorInferenceLlamaCpp,
    "ErrorInferenceXDit": _C.base.StatusCode.ErrorInferenceXDit,
    "ErrorInferenceOneDiff": _C.base.StatusCode.ErrorInferenceOneDiff,
    "ErrorInferenceDiffusers": _C.base.StatusCode.ErrorInferenceDiffusers,
    "ErrorInferenceDiff": _C.base.StatusCode.ErrorInferenceDiff,
    "ErrorInferenceOther": _C.base.StatusCode.ErrorInferenceOther,
    "ErrorDag": _C.base.StatusCode.ErrorDag,
}


status_code_to_name = {v: k for k, v in name_to_status_code.items()}


# def get_status_code_enum_json():
#     status_code_enum = {
#         "StatusCode": []
#     }
#     for status_code_name, status_code_code in name_to_status_code.items():
#         status_code_str = _C.base.status_code_to_string(status_code_code)
#         status_code_enum["StatusCode"].append(status_code_str)
#     return status_code_enum


# all_type_enum.append(get_status_code_enum_json)


class StatusCode(_C.base.StatusCode):
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
    ErrorInferenceVllm = _C.base.StatusCode.ErrorInferenceVllm
    ErrorInferenceSGLang = _C.base.StatusCode.ErrorInferenceSGLang
    ErrorInferenceLmdeploy = _C.base.StatusCode.ErrorInferenceLmdeploy
    ErrorInferenceLlamaCpp = _C.base.StatusCode.ErrorInferenceLlamaCpp
    ErrorInferenceLLM = _C.base.StatusCode.ErrorInferenceLLM
    ErrorInferenceXDit = _C.base.StatusCode.ErrorInferenceXDit
    ErrorInferenceOneDiff = _C.base.StatusCode.ErrorInferenceOneDiff
    ErrorInferenceDiffusers = _C.base.StatusCode.ErrorInferenceDiffusers
    ErrorInferenceDiff = _C.base.StatusCode.ErrorInferenceDiff
    ErrorInferenceOther = _C.base.StatusCode.ErrorInferenceOther
    ErrorDag = _C.base.StatusCode.ErrorDag
    
    @classmethod
    def from_name(cls, status_code_name: str):
        if status_code_name not in name_to_status_code:
            return cls(StatusCode.ErrorUnknown)
        else:
            return cls(name_to_status_code[status_code_name])
    
class Status(_C.base.Status):
    def __init__(self, status_code: Union[str, StatusCode]):
        if isinstance(status_code, str):
            if status_code in name_to_status_code:
                status_code = StatusCode.from_name(status_code)
            else:
                status_code = StatusCode.ErrorUnknown
        super().__init__(status_code)
    
    def get_code(self):
        return super().get_code()
    
    def get_code_name(self):
        return status_code_to_name[self.get_code()]
    
    @staticmethod
    def ok():
        return Status(StatusCode.Ok)
    
    @staticmethod
    def error():
        return Status(StatusCode.ErrorUnknown)
        
    @staticmethod
    def error_out_of_memory():
        return Status(StatusCode.ErrorOutOfMemory)
        
    @staticmethod
    def error_not_support():
        return Status(StatusCode.ErrorNotSupport)
        
    @staticmethod
    def error_not_implement():
        return Status(StatusCode.ErrorNotImplement)
        
    @staticmethod
    def error_invalid_value():
        return Status(StatusCode.ErrorInvalidValue)
        
    @staticmethod
    def error_invalid_param():
        return Status(StatusCode.ErrorInvalidParam)
        
    @staticmethod
    def error_null_param():
        return Status(StatusCode.ErrorNullParam)
        
    @staticmethod
    def error_thread_pool():
        return Status(StatusCode.ErrorThreadPool)
        
    @staticmethod
    def error_io():
        return Status(StatusCode.ErrorIO)
        
    @staticmethod
    def error_device_cpu():
        return Status(StatusCode.ErrorDeviceCpu)
        
    @staticmethod
    def error_device_arm():
        return Status(StatusCode.ErrorDeviceArm)
        
    @staticmethod
    def error_device_x86():
        return Status(StatusCode.ErrorDeviceX86)
        
    @staticmethod
    def error_device_riscv():
        return Status(StatusCode.ErrorDeviceRiscV)
        
    @staticmethod
    def error_device_cuda():
        return Status(StatusCode.ErrorDeviceCuda)
        
    @staticmethod
    def error_device_rocm():
        return Status(StatusCode.ErrorDeviceRocm)
        
    @staticmethod
    def error_device_sycl():
        return Status(StatusCode.ErrorDeviceSyCL)
        
    @staticmethod
    def error_device_opencl():
        return Status(StatusCode.ErrorDeviceOpenCL)
        
    @staticmethod
    def error_device_opengl():
        return Status(StatusCode.ErrorDeviceOpenGL)
        
    @staticmethod
    def error_device_metal():
        return Status(StatusCode.ErrorDeviceMetal)
        
    @staticmethod
    def error_device_vulkan():
        return Status(StatusCode.ErrorDeviceVulkan)
        
    @staticmethod
    def error_device_hexagon():
        return Status(StatusCode.ErrorDeviceHexagon)
        
    @staticmethod
    def error_device_mtk_vpu():
        return Status(StatusCode.ErrorDeviceMtkVpu)
        
    @staticmethod
    def error_device_ascend_cl():
        return Status(StatusCode.ErrorDeviceAscendCL)
        
    @staticmethod
    def error_device_apple_npu():
        return Status(StatusCode.ErrorDeviceAppleNpu)
        
    @staticmethod
    def error_device_rk_npu():
        return Status(StatusCode.ErrorDeviceRkNpu)
        
    @staticmethod
    def error_device_qualcomm_npu():
        return Status(StatusCode.ErrorDeviceQualcommNpu)
        
    @staticmethod
    def error_device_mtk_npu():
        return Status(StatusCode.ErrorDeviceMtkNpu)
        
    @staticmethod
    def error_device_sophon_npu():
        return Status(StatusCode.ErrorDeviceSophonNpu)
        
    @staticmethod
    def error_op_ascend_cl():
        return Status(StatusCode.ErrorOpAscendCL)
        
    @staticmethod
    def error_inference_default():
        return Status(StatusCode.ErrorInferenceDefault)
        
    @staticmethod
    def error_inference_openvino():
        return Status(StatusCode.ErrorInferenceOpenVino)
        
    @staticmethod
    def error_inference_tensorrt():
        return Status(StatusCode.ErrorInferenceTensorRt)
        
    @staticmethod
    def error_inference_coreml():
        return Status(StatusCode.ErrorInferenceCoreML)
        
    @staticmethod
    def error_inference_tflite():
        return Status(StatusCode.ErrorInferenceTfLite)
        
    @staticmethod
    def error_inference_onnxruntime():
        return Status(StatusCode.ErrorInferenceOnnxRuntime)
        
    @staticmethod
    def error_inference_ascend_cl():
        return Status(StatusCode.ErrorInferenceAscendCL)
        
    @staticmethod
    def error_inference_ncnn():
        return Status(StatusCode.ErrorInferenceNcnn)
        
    @staticmethod
    def error_inference_tnn():
        return Status(StatusCode.ErrorInferenceTnn)
        
    @staticmethod
    def error_inference_mnn():
        return Status(StatusCode.ErrorInferenceMnn)
        
    @staticmethod
    def error_inference_paddlelite():
        return Status(StatusCode.ErrorInferencePaddleLite)
        
    @staticmethod
    def error_inference_rknn():
        return Status(StatusCode.ErrorInferenceRknn)
        
    @staticmethod
    def error_inference_tvm():
        return Status(StatusCode.ErrorInferenceTvm)
        
    @staticmethod
    def error_inference_aitemplate():
        return Status(StatusCode.ErrorInferenceAITemplate)
        
    @staticmethod
    def error_inference_snpe():
        return Status(StatusCode.ErrorInferenceSnpe)
        
    @staticmethod
    def error_inference_qnn():
        return Status(StatusCode.ErrorInferenceQnn)
        
    @staticmethod
    def error_inference_sophon():
        return Status(StatusCode.ErrorInferenceSophon)
        
    @staticmethod
    def error_inference_torch():
        return Status(StatusCode.ErrorInferenceTorch)
        
    @staticmethod
    def error_inference_tensorflow():
        return Status(StatusCode.ErrorInferenceTensorFlow)
        
    @staticmethod
    def error_inference_neuropilot():
        return Status(StatusCode.ErrorInferenceNeuroPilot)
        
    @staticmethod
    def error_inference_vllm():
        return Status(StatusCode.ErrorInferenceVllm)
        
    @staticmethod
    def error_inference_sglang():
        return Status(StatusCode.ErrorInferenceSGLang)
        
    @staticmethod
    def error_inference_lmdeploy():
        return Status(StatusCode.ErrorInferenceLmdeploy)
        
    @staticmethod
    def error_inference_llamacpp():
        return Status(StatusCode.ErrorInferenceLlamaCpp)
        
    @staticmethod
    def error_inference_llm():
        return Status(StatusCode.ErrorInferenceLLM)
        
    @staticmethod
    def error_inference_xdit():
        return Status(StatusCode.ErrorInferenceXDit)
        
    @staticmethod
    def error_inference_onediff():
        return Status(StatusCode.ErrorInferenceOneDiff)
        
    @staticmethod
    def error_inference_diffusers():
        return Status(StatusCode.ErrorInferenceDiffusers)
        
    @staticmethod
    def error_inference_diff():
        return Status(StatusCode.ErrorInferenceDiff)
        
    @staticmethod
    def error_inference_other():
        return Status(StatusCode.ErrorInferenceOther)
        
    @staticmethod
    def error_dag():
        return Status(StatusCode.ErrorDag)
    

name_to_pixel_type = {
    "GRAY": _C.base.PixelType.GRAY,
    "RGB": _C.base.PixelType.RGB,
    "BGR": _C.base.PixelType.BGR,
    "RGBA": _C.base.PixelType.RGBA,
    "BGRA": _C.base.PixelType.BGRA,
    "NotSupport": _C.base.PixelType.NotSupport,
}

pixel_type_to_name = {v: k for k, v in name_to_pixel_type.items()}

def get_pixel_type_enum_json():
    enum_list = []
    for pixel_type_name, pixel_type_code in name_to_pixel_type.items():
        pixel_type_str = _C.base.pixel_type_to_string(pixel_type_code)
        enum_list.append(pixel_type_str)
    pixel_type_enum = {}
    for single_enum in enum_list:
        pixel_type_enum[f"{single_enum}"] = enum_list
    return pixel_type_enum


all_type_enum.append(get_pixel_type_enum_json)


class PixelType(_C.base.PixelType):
    GRAY = _C.base.PixelType.GRAY
    RGB = _C.base.PixelType.RGB
    BGR = _C.base.PixelType.BGR
    RGBA = _C.base.PixelType.RGBA
    BGRA = _C.base.PixelType.BGRA
    NotSupport = _C.base.PixelType.NotSupport
    
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

def get_cvt_color_type_enum_json():
    enum_list = []
    for cvt_color_type_name, cvt_color_type_code in name_to_cvt_color_type.items():
        cvt_color_type_str = _C.base.cvt_color_type_to_string(cvt_color_type_code)
        enum_list.append(cvt_color_type_str)
    cvt_color_type_enum = {}
    for single_enum in enum_list:
        cvt_color_type_enum[f"{single_enum}"] = enum_list
    return cvt_color_type_enum


all_type_enum.append(get_cvt_color_type_enum_json)


class CvtColorType(_C.base.CvtColorType):
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

def get_interp_type_enum_json():
    enum_list = []
    for interp_type_name, interp_type_code in name_to_interp_type.items():
        interp_type_str = _C.base.interp_type_to_string(interp_type_code)
        enum_list.append(interp_type_str)
    interp_type_enum = {}
    for single_enum in enum_list:
        interp_type_enum[f"{single_enum}"] = enum_list
    return interp_type_enum

all_type_enum.append(get_interp_type_enum_json)

class InterpType(_C.base.InterpType):
    Nearst = _C.base.InterpType.Nearst
    Linear = _C.base.InterpType.Linear
    Cubic = _C.base.InterpType.Cubic
    Arer = _C.base.InterpType.Arer
    NotSupport = _C.base.InterpType.NotSupport
    
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


def get_border_type_enum_json():    
    enum_list = []
    for border_type_name, border_type_code in name_to_border_type.items():
        border_type_str = _C.base.border_type_to_string(border_type_code)
        enum_list.append(border_type_str)
    border_type_enum = {}
    for single_enum in enum_list:
        border_type_enum[f"{single_enum}"] = enum_list
    return border_type_enum

all_type_enum.append(get_border_type_enum_json)


class BorderType(_C.base.BorderType):
    Constant = _C.base.BorderType.Constant
    Reflect = _C.base.BorderType.Reflect
    Edge = _C.base.BorderType.Edge
    NotSupport = _C.base.BorderType.NotSupport
    
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

    def get_cost_time(self, key: str):
        return self._profiler.get_cost_time(key)
    
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

def time_profiler_get_cost_time(key: str):
    return _C.base.time_profiler_get_cost_time(key)

def time_profiler_print(title: str = ""):
    _C.base.time_profiler_print(title)

def time_profiler_print_index(title: str, index: int):
    _C.base.time_profiler_print_index(title, index)

def time_profiler_print_remove_warmup(title: str, warmup_times: int):
    _C.base.time_profiler_print_remove_warmup(title, warmup_times)


class Param(_C.base.Param):
    def __init__(self):
        super().__init__()

    def set(self, key: str, value: any):
        return super().set(key, value)

    def get(self, key: str):
        return super().get(key)
    
    def set_required_params(self, params):
        return super().set_required_params(params)
    
    def add_required_param(self, param):
        return super().add_required_param(param)
    
    def remove_required_param(self, param):
        return super().remove_required_param(param)
    
    def clear_required_params(self):
        return super().clear_required_params()
    
    def get_required_params(self):
        return super().get_required_params()
    
    def serialize(self) -> str:
        return super().serialize()

    def save_file(self, path: str):
        return super().save_file(path)
        
    def deserialize(self, value):
        return super().deserialize(value)
    
    def load_file(self, path: str):
        return super().load_file(path)
  
def remove_json_brackets(json_str: str):
    return _C.base.remove_json_brackets(json_str) 
    
def pretty_json_str(json_str: str):
    return _C.base.pretty_json_str(json_str) 


def load_library_from_path(path: str, update: bool = False):
    return _C.base.load_library_from_path(path, update)

def free_library(path: str):
    return _C.base.free_library(path)

def get_library_handle(path: str, update: bool = False):
    return _C.base.get_library_handle(path, update)

