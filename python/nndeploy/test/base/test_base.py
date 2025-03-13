
import nndeploy._nndeploy_internal as _C

from enum import Enum
import numpy as np

import nndeploy.base
import time


## cd ../../../
## pip install -e .
## cd nndeploy/test/base python3 test_base.py
## python3 nndeploy/test/base/test_base.py


def test_data_type():
    data_type = _C.base.DataType()
    assert data_type.code_ == _C.base.DataTypeCode.Fp
    assert data_type.bits_ == 32
    assert data_type.lanes_ == 1
    print(data_type)

    data_type_1 = _C.base.DataType(_C.base.DataTypeCode.Int, 16, 4)
    assert data_type_1.code_ == _C.base.DataTypeCode.Int
    assert data_type_1.bits_ == 16
    assert data_type_1.lanes_ == 4
    print(data_type_1)

    data_type_2 = _C.base.DataType(data_type_1)
    assert data_type_2 == data_type_1
    print(data_type_2)

    data_type_3 = _C.base.DataType()
    data_type_3 = data_type_1
    assert data_type_3 == data_type_1
    print(data_type_3)

    data_type = nndeploy.base.DataType(nndeploy.base.DataTypeCode.Int)
    print(data_type)
    print(data_type.get_data_type_code())
    print(data_type.get_bits())
    print(data_type.get_lanes())
    print(data_type.get_bytes())
    print(data_type.get_name())

    data_type = nndeploy.base.DataType(nndeploy.base.DataTypeCode.Int, 32)
    print(data_type)
    print(data_type.get_data_type_code())
    print(data_type.get_bits())
    print(data_type.get_lanes())
    print(data_type.get_bytes())
    print(data_type.get_name())

    data_type = nndeploy.base.DataType.from_numpy_dtype(np.int32)
    print(data_type)
    print(data_type.get_data_type_code())
    print(data_type.get_bits())
    print(data_type.get_lanes())
    print(data_type.get_bytes())
    print(data_type.get_name())
    print(data_type.get_numpy_dtype())
    print(type(np.int32))
    data_type = nndeploy.base.DataType(np.int32)
    print(data_type)
    print(data_type.get_data_type_code())
    print(data_type.get_bits())
    print(data_type.get_lanes())
    print(data_type.get_bytes())
    print(data_type.get_name())
    print(data_type.get_numpy_dtype())

    data_type = nndeploy.base.DataType.from_name("float32")
    print(data_type)
    print(data_type.get_data_type_code())
    print(data_type.get_bits())
    print(data_type.get_lanes())
    print(data_type.get_bytes())
    print(data_type.get_name())
    print(data_type.get_numpy_dtype())
    data_type = nndeploy.base.DataType("float32")
    print(data_type)
    print(data_type.get_data_type_code())
    print(data_type.get_bits())
    print(data_type.get_lanes())
    print(data_type.get_bytes())
    print(data_type.get_name())
    print(data_type.get_numpy_dtype())

def test_device_type():
    device_type = _C.base.DeviceType()
    assert device_type.code_ == _C.base.DeviceTypeCode.cpu
    assert device_type.device_id_ == 0
    print(device_type)

    device_type_1 = _C.base.DeviceType(_C.base.DeviceTypeCode.cuda, 1)
    assert device_type_1.code_ == _C.base.DeviceTypeCode.cuda
    assert device_type_1.device_id_ == 1
    print(device_type_1)

    device_type_2 = _C.base.DeviceType(device_type_1)
    assert device_type_2 == device_type_1
    print(device_type_2)

    device_type_3 = _C.base.DeviceType()
    device_type_3 = device_type_1
    assert device_type_3 == device_type_1
    print(device_type_3)

    assert device_type != device_type_1
    assert device_type_1 == _C.base.DeviceTypeCode.cuda

    device_type = nndeploy.base.DeviceType("cpu", 0)
    print(device_type)
    print(device_type.get_device_type_code())
    print(device_type.get_device_id())
    print(device_type.get_device_name())

    device_type = nndeploy.base.DeviceType("cpu:0")
    print(device_type)
    print(device_type.get_device_type_code())
    print(device_type.get_device_id())
    print(device_type.get_device_name())

    device_type = nndeploy.base.DeviceType("cuda:0")
    print(device_type)
    print(device_type.get_device_type_code())
    print(device_type.get_device_id())
    print(device_type.get_device_name())

    print(nndeploy.base.DeviceTypeCode)
    print(type(nndeploy.base.DeviceTypeCode.cuda))
    device_type = nndeploy.base.DeviceType(nndeploy.base.DeviceTypeCode.cuda, 0)
    print(device_type)
    print(device_type.get_device_type_code())
    print(device_type.get_device_id())
    print(device_type.get_device_name())

    device_type = nndeploy.base.DeviceType(nndeploy.base.DeviceTypeCode.cuda)
    print(device_type)
    print(device_type.get_device_type_code())
    print(device_type.get_device_id())
    print(device_type.get_device_name())

    device_type = nndeploy.base.DeviceType()
    print(device_type)
    print(device_type.get_device_type_code())
    print(device_type.get_device_id())
    print(device_type.get_device_name())
    

def test_data_format():
    data_format = nndeploy.base.DataFormat.N
    print(data_format)

    data_format = nndeploy.base.DataFormat.NCHW
    print(data_format)

    data_format = nndeploy.base.DataFormat.Auto
    print(data_format)

    data_format = nndeploy.base.DataFormat.NotSupport
    print(data_format)

    data_format = nndeploy.base.DataFormat.from_name("NCHW")
    print(data_format)


def test_status():
    status = nndeploy.base.Status("Ok")
    print(status)
    print(status.get_code())
    print(status.get_code_name())

    status = nndeploy.base.Status(nndeploy.base.StatusCode.ErrorUnknown)
    print(status)
    print(status.get_code())
    print(status.get_code_name())

    status = nndeploy.base.Status("ErrorInvalidValue")
    print(status)
    print(status.get_code())
    print(status.get_code_name())

    status = nndeploy.base.Status("NotExistStatusCode")
    print(status)
    print(status.get_code())
    print(status.get_code_name())



def test_time_profiler():
    _C.base.time_profiler_reset()
    _C.base.time_point_start("test")
    time.sleep(1)
    _C.base.time_point_end("test")
    _C.base.time_profiler_print("_C.base.time_profiler_reset")

    # nndeploy.base.time_profiler_reset()
    nndeploy.base.time_point_start("test")
    time.sleep(1)
    nndeploy.base.time_point_end("test")
    nndeploy.base.time_profiler_print("nndeploy.base.time_profiler_reset")

    time_profiler = nndeploy.base.TimeProfiler()
    time_profiler.reset()
    time_profiler.start("test")
    time.sleep(1)
    time_profiler.end("test")
    time_profiler.print("nndeploy.base.TimeProfiler")

    print(nndeploy.base.time_profiler_get_cost_time("test"))

class Param(_C.base.Param):
    def __init__(self):
        super().__init__()
        self._value = 0
    
    def set(self, key: str, value: int):
        if key == "value":
            self._value = value
        else:
            raise ValueError(f"Unsupported key: {key}")
        
    
    def get(self, key: str):
        if key == "value":
            return self._value
        else:
            raise ValueError(f"Unsupported key: {key}")
    

def test_param():
    param = _C.base.Param()
    print(param)

    param = Param()
    print(param)

    param.set("value", 1)
    print(param.get("value"))

    param = nndeploy.base.Param()
    param.serialize({"value": 1})
    print(param)
    param.set({"value": 2})
    print(param)

    vaule = {"value.json": {"value": 1}}
    param.deserialize(vaule)
    
    # vaule = "/home/ascenduserdg01/github/nndeploy/build/value.json"
    # param.deserialize(vaule)
    # print(param)


if __name__ == "__main__":
    print("test_base start")
    test_data_type()
    test_device_type()
    test_data_format()
    test_status()
    test_time_profiler()
    test_param()
    print("test_base end")