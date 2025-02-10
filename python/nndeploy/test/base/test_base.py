
import nndeploy._nndeploy_internal as _C

from enum import Enum
import numpy as np
import nndeploy.base.common as common
import time


## cd ../../../
## pip install -e .
## cd nndeploy/test/base python3 test_base.py
## python3 nndeploy/test/base/test_base.py


def test_data_type():
    # 测试默认构造函数
    data_type = _C.base.DataType()
    assert data_type.code_ == _C.base.DataTypeCode.Fp
    assert data_type.bits_ == 32
    assert data_type.lanes_ == 1
    # print(data_type)

    # 测试带参数的构造函数
    data_type = _C.base.DataType(_C.base.DataTypeCode.Uint, 32, 1)
    assert data_type.code_ == _C.base.DataTypeCode.Uint
    assert data_type.bits_ == 32
    assert data_type.lanes_ == 1
    # print(data_type)

    # 测试相等运算符
    data_type_1 = _C.base.DataType(_C.base.DataTypeCode.Uint, 32, 1)
    data_type_2 = _C.base.DataType(_C.base.DataTypeCode.Uint, 32, 1)
    assert data_type_1 == _C.base.DataTypeCode.Uint
    assert data_type_1 != _C.base.DataType(_C.base.DataTypeCode.Int, 32, 1)
    assert data_type_1 != _C.base.DataType(_C.base.DataTypeCode.Fp, 32, 1)
    assert data_type_1 != _C.base.DataType(_C.base.DataTypeCode.BFp, 32, 1)
    assert data_type_1 != _C.base.DataType(_C.base.DataTypeCode.OpaqueHandle, 32, 1)
    assert data_type_1 != _C.base.DataType(_C.base.DataTypeCode.NotSupport, 32, 1)

    # 测试拷贝构造函数
    data_type_3 = _C.base.DataType(data_type_1)
    assert data_type_3 == data_type_1

    # 测试拷贝赋值运算符
    data_type_4 = _C.base.DataType()
    data_type_4 = data_type_1
    assert data_type_4 == data_type_1

    # 测试打印
    print(_C.base.DataType)
    print(data_type_4)

    data_type = common.DataType.from_numpy_dtype(np.float32)
    print(data_type)

    data_type = common.DataType(common.DataTypeCode.Fp, 16, 1)
    print(data_type)
    print(data_type.get_numpy_dtype())
    print(data_type.get_data_type_code())
    print(data_type.get_bits())
    print(data_type.get_lanes())
    print(data_type.get_bytes())
    print(data_type.get_name())


def test_device_type():
    device_type = _C.base.DeviceType()
    assert device_type.code_ == _C.base.DeviceTypeCode.cpu
    assert device_type.device_id_ == 0
    print(device_type)

    device_type_1 = _C.base.DeviceType(_C.base.DeviceTypeCode.arm, 1)
    assert device_type_1.code_ == _C.base.DeviceTypeCode.arm
    assert device_type_1.device_id_ == 1
    print(device_type_1)

    device_type_2 = _C.base.DeviceType(device_type_1)
    assert device_type_2 == device_type_1
    print(device_type_2)

    device_type_3 = _C.base.DeviceType()
    device_type_3 = device_type_1
    assert device_type_3 == device_type_1
    print(device_type_3)

    device_type = common.DeviceType("ascendcl", 1)
    print(device_type)
    print(device_type.get_device_type_code())
    print(device_type.get_device_id())
    print(device_type.get_device_name())

    device_type = common.DeviceType.from_device_type(device_type_3)
    print(device_type)
    print(device_type.get_device_type_code())
    print(device_type.get_device_id())
    print(device_type.get_device_name())

    device_type = common.DeviceType.from_device_type_code(_C.base.DeviceTypeCode.ascendcl, 1)
    print(device_type)
    print(device_type.get_device_type_code())
    print(device_type.get_device_id())
    print(device_type.get_device_name())

    device_type = common.DeviceType.from_device_type_code(common.DeviceTypeCode.ascendcl, 1)
    print(device_type)
    print(device_type.get_device_type_code())
    print(device_type.get_device_id())
    print(device_type.get_device_name())


def test_data_format():
    data_format = common.DataFormat.from_data_format(_C.base.DataFormat.N)
    print(data_format)

    data_format = common.DataFormat.NC
    print(data_format)


    data_format = common.DataFormat.from_name("NCHW")
    print(data_format)


def test_status():
    c_status_code = _C.base.StatusCode.ErrorUnknown
    # print(_C.base.StatusCode) #<class 'nndeploy._nndeploy_internal.base.StatusCode'>
    # print(c_status_code) # StatusCode.ErrorUnknown
    # print(c_status_code.value) # 1
    # print(c_status_code.name) # ErrorUnknown

    py_status_code = common.StatusCode.ErrorUnknown #<enum 'StatusCode'>
    # print(common.StatusCode) # <enum 'StatusCode'>
    # print(py_status_code) # StatusCode.ErrorUnknown 
    # print(py_status_code.value) # StatusCode.ErrorUnknown
    # print(py_status_code.name) # ErrorUnknown

    # assert c_status_code != py_status_code
    # assert c_status_code == py_status_code.value


    c_status = _C.base.Status(c_status_code)
    print(c_status)
    print(type(c_status))
    print(c_status.get_code())
    print(type(c_status.get_code()))
    print(c_status.get_desc())

    c_status_1 = c_status
    print(c_status_1)

    print(c_status is c_status_1)
    print(c_status == c_status_1)

    c_status_2 = _C.base.Status(c_status)
    print(c_status_2)

    print(c_status == c_status_code)
    print(c_status_1 == c_status_2)
    print(c_status_2 == 1)

    py_status = common.Status(py_status_code)
    print(c_status == py_status)
    print(c_status == py_status.get_code())

    py_status_1 = common.Status.from_status(c_status)
    print(py_status_1 == py_status)

    py_status_2 = common.Status.from_status_code(_C.base.StatusCode.ErrorUnknown)
    print(py_status_2 == py_status)
    
    py_status_3 = common.Status.from_name("ErrorUnknown")
    print(py_status_3 == py_status)



def test_time_profiler():
    _C.base.time_profiler_reset()
    _C.base.time_point_start("test")
    time.sleep(1)
    _C.base.time_point_end("test")
    _C.base.time_profiler_print("_C.base.time_profiler_reset")

    # common.time_profiler_reset()
    common.time_point_start("test")
    time.sleep(1)
    common.time_point_end("test")
    common.time_profiler_print("common.time_profiler_reset")

    time_profiler = common.TimeProfiler()
    time_profiler.reset()
    time_profiler.start("test")
    time.sleep(1)
    time_profiler.end("test")
    time_profiler.print("common.TimeProfiler")

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

    param = common.Param()
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