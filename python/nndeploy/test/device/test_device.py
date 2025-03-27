
from functools import singledispatch
import nndeploy._nndeploy_internal as _C

from enum import Enum
import numpy as np
import nndeploy.base
from nndeploy.device import BufferDesc, TensorDesc, Architecture, Device, Stream, Event, MemoryPool, Buffer, Tensor
import time

## cd ../../../
## pip install -e .
## cd nndeploy/test/device python3 test_device.py
## python3 nndeploy/test/device/test_device.py


def test_device():
    architecture = Architecture(nndeploy.base.DeviceTypeCode.cpu)
    print(architecture.get_device(0))
    architecture = Architecture("ascendcl")
    print(architecture.get_device(0))
    print(architecture.get_device(1))
    print(architecture.get_device(2))
    device = Device("ascendcl:3")
    print(device)
    stream = Stream(device)
    print(stream)
    event = Event(device)
    print(event)


def test_desc():
    # 测试BufferDesc类
    print("\nBufferDesc测试:")
    # 测试不同构造方式
    buf1 = BufferDesc()
    buf2 = BufferDesc(1024)
    buf3 = BufferDesc([1, 224, 224, 3])  # 使用size_vector构造
    
    # 测试比较操作
    print("buf1 == buf2?", buf1 == buf2)
    print("buf1 != buf3?", buf1 != buf3)
    
    # 测试方法调用
    print("buf2 size:", buf2.get_size())
    buf2.just_modify(2048)
    print("修改后buf2 size:", buf2.get_size())
    
    # 测试TensorDesc类
    print("\nTensorDesc测试:")
    # 创建不同格式的TensorDesc
    desc1 = TensorDesc()
    desc2 = TensorDesc(nndeploy.base.DataType("float32"), 
                      nndeploy.base.DataFormat.NCHW,
                      [1, 3, 224, 224])
    desc3 = TensorDesc(nndeploy.base.DataType(np.int8),
                      nndeploy.base.DataFormat.NHWC,
                      [1, 224, 224, 3],
                      [224*224*3, 224*3, 3, 1])
    
    # 测试属性访问和修改
    desc2.data_format = nndeploy.base.DataFormat.NHWC
    desc3.shape = [2, 128, 128, 3]
    
    # 测试比较操作
    print("desc1 == desc2?", desc1 == desc2)
    print("desc2 != desc3?", desc2 != desc3)
    
    # 测试打印输出
    print("\nTensorDesc输出测试:")
    print("desc1:", desc1)
    print("desc2:", desc2)
    print("desc3:", desc3)
    

def test_memory_pool():
    device = Device("cpu")
    memory_pool = MemoryPool(device, nndeploy.base.MemoryPoolType.Embed)
    print(memory_pool.get_device())
    print(memory_pool.get_memory_pool_type())


def test_buffer():
    buffer = Buffer(Device("cpu"), 8)
    print(buffer)
    import numpy as np
    numpy_array = np.asarray(buffer)
    print(numpy_array)
    numpy_array = buffer.to_numpy(np.dtype(np.float32))
    print(numpy_array)
    numpy_array = buffer.to_numpy(np.float32)
    print(numpy_array)

    print(type(numpy_array))
    buffer_from_numpy = Buffer.from_numpy(numpy_array)
    print(numpy_array)
    print(buffer_from_numpy)

# test tensor
def test_tensor():
    tensor = Tensor(Device("cpu"), TensorDesc(nndeploy.base.DataType("float32"), nndeploy.base.DataFormat.NCHW, [1, 3, 224, 224]), "test")
    print(tensor)
    import numpy as np  
    numpy_array = np.asarray(tensor)
    print(numpy_array)
    numpy_array = tensor.to_numpy()
    print(numpy_array)

    print(type(numpy_array))
    tensor_from_numpy = Tensor.from_numpy(numpy_array)
    print(numpy_array)
    print(tensor_from_numpy)
    

def test_tensor_from_numpy():
    x = np.int32(123)
    y = np.int32(4)
    # z = np.int32(79)
    x_numpy = np.array([1,2,3],dtype=np.int32)
    # x_tensor = Tensor.from_numpy(x_numpy)
    x_tensor = Tensor.from_numpy(np.array([1,2,3],dtype=np.int32))
    print("x_tensor:",x_tensor)
    y_numpy = np.array([4,5,6],dtype=np.int32)
    # y_tensor = Tensor.from_numpy(y_numpy)
    y_tensor = Tensor.from_numpy(np.array([4,5,6],dtype=np.int32))
    # z_tensor = Tensor.from_numpy(np.array([z],dtype=np.int32))

    # x_a=x_tensor.to_numpy()
    # y_a=y_tensor.to_numpy()
    # z_a=z_tensor.to_numpy()

    print("x:",x)
    print("x_tensor:",x_tensor)
    # print("x_a:",x_a)
    print("xxxxxxxx")
    print("y:",y)
    print("y_tensor:",y_tensor)
    # print("y_a:",y_a)


if __name__ == "__main__":
    print("test_device start")
    # test_device()
    # test_desc()
    # test_memory_pool()
    # test_buffer()
    # test_tensor()
    test_tensor_from_numpy()
    print("test_device end")
