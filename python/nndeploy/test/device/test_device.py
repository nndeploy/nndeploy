
from functools import singledispatch
import nndeploy._nndeploy_internal as _C

from enum import Enum
import numpy as np
import nndeploy.base.common as common
import time


## cd ../../../
## pip install -e .
## cd nndeploy/test/device python3 test_device.py
## python3 nndeploy/test/device/test_device.py

# 继承导致函数怪异
class VulkanArchitecture(_C.device.Architecture):
    def __init__(self, device_type_code):
        # super().__init__(common.DeviceTypeCode.vulkan.value)
        super().__init__(device_type_code.value)          
        
    def checkDevice(self, device_id=0, library_path=""):
        print("checkDevice")
        return common.Status(common.StatusCode.Ok)

    def enableDevice(self, device_id=0, library_path=""):
        print("enableDevice")
        return common.Status(common.StatusCode.Ok)
    
    def getDevice(self, device_id):
        return VulkanDevice()
    
    def getDeviceInfo(self, library_path=""):
        print("getDeviceInfo")
        return []

# _C.device.registerArchitecture(common.DeviceTypeCode.vulkan.value, VulkanArchitecture())
_C.device.registerArchitecture(common.DeviceTypeCode.NewEnum.value, VulkanArchitecture(common.DeviceTypeCode.NewEnum.value))

class VulkanDevice(_C.device.Device):
    def __init__(self):
        super().__init__(common.DeviceType.vulkan)

    # 重点函数
    def toBufferDesc(self, desc, config):
        print("toBufferDesc")
        return _C.device.BufferDesc()

    def allocate(self, size):
        print("allocate")
        return None

    def deallocate(self, ptr):
        print("deallocate")

    def copy(self, src, dst, size, stream=None):
        print("copy")
        return common.Status(common.StatusCode.Ok)

    def download(self, src, dst, size, stream=None):
        print("download") 
        return common.Status(common.StatusCode.Ok)

    def upload(self, src, dst, size, stream=None):
        print("upload")
        return common.Status(common.StatusCode.Ok)

    def init(self):
        print("init")
        return common.Status(common.StatusCode.Ok)

    def deinit(self):
        print("deinit") 
        return common.Status(common.StatusCode.Ok)


def test_device():
    architecture = _C.device.getArchitecture(common.DeviceTypeCode.NewEnum.value)
    print(architecture.getDeviceTypeCode())
    architecture = _C.device.getArchitecture(common.DeviceTypeCode.cpu)

    _C.device.printArchitectureMap()
    # print(architecture.getDeviceTypeCode())
    # device = architecture.getDevice(0)
    # stream = device.createStream()
    # print(stream)
    # event = device.createEvent()
    # print(event)
    # print(device)

    # device_type = common.DeviceType("cpu", 0)
    # print(device_type)
    # device = _C.device.getDevice(device_type)
    # print(device)
    # stream = device.createStream()
    # print(stream)
    # event = device.createEvent()
    # print(event)

    device = _C.device.getDevice(common.DeviceType("ascendcl", 0))
    print(device)
    stream = _C.device.createStream(common.DeviceType("ascendcl", 0))
    print(stream)
    event = _C.device.createEvent(common.DeviceType("ascendcl", 0))
    print(event)


def test_buffer_desc():
    buffer_desc = _C.device.BufferDesc(1)
    print(buffer_desc)

    buffer_desc_1 = _C.device.BufferDesc(1024)
    print(buffer_desc_1)

    print(buffer_desc_1 >= buffer_desc)

    buffer_desc_2 = _C.device.BufferDesc([2, 512])
    print(buffer_desc_2)

    buffer_desc_3 = _C.device.BufferDesc([2, 512])
    print(buffer_desc_3)

    print(buffer_desc_3 == buffer_desc_2)

    buffer_desc_3 = _C.device.BufferDesc(1, [1, 2])

    buffer_desc_3.justModify([1, 512])
    print(buffer_desc_3)
    
    print(buffer_desc_3 != buffer_desc_2)

    print(buffer_desc_3.getSize())
    print(buffer_desc_3.getSizeVector())
    print(buffer_desc_3.getRealSize())
    print(buffer_desc_3.getRealSizeVector())
    print(buffer_desc_3.getConfig())
    print(buffer_desc_3.isSameConfig(buffer_desc_2))
    print(buffer_desc_3.isSameDim(buffer_desc_2))
    print(buffer_desc_3.is1D())

    buffer_desc_3.clear()
    print(buffer_desc_3)

    
# test tensor desc
def test_tensor_desc():
    tensor_desc = _C.device.TensorDesc()
    print(tensor_desc)  # 预期输出: <nndeploy._nndeploy_internal.device.TensorDesc object at 0x...> : data_type_: 0, data_format_: 0, shape_: [], stride_: []

    data_type = common.DataType.from_name("float32")
    data_format = common.DataFormat.NCHW
    shape = [1, 3, 224, 224]
    tensor_desc_1 = _C.device.TensorDesc(data_type, data_format.value, shape) 
    print(tensor_desc_1)  # 预期输出: <nndeploy._nndeploy_internal.device.TensorDesc object at 0x...> : data_type_: 0, data_format_: 0, shape_: [1, 3, 224, 224], stride_: []

    data_type = common.DataType.from_name("float32")
    data_format = common.DataFormat.NCHW
    shape = [1, 3, 224, 224]
    stride = [150528, 50176, 224, 1]
    tensor_desc_2 = _C.device.TensorDesc(data_type, data_format.value, shape, stride)
    print(tensor_desc_2)  # 预期输出: <nndeploy._nndeploy_internal.device.TensorDesc object at 0x...> : data_type_: 0, data_format_: 0, shape_: [1, 3, 224, 224], stride_: [150528, 50176, 224, 1]

    tensor_desc_3 = _C.device.TensorDesc(tensor_desc_2)
    print(tensor_desc_3)  # 预期输出: <nndeploy._nndeploy_internal.device.TensorDesc object at 0x...> : data_type_: 0, data_format_: 0, shape_: [1, 3, 224, 224], stride_: [150528, 50176, 224, 1]

    print(tensor_desc_2 == tensor_desc_3)  # 预期输出: True
    print(tensor_desc_1 != tensor_desc_2)  # 预期输出: True

    tensor_desc_2.data_type_ = common.DataType.from_name("float64")
    tensor_desc_2.data_format_ = common.DataFormat.NCHW.value
    tensor_desc_2.shape_ = [1, 224, 224, 3]
    tensor_desc_2.stride_ = [150528, 672, 3, 1]
    print(tensor_desc_2)  # 预期输出: <nndeploy._nndeploy_internal.device.TensorDesc object at 0x...> : data_type_: 1, data_format_: 1, shape_: [1, 224, 224, 3], stride_: [150528, 672, 3, 1]


class TestMemoryPool(_C.device.MemoryPool):
    def __init__(self, device, memory_pool_type):
        super().__init__(device, memory_pool_type)

    from functools import singledispatch

    def init(self, *args):
        if len(args) == 0:
            print("init")
            return common.Status(common.StatusCode.Ok)
        elif len(args) == 1:
            if isinstance(args[0], int):
                size = args[0]
                print(f"init with size {size}")
                return common.Status(common.StatusCode.Ok)
            elif isinstance(args[0], _C.device.Buffer):
                buffer = args[0]
                print(f"init with buffer {buffer}")
                return common.Status(common.StatusCode.Ok)
        elif len(args) == 2:
            ptr, size = args
            print(f"init with ptr {ptr} and size {size}")
            return common.Status(common.StatusCode.Ok)
        else:
            raise ValueError(f"Unsupported arguments: {args}")

    def deinit(self):
        print("deinit")
        return common.Status(common.StatusCode.Ok)

    def allocate(self, size):
        print(f"allocate {size}")
        return 0
    
    def allocate(self, desc):
        print(f"allocate {desc}")
        return 0
    
    def deallocate(self, ptr):
        print(f"deallocate {ptr}")
    
    def allocatePinned(self, size):
        print(f"allocatePinned {size}")
        return 0
    
    def allocatePinned(self, desc):
        print(f"allocatePinned {desc}")
        return 0
    
    def deallocatePinned(self, ptr):
        print(f"deallocatePinned {ptr}")
    

def test_memory_pool():
    device = _C.device.getDevice(common.DeviceType("cpu", 0))
    memory_pool_type = common.MemoryPoolType.ChunkIndepend
    memory_pool = TestMemoryPool(device, memory_pool_type.value)
    print(memory_pool)  # 预期输出: <nndeploy._nndeploy_internal.device.MemoryPool object at 0x...>

    memory_pool.init()
    print(memory_pool.getDevice())  # 预期输出: <nndeploy._nndeploy_internal.device.Device object at 0x...>
    print(memory_pool.getMemoryPoolType())  # 预期输出: 0

    size = 1024
    memory_pool.init(size)
    ptr = memory_pool.allocate(size)
    print(ptr)  # 预期输出: <capsule object NULL at 0x...>
    memory_pool.deallocate(ptr)

    buffer_desc = _C.device.BufferDesc([1, 3, 224, 224])
    ptr = memory_pool.allocate(buffer_desc)
    print(ptr)  # 预期输出: <capsule object NULL at 0x...>
    memory_pool.deallocate(ptr)

    size = 2048
    ptr = memory_pool.allocatePinned(size) 
    print(ptr)  # 预期输出: <capsule object NULL at 0x...>
    memory_pool.deallocatePinned(ptr)

    buffer_desc = _C.device.BufferDesc([1, 3, 224, 224])  
    ptr = memory_pool.allocatePinned(buffer_desc)
    print(ptr)  # 预期输出: <capsule object NULL at 0x...>
    memory_pool.deallocatePinned(ptr)

    memory_pool.deinit()


def test_buffer():
    device = _C.device.getDevice(common.DeviceType("cpu", 0))
    size = 1024
    buffer = _C.device.Buffer(device, size)
    print(buffer)  # 预期输出: <nndeploy._nndeploy_internal.device.Buffer object at 0x...>

    buffer_desc = _C.device.BufferDesc([1, 3, 448, 448])
    buffer = _C.device.Buffer(device, buffer_desc)
    print(buffer)  # 预期输出: <nndeploy._nndeploy_internal.device.Buffer object at 0x...>

    ptr = device.allocate(size)
    buffer = _C.device.Buffer(device, size, ptr)
    print(buffer)  # 预期输出: <nndeploy._nndeploy_internal.device.Buffer object at 0x...>

    ptr = device.allocate(buffer_desc.getRealSize())
    print(ptr)
    buffer = _C.device.Buffer(device, buffer_desc, ptr)
    print(buffer)  # 预期输出: <nndeploy._nndeploy_internal.device.Buffer object at 0x...>

    memory_type = common.MemoryType.External
    buffer = _C.device.Buffer(device, size, ptr, memory_type.value)
    print(buffer)  # 预期输出: <nndeploy._nndeploy_internal.device.Buffer object at 0x...>

    buffer = _C.device.Buffer(device, buffer_desc, ptr, memory_type.value) 
    print(buffer)  # 预期输出: <nndeploy._nndeploy_internal.device.Buffer object at 0x...>

    device.deallocate(ptr)

    buffer2 = _C.device.Buffer(buffer)
    print(buffer2)  # 预期输出: <nndeploy._nndeploy_internal.device.Buffer object at 0x...>

    clone_buffer = buffer.clone()
    print(clone_buffer)  # 预期输出: <nndeploy._nndeploy_internal.device.Buffer object at 0x...>

    buffer.copyTo(clone_buffer)

    # import io
    # stream = io.BytesIO()
    # buffer.serialize(stream)
    # stream.seek(0)
    # buffer.deserialize(stream)

    buffer.print()

    buffer.justModify(2048)
    buffer.justModify([1, 3, 448, 448])
    buffer.justModify(buffer_desc)

    buffer = _C.device.Buffer(clone_buffer)

    print(buffer.empty())  # 预期输出: False
    print(buffer.getDeviceType())  # 预期输出: 0
    print(buffer.getDevice())  # 预期输出: <nndeploy._nndeploy_internal.device.Device object at 0x...>
    print(buffer.getMemoryPool())  # 预期输出: <nndeploy._nndeploy_internal.device.MemoryPool object at 0x...>
    print(buffer.isMemoryPool())  # 预期输出: True
    print(buffer.getDesc())  # 预期输出: <nndeploy._nndeploy_internal.device.BufferDesc object at 0x...>
    print(buffer.getSize())  # 预期输出: 1605632
    print(buffer.getSizeVector())  # 预期输出: [1, 3, 448, 448]
    print(buffer.getRealSize())  # 预期输出: 1605632
    print(buffer.getRealSizeVector())  # 预期输出: [1, 3, 448, 448] 
    print(buffer.getConfig())  # 预期输出: []
    print(buffer.getData())  # 预期输出: <capsule object NULL at 0x...>
    print(buffer.getMemoryType())  # 预期输出: 0

    # buffer.addRef()
    # buffer.subRef()

    numpy_array = np.asarray(buffer)
    print(numpy_array.data) 
    # shape = list(numpy_array.shape)
    # shape[-1] //= 4
    # numpy_array = numpy_array.reshape(shape)
    print(numpy_array.shape) 
    print(numpy_array.strides)
    print(numpy_array.dtype)
    print(numpy_array.data) 

    # 测试 Buffer 的 __array__ 接口
    numpy_array = np.array(buffer)
    print(numpy_array.shape)  # 预期输出: [1, 3, 448, 448]
    print(numpy_array.dtype)  # 预期输出: uint8

    # 测试 Buffer 的 to_numpy 接口
    numpy_array = buffer.to_numpy(np.dtype(np.float32))
    print(numpy_array.shape)  # 预期输出: [1, 3, 448, 112]
    print(numpy_array.dtype)  # 预期输出: float32

    # 测试 buffer_to_numpy 接口
    numpy_array = _C.device.buffer_to_numpy(buffer, np.dtype(np.float32))
    print(numpy_array.shape)  # 预期输出: [1, 3, 448, 112]
    print(numpy_array.dtype)  # 预期输出: float32

    # 测试 buffer_from_numpy 接口
    numpy_array = np.random.rand(1, 3, 224, 224).astype(np.float32)
    buffer = _C.device.buffer_from_numpy(numpy_array)
    print(buffer.getSizeVector())  # 预期输出: [1, 3, 224, 224]
    print(buffer.getSize())  # 预期输出: 602112
    print(buffer.getMemoryType())  # 预期输出: 0

# test tensor
def test_tensor():
    # 测试默认构造函数
    tensor = _C.device.Tensor()
    print(tensor)  # 预期输出: <nndeploy._nndeploy_internal.device.Tensor object at 0x...>

    # 测试带名称的构造函数 
    tensor = _C.device.Tensor("test_tensor")
    print(tensor.getName())  # 预期输出: test_tensor

    # 测试带TensorDesc的构造函数
    desc = _C.device.TensorDesc(common.DataType.from_name("float64"), common.DataFormat.NCHW.value, [1, 3, 640, 640])
    device = _C.device.getDevice(common.DeviceType("cpu", 0))
    tensor = _C.device.Tensor(device, desc, "test_tensor")
    print(tensor.getDesc())  # 预期输出: <nndeploy._nndeploy_internal.device.TensorDesc object at 0x...>

    # 测试clone方法
    cloned_tensor = tensor.clone()
    print(cloned_tensor)  # 预期输出: <nndeploy._nndeploy_internal.device.Tensor object at 0x...>

    # 测试reshape方法
    tensor.reshape([1, 3, 224, 224])
    print(tensor.getShape())  # 预期输出: [1, 3, 224, 224]

    # 测试empty方法
    print(tensor.empty())  # 预期输出: False

    # 测试getDataType方法
    print(tensor.getDataType())  # 预期输出: <nndeploy._nndeploy_internal.base.DataType object at 0x...>

    # 测试getDataFormat方法  
    print(tensor.getDataFormat())  # 预期输出: 0

    # 测试getShape方法
    print(tensor.getShape())  # 预期输出: [1, 3, 224, 224]

    # 测试getDeviceType方法
    print(tensor.getDeviceType())  # 预期输出: 0

    # 测试getDevice方法
    print(tensor.getDevice())  # 预期输出: <nndeploy._nndeploy_internal.device.Device object at 0x...>

    # 测试getData方法
    print(tensor.getData())  # 预期输出: <capsule object NULL at 0x...>

    # 测试to方法
    cpu_tensor = tensor.to(common.DeviceType("cpu", 0))
    print(cpu_tensor.getDeviceType())  # 预期输出: 0

if __name__ == "__main__":
    print("test_device start")
    test_device()
    # test_buffer_desc()
    # test_tensor_desc()
    # test_memory_pool()
    # test_buffer()
    # test_tensor()
    print("test_device end")
