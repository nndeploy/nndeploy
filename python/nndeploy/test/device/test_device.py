
import nndeploy._nndeploy_internal as _C

from enum import Enum
import numpy as np
import nndeploy.base.common as common
import time


## cd ../../../
## pip install -e .
## cd nndeploy/test/device python3 test_device.py
## python3 nndeploy/test/device/test_device.py


class VulkanArchitecture(_C.device.Architecture):
    def __init__(self):
        super().__init__(common.DeviceTypeCode.vulkan.value)        
        
    def checkDevice(self, device_id=0, library_path=""):
        print("checkDevice")
        return common.Status(common.StatusCode.ok)

    def enableDevice(self, device_id=0, library_path=""):
        print("enableDevice")
        return common.Status(common.StatusCode.ok)
    
    def getDevice(self, device_id):
        return VulkanDevice()
    
    def getDeviceInfo(self, library_path=""):
        print("getDeviceInfo")
        return []

_C.device.registerArchitecture(common.DeviceTypeCode.vulkan.value, VulkanArchitecture())

class VulkanDevice(_C.device.Device):
    def __init__(self):
        super().__init__(common.DeviceType.vulkan)

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
        return common.Status(common.StatusCode.ok)

    def download(self, src, dst, size, stream=None):
        print("download") 
        return common.Status(common.StatusCode.ok)

    def upload(self, src, dst, size, stream=None):
        print("upload")
        return common.Status(common.StatusCode.ok)

    def init(self):
        print("init")
        return common.Status(common.StatusCode.ok)

    def deinit(self):
        print("deinit") 
        return common.Status(common.StatusCode.ok)


def test_device():
    architecture = _C.device.getArchitecture(common.DeviceTypeCode.vulkan.value)
    print(architecture.getDeviceTypeCode())
    architecture = _C.device.getArchitecture(common.DeviceTypeCode.cpu.value)
    print(architecture.getDeviceTypeCode())
    device = architecture.getDevice(0)
    stream = device.createStream()
    print(stream)
    event = device.createEvent()
    print(event)
    print(device)

    device_type = common.DeviceType("cpu", 0)
    print(device_type)
    device = _C.device.getDevice(device_type)
    print(device)
    stream = device.createStream()
    print(stream)
    event = device.createEvent()
    print(event)

    device = _C.device.getDevice(common.DeviceType("cuda", 0))
    stream = _C.device.createStream(common.DeviceType("cuda", 0))
    print(stream)
    event = _C.device.createEvent(common.DeviceType("cuda", 0))
    print(event)


def test_type():
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

    


    tensor_desc = _C.device.TensorDesc()
    print(tensor_desc)

if __name__ == "__main__":
    print("test_device start")
    # test_device()
    test_type()
    print("test_device end")
