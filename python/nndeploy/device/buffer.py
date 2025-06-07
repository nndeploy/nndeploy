import nndeploy._nndeploy_internal as _C

import numpy as np

import nndeploy.base
from .device import Device
from .type import BufferDesc
from .memory_pool import MemoryPool


class Buffer(_C.device.Buffer):
    def __init__(self, *args, **kwargs):
        # 将Device转换为_C.device.Device
        if len(args) > 0 and isinstance(args[0], Device):
            args = (args[0]._device,) + args[1:]
        # 将MemoryPool转换为_C.device.MemoryPool
        elif len(args) > 0 and isinstance(args[0],MemoryPool):
            args = (args[0]._memory_pool,) + args[1:]
        super().__init__(*args, **kwargs)
    
    def clone(self):
        """Clone the buffer"""
        return super().clone()
    
    def copy_to(self, dst):
        """Copy the buffer to the destination buffer"""
        return super().copyTo(dst)
    
    def serialize(self, bin_str: str):
        """Serialize the buffer to a binary string"""
        return super().serialize(bin_str)
    
    def deserialize(self, bin_str: str):
        """Deserialize the buffer from a binary string"""
        return super().deserialize(bin_str)
    
    def print(self):
        """Print buffer information"""
        return super().print()
    
    def just_modify(self, size):
        """Modify the buffer size"""
        return super().justModify(size)
    
    def empty(self):
        """Check if the buffer is empty"""
        return super().empty()
    
    def get_device_type(self):
        """Get the device type of the buffer"""
        return super().getDeviceType()
    
    def get_device(self):
        """Get the device of the buffer"""
        c_device = super().getDevice()
        return Device(c_device)
    
    def get_memory_pool(self):
        """Get the memory pool of the buffer"""
        c_memory_pool = super().getMemoryPool()
        return MemoryPool(c_memory_pool)
    
    def is_memory_pool(self):
        """Check if the buffer is from a memory pool"""
        return super().isMemoryPool()
    
    def get_desc(self):
        """Get the buffer descriptor"""
        return super().getDesc()
    
    def get_size(self):
        """Get the size of the buffer"""
        return super().getSize()
    
    def get_size_vector(self):
        """Get the size vector of the buffer"""
        return super().getSizeVector()
    
    def get_real_size(self):
        """Get the real size of the buffer"""
        return super().getRealSize()
    
    def get_real_size_vector(self):
        """Get the real size vector of the buffer"""
        return super().getRealSizeVector()
    
    def get_config(self):
        """Get the configuration of the buffer"""
        return super().getConfig()
    
    def get_data(self):
        """Get the data pointer of the buffer"""
        return super().getData()
    
    def get_memory_type(self):
        """Get the memory type of the buffer"""
        return super().getMemoryType()
    
    def add_ref(self):
        """Increase the reference count of the buffer"""
        return super().addRef()
    
    def sub_ref(self):
        """Decrease the reference count of the buffer"""
        return super().subRef()
    
    def to_numpy(self, *args, **kwargs):
        """Convert the buffer to numpy array
        支持两种调用方式：
        1. to_numpy(dtype) - 直接传入dtype对象
        2. to_numpy(dtype_obj) - 传入可转换为dtype的对象
        """
        if len(args) == 1 and isinstance(args[0], np.dtype):
            return super().to_numpy_v0(args[0])
        else:
            return super().to_numpy_v1(*args, **kwargs)
    
    @staticmethod
    def from_numpy(array):
        """Convert numpy array to buffer"""
        return _C.device.Buffer.from_numpy(array)   

