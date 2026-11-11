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
        return super().copy_to(dst)
    
    def serialize(self, bin_str=None):
        """Return binary data, or the legacy status when an argument is supplied."""
        if bin_str is None:
            return super().serialize()
        return super().serialize(bin_str)

    def deserialize(self, bin_str: bytes):
        """Deserialize the buffer from a binary string"""
        return super().deserialize(bin_str)
    
    def print(self):
        """Print buffer information"""
        return super().print()
    
    def just_modify(self, size):
        """Modify the buffer size"""
        return super().just_modify(size)
    
    def empty(self):
        """Check if the buffer is empty"""
        return super().empty()
    
    def get_device_type(self):
        """Get the device type of the buffer"""
        return super().get_device_type()
    
    def get_device(self):
        """Get the device of the buffer"""
        c_device = super().get_device()
        return Device(c_device)
    
    def get_memory_pool(self):
        """Get the memory pool of the buffer"""
        c_memory_pool = super().get_memory_pool()
        return MemoryPool(c_memory_pool) if c_memory_pool is not None else None
    
    def is_memory_pool(self):
        """Check if the buffer is from a memory pool"""
        return super().is_memory_pool()
    
    def get_desc(self):
        """Get the buffer descriptor"""
        return super().get_desc()
    
    def get_size(self):
        """Get the size of the buffer"""
        return super().get_size()
    
    def get_size_vector(self):
        """Get the size vector of the buffer"""
        return super().get_size_vector()
    
    def get_real_size(self):
        """Get the real size of the buffer"""
        return super().get_real_size()
    
    def get_real_size_vector(self):
        """Get the real size vector of the buffer"""
        return super().get_real_size_vector()
    
    def get_config(self):
        """Get the configuration of the buffer"""
        return super().get_config()
    
    def get_data(self):
        """Get the data pointer of the buffer"""
        return super().get_data()
    
    def get_memory_type(self):
        """Get the memory type of the buffer"""
        return super().get_memory_type()
    
    def add_ref(self):
        """Increase the reference count of the buffer"""
        return super().add_ref()
    
    def sub_ref(self):
        """Decrease the reference count of the buffer"""
        return super().sub_ref()
    
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
        return Buffer(_C.device.Buffer.from_numpy(array))

