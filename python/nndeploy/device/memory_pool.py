import nndeploy._nndeploy_internal as _C

import nndeploy.base
from .device import Device
from .type import BufferDesc


class MemoryPool():
    def __init__(self, *args, **kwargs):
        """
        MemoryPool构造函数。

        Args:
            device (Device): 设备对象。
            memory_pool_type (MemoryPoolType): 内存池类型。
            memory_pool (_C.device.MemoryPool): 内部内存池对象。
        """
        if len(args) == 1 and isinstance(args[0], _C.device.MemoryPool):
            self._memory_pool = args[0]
        elif len(args) == 2:
            device, memory_pool_type = args
            self._memory_pool = _C.device.MemoryPool(device._device, memory_pool_type)
        else:
            raise ValueError("Invalid arguments for MemoryPool constructor")

    def init(self, *args):
        if len(args) == 0:
            return self._memory_pool.init()
        elif len(args) == 1:
            return self._memory_pool.init(args[0])
        elif len(args) == 2:
            ptr, size = args
            if isinstance(size, int):
                return self._memory_pool.init(ptr, size)
        raise TypeError("Invalid arguments for init()")

    def deinit(self):
        return self._memory_pool.deinit()

    def allocate(self, size: int):
        return self._memory_pool.allocate(size)

    def allocate(self, desc: BufferDesc):
        return self._memory_pool.allocate(desc)

    def deallocate(self, ptr):
        return self._memory_pool.deallocate(ptr)

    def allocate_pinned(self, size: int):
        return self._memory_pool.allocate_pinned(size)
    
    def allocate_pinned(self, desc: BufferDesc):
        return self._memory_pool.allocate_pinned(desc)
    
    def deallocate_pinned(self, ptr):
        return self._memory_pool.deallocate_pinned(ptr)

    def get_device(self):
        return self._memory_pool.get_device()
    
    def get_memory_pool_type(self):
        return self._memory_pool.get_memory_pool_type()
