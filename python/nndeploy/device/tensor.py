import nndeploy._nndeploy_internal as _C

import numpy as np

import nndeploy.base
from .device import Device
from .type import BufferDesc, TensorDesc
from .memory_pool import MemoryPool
from .buffer import Buffer

# 从numpy array返回一个Tensor
def create_tensor_from_numpy(np_data, device="cpu"):
    device_type = nndeploy.base.DeviceType(device)
    c_tensor = _C.device.Tensor.from_numpy(np_data, device_type)
    tensor = Tensor(c_tensor)
    return tensor

# 从Tensor返回一个numpy array
def create_numpy_from_tensor(tensor):
    # return np.array(tensor.to(nndeploy.base.name_to_device_type_code["cpu"]))
    return np.array(tensor.to(nndeploy.base.DeviceType("cpu")))


class Tensor(_C.device.Tensor):
    def __init__(self, *args, **kwargs):
        """
        Tensor构造函数。

        Args:
            name (str, optional): Tensor的名称。
            desc (TensorDesc, optional): Tensor的描述信息，包含数据类型、形状等。
            buffer (Buffer, optional): 已存在的Buffer对象，用于直接构造Tensor。
            device (Device, optional): 设备对象，指定Tensor所在的设备。
            data_ptr (void*, optional): 外部数据指针，用于直接使用已有内存构造Tensor。
            memory_pool (MemoryPool, optional): 内存池对象，用于从内存池中分配内存。
            config (List[int], optional): 配置参数，用于指定内存对齐等配置。
            *args: 其他位置参数。
            **kwargs: 其他关键字参数。

        Note:
            构造函数有以下几种形式:
            1. Tensor(name) - 仅指定名称创建空Tensor
            2. Tensor(desc, name="") - 使用TensorDesc创建Tensor
            3. Tensor(desc, buffer, name="") - 使用已有Buffer创建Tensor
            4. Tensor(device, desc, name="", config=[]) - 在指定设备上创建Tensor
            5. Tensor(device, desc, data_ptr, name="", config=[]) - 使用已有内存在设备上创建Tensor
            6. Tensor(memory_pool, desc, name="", config=[]) - 从内存池创建Tensor
            7. Tensor(memory_pool, desc, data_ptr, name="", config=[]) - 使用已有内存从内存池创建Tensor

        Notes:
            - 如果第一个参数是Device对象，它将被转换为_C.device.Device对象。
            - 如果第一个参数是MemoryPool对象，它将被转换为_C.device.MemoryPool对象。
        """
        # 将Device转换为_C.device.Device
        if len(args) > 1 and isinstance(args[0], Device):
            args = (args[0]._device,) + args[1:]
        # 将MemoryPool转换为_C.device.MemoryPool
        elif len(args) > 1 and isinstance(args[0], MemoryPool):
            args = (args[0]._memory_pool,) + args[1:]
        super().__init__(*args, **kwargs)

    def create(self, *args, **kwargs):
        """
        创建Tensor。

        Args:
            *args: 传递给父类create函数的位置参数。
            **kwargs: 传递给父类create函数的关键字参数。

        Returns:
            None
        """
        return super().create(*args, **kwargs)

    def clear(self):
        """
        清空Tensor。

        Returns:
            None
        """
        return super().clear()

    def allocate(self, *args, **kwargs):
        """
        分配Tensor内存。

        Args:
            *args: 传递给父类allocate函数的位置参数。
            **kwargs: 传递给父类allocate函数的关键字参数。

        Returns:
            None
        """
        return super().allocate(*args, **kwargs)

    def deallocate(self):
        """
        释放Tensor内存。

        Returns:
            None
        """
        return super().deallocate()

    def set(self, value):
        """
        设置Tensor的值。

        Args:
            value: 要设置的值。

        Returns:
            None
        """
        return super().set(value)

    def reshape(self, shape):
        """
        重塑Tensor的形状。

        Args:
            shape (tuple): 新的Tensor形状。

        Returns:
            None
        """
        return super().reshape(shape)

    def just_modify(self, desc=None, buffer=None, is_external=True):
        """
        修改Tensor描述符或缓冲区。

        Args:
            desc (TensorDesc, optional): 新的Tensor描述符。默认为None。
            buffer (Buffer, optional): 新的缓冲区。默认为None。
            is_external (bool, optional): 缓冲区是否为外部的。默认为True。

        Returns:
            None

        Raises:
            ValueError: 如果desc和buffer都没有提供。
        """
        if desc is not None:
            return super().just_modify(desc)
        elif buffer is not None:
            return super().just_modify(buffer, is_external)
        else:
            raise ValueError("必须提供desc或buffer。")

    def clone(self):
        """
        克隆Tensor。

        Returns:
            Tensor: 克隆的Tensor。
        """
        return super().clone()

    def copy_to(self, dst):
        """
        将Tensor复制到目标Tensor。

        Args:
            dst (Tensor): 目标Tensor。

        Returns:
            None
        """
        return super().copy_to(dst)

    def serialize(self, bin_str: str):
        """
        序列化Tensor到流。

        Args:
            stream: 输出流。

        Returns:
            None
        """
        return super().serialize(bin_str)

    def deserialize(self, bin_str: str):
        """
        从流反序列化Tensor。

        Args:
            stream: 输入流。

        Returns:
            None
        """
        return super().deserialize(bin_str)

    def print(self, stream=None):
        """
        打印Tensor信息。

        Args:
            stream: 输出流。默认为None。

        Returns:
            None
        """
        return super().print(stream)

    def is_same_device(self, tensor):
        """
        检查Tensor是否在同一设备上。

        Args:
            tensor (Tensor): 要比较的Tensor。

        Returns:
            bool: 如果Tensor在同一设备上，则为True，否则为False。
        """
        return super().is_same_device(tensor)

    def is_same_memory_pool(self, tensor):
        """
        检查Tensor是否在同一内存池中。

        Args:
            tensor (Tensor): 要比较的Tensor。

        Returns:
            bool: 如果Tensor在同一内存池中，则为True，否则为False。
        """
        return super().is_same_memory_pool(tensor)

    def is_same_desc(self, tensor):
        """
        检查Tensor是否具有相同的描述符。

        Args:
            tensor (Tensor): 要比较的Tensor。

        Returns:
            bool: 如果Tensor具有相同的描述符，则为True，否则为False。
        """
        return super().is_same_desc(tensor)

    def empty(self):
        """
        检查Tensor是否为空。

        Returns:
            bool: 如果Tensor为空，则为True，否则为False。
        """
        return super().empty()

    def is_continue(self):
        """
        检查Tensor数据是否连续。

        Returns:
            bool: 如果Tensor数据连续，则为True，否则为False。
        """
        return super().is_continue()

    def is_external_buffer(self):
        """
        检查Tensor缓冲区是否为外部的。

        Returns:
            bool: 如果Tensor缓冲区为外部的，则为True，否则为False。
        """
        return super().is_external_buffer()

    def get_name(self):
        """
        获取Tensor名称。

        Returns:
            str: Tensor名称。
        """
        return super().get_name()

    def set_name(self, name):
        """
        设置Tensor名称。

        Args:
            name (str): 新的Tensor名称。

        Returns:
            None
        """
        return super().set_name(name)

    def get_desc(self):
        """
        获取Tensor描述符。

        Returns:
            TensorDesc: Tensor描述符。
        """
        return super().get_desc()

    def get_data_type(self):
        """
        获取Tensor数据类型。

        Returns:
            DataType: Tensor数据类型。
        """
        return super().get_data_type()

    def set_data_type(self, data_type):
        """
        设置Tensor数据类型。

        Args:
            data_type (DataType): 新的Tensor数据类型。

        Returns:
            None
        """
        return super().set_data_type(data_type)

    def get_data_format(self):
        """
        获取Tensor数据格式。

        Returns:
            DataFormat: Tensor数据格式。
        """
        return super().get_data_format()

    def set_data_format(self, data_format):
        """
        设置Tensor数据格式。

        Args:
            data_format (DataFormat): 新的Tensor数据格式。

        Returns:
            None
        """
        return super().set_data_format(data_format)

    def get_shape(self):
        """
        获取Tensor形状。

        Returns:
            tuple: Tensor形状。
        """
        return super().get_shape()

    def get_shape_index(self, index):
        """
        获取指定索引处的形状值。

        Args:
            index (int): 索引。

        Returns:
            int: 指定索引处的形状值。
        """
        return super().get_shape_index(index)

    def get_batch(self):
        """
        获取Tensor批次大小。

        Returns:
            int: Tensor批次大小。
        """
        return super().get_batch()

    def get_channel(self):
        """
        获取Tensor通道数。

        Returns:
            int: Tensor通道数。
        """
        return super().get_channel()

    def get_depth(self):
        """
        获取Tensor深度。

        Returns:
            int: Tensor深度。
        """
        return super().get_depth()

    def get_height(self):
        """
        获取Tensor高度。

        Returns:
            int: Tensor高度。
        """
        return super().get_height()

    def get_width(self):
        """
        获取Tensor宽度。

        Returns:
            int: Tensor宽度。
        """
        return super().get_width()

    def get_stride(self):
        """
        获取Tensor步长。

        Returns:
            tuple: Tensor步长。
        """
        return super().get_stride()

    def get_stride_index(self, index):
        """
        获取指定索引处的步长值。

        Args:
            index (int): 索引。

        Returns:
            int: 指定索引处的步长值。
        """
        return super().get_stride_index(index)

    def get_buffer(self):
        """
        获取Tensor缓冲区。

        Returns:
            Buffer: Tensor缓冲区。
        """
        return super().get_buffer()

    def get_device_type(self):
        """
        获取Tensor设备类型。

        Returns:
            DeviceType: Tensor设备类型。
        """
        return super().get_device_type()

    def get_device(self):
        """
        获取Tensor设备。

        Returns:
            Device: Tensor设备。
        """
        c_device = super().get_device()
        return Device(c_device)

    def get_memory_pool(self):
        """
        获取Tensor内存池。

        Returns:
            MemoryPool: Tensor内存池。
        """
        c_memory_pool = super().get_memory_pool()
        return MemoryPool(c_memory_pool)

    def is_memory_pool(self):
        """
        检查Tensor是否来自内存池。

        Returns:
            bool: 如果Tensor来自内存池，则为True，否则为False。
        """
        return super().is_memory_pool()

    def get_buffer_desc(self):
        """
        获取Tensor缓冲区描述符。

        Returns:
            BufferDesc: Tensor缓冲区描述符。
        """
        return super().get_buffer_desc()

    def get_size(self):
        """
        获取Tensor大小。

        Returns:
            int: Tensor大小。
        """
        return super().get_size()

    def get_size_vector(self):
        """
        获取Tensor大小向量。

        Returns:
            list: Tensor大小向量。
        """
        return super().get_size_vector()

    def get_real_size(self):
        """
        获取Tensor实际大小。

        Returns:
            int: Tensor实际大小。
        """
        return super().get_real_size()

    def get_real_size_vector(self):
        """
        获取Tensor实际大小向量。

        Returns:
            list: Tensor实际大小向量。
        """
        return super().get_real_size_vector()

    def get_config(self):
        """
        获取Tensor配置。

        Returns:
            list: Tensor配置。
        """
        return super().get_config()

    def get_data(self):
        """
        获取Tensor数据指针。

        Returns:
            int: Tensor数据指针。
        """
        return super().get_data()

    def get_memory_type(self):
        """
        获取Tensor内存类型。

        Returns:
            MemoryType: Tensor内存类型。
        """
        return super().get_memory_type()

    def add_ref(self):
        """
        增加Tensor引用计数。

        Returns:
            None
        """
        return super().add_ref()

    def sub_ref(self):
        """
        减少Tensor引用计数。

        Returns:
            None
        """
        return super().sub_ref()

    def to_numpy(self):
        """
        将Tensor转换为numpy数组。

        Returns:
            numpy.ndarray: numpy数组。
        """
        return super().to_numpy()

    def to(self, device_type):
        """
        将Tensor移动到指定设备。

        Args:
            device_type (DeviceType): 目标设备类型。

        Returns:
            Tensor: 移动后的Tensor。
        """
        return super().to(device_type)

    @staticmethod
    def from_numpy(array, device_type=nndeploy.base.DeviceType("cpu")):
        """
        从numpy数组创建Tensor。

        Args:
            array (numpy.ndarray): numpy数组。
            device_type (DeviceType): 目标设备类型。

        Returns:
            Tensor: 创建的Tensor。
        """
        c_tensor = _C.device.Tensor.from_numpy(array, device_type)
        return Tensor(c_tensor)
    