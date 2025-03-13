
import nndeploy._nndeploy_internal as _C

import nndeploy.base


class DeviceInfo(_C.device.DeviceInfo):
    def __init__(self):
        super().__init__()
    
    @property
    def device_type(self):
        return self.device_type_
    
    @device_type.setter
    def device_type(self, value):
        self.device_type_ = value
    
    @property
    def is_support_fp16(self):
        return self.is_support_fp16_
    
    @is_support_fp16.setter
    def is_support_fp16(self, value):
        self.is_support_fp16_ = value


class Architecture():
    def __init__(self, *args, **kwargs):
        """
        Constructs an Architecture object.

        The constructor can be called in the following ways:
        1. str: Constructs an Architecture from a string. eg: "cpu", "ascendcl", etc.
        2. DeviceTypeCode: Constructs an Architecture from a DeviceTypeCode enum value.
        3. _C.base.DeviceTypeCode: Constructs an Architecture from a DeviceTypeCode enum value.
        """
        if len(args) == 1 and isinstance(args[0], str):
            device_type_code = nndeploy.base.DeviceTypeCode.from_name(args[0])
        elif len(args) == 1 and isinstance(args[0], nndeploy.base.DeviceTypeCode):
            device_type_code = args[0]
        elif len(args) == 1 and isinstance(args[0], _C.base.DeviceTypeCode):
            device_type_code = nndeploy.base.DeviceTypeCode(args[0])
        elif len(args) == 1 and isinstance(args[0], _C.device.Architecture):
            device_type_code = args[0].get_device_type_code()
        else:
            raise ValueError("Invalid arguments for Architecture constructor")
        # shared_ptr
        self._architecture = _C.device.get_architecture(device_type_code)

    def check_device(self, device_id: int = 0, library_path: str = ""):
        return self._architecture.check_device(device_id, library_path)

    def enable_device(self, device_id: int = 0, library_path: str = ""):
        return self._architecture.enable_device(device_id, library_path)

    def disable_device(self):
        return self._architecture.disable_device()

    def get_device(self, device_id: int):
        return self._architecture.get_device(device_id)

    def get_device_info(self, library_path: str = ""):
        return self._architecture.get_device_info(library_path)

    def get_device_type_code(self):
        return self._architecture.get_device_type_code()
    
    def insert_device(self, device_id: int, device: _C.device.Device):
        return self._architecture.insert_device(device_id, device)

    def __str__(self):           
        return self._architecture.__str__()
    
class Device():
    def __init__(self, *args, **kwargs):
        """
        Constructs a Device object.

        The constructor can be called in the following ways:
        1. Device(device_name_and_id): Constructs a Device from a string in the format "device_name:device_id" (e.g., "cuda:0").
        2. Device(device_name, device_id): Constructs a Device from a string device name and an integer device ID.
        3. Device(device_type_code): Constructs a Device from a DeviceTypeCode enum value. The device ID defaults to 0.
        4. Device(device_type_code, device_id): Constructs a Device from a DeviceTypeCode enum value and an integer device ID.
        5. Device(): Constructs a default Device with DeviceTypeCode.cpu and device ID 0.
        """
        if len(args) == 1 and isinstance(args[0], str):
            device_type = nndeploy.base.DeviceType(args[0])
        elif len(args) == 1 and isinstance(args[0], nndeploy.base.DeviceType):
            device_type = args[0]
        elif len(args) == 1 and isinstance(args[0], _C.base.DeviceType):
            device_type = nndeploy.base.DeviceType(args[0])
        elif len(args) == 1 and isinstance(args[0], _C.device.Device):
            device_type = args[0].get_device_type()
        elif len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], int):
            device_type = nndeploy.base.DeviceType(args[0], args[1])
        elif len(args) == 1 and isinstance(args[0], nndeploy.base.DeviceTypeCode):
            device_type = nndeploy.base.DeviceType(args[0])
        elif len(args) == 2 and isinstance(args[0], nndeploy.base.DeviceTypeCode) and isinstance(args[1], int):
            device_type = nndeploy.base.DeviceType(args[0], args[1])
        elif len(args) == 1 and isinstance(args[0], _C.base.DeviceTypeCode):
            device_type = nndeploy.base.DeviceType(args[0])
        elif len(args) == 2 and isinstance(args[0], _C.base.DeviceTypeCode) and isinstance(args[1], int):
            device_type = nndeploy.base.DeviceType(args[0], args[1])
        elif len(args) == 0:
            device_type = nndeploy.base.DeviceType()
        else:
            raise ValueError("Invalid arguments for Device constructor")
        
        library_path = kwargs.get("library_path", "")
        status = _C.device.enable_device(device_type, library_path)
        if status == nndeploy.base.StatusCode.Ok:
            self._device = _C.device.get_device(device_type)
        else:
            raise ValueError("Failed to enable device")
    
    def get_data_format_by_shape(self, shape):
        return self._device.get_data_format_by_shape(shape)
    
    def to_buffer_desc(self, desc, config):
        return self._device.to_buffer_desc(desc, config)
    
    def allocate(self, size_or_desc):
        """
        Allocate memory that must be freed using deallocate()
        Args:
            size_or_desc: int for allocation size, or BufferDesc for buffer descriptor
        Returns:
            Memory pointer that must be freed using deallocate()
        """
        if isinstance(size_or_desc, int):
            return self._device.allocate(size_or_desc)
        else:
            return self._device.allocate(size_or_desc)
    
    def deallocate(self, ptr):
        return self._device.deallocate(ptr)
    
    def allocate_pinned(self, size_or_desc):
        """
        Allocate pinned memory that must be freed using deallocate_pinned()
        Args:
            size_or_desc: int for allocation size, or BufferDesc for buffer descriptor
        Returns:
            Memory pointer that must be freed using deallocate_pinned()
        """
        if isinstance(size_or_desc, int):
            return self._device.allocate_pinned(size_or_desc)
        else:
            return self._device.allocate_pinned(size_or_desc)
    
    def deallocate_pinned(self, ptr):
        return self._device.deallocate_pinned(ptr)
    
    def copy(self, src, dst, size, stream=None):
        return self._device.copy(src, dst, size, stream)
    
    def download(self, src, dst, size, stream=None):
        return self._device.download(src, dst, size, stream)
    
    def upload(self, src, dst, size, stream=None):
        return self._device.upload(src, dst, size, stream)
    
    def copy_buffer(self, src, dst, stream=None):
        return self._device.copy(src, dst, stream)
    
    def download_buffer(self, src, dst, stream=None):
        return self._device.download(src, dst, stream)
    
    def upload_buffer(self, src, dst, stream=None):
        return self._device.upload(src, dst, stream)
    
    def get_context(self):
        return self._device.get_context()
    
    def create_stream(self, stream=None):
        if stream is None:
            return self._device.create_stream()
        else:
            return self._device.create_stream(stream)
    
    def create_event(self):
        return self._device.create_event()
    
    def create_events(self, events):
        return self._device.create_events(events)
    
    def get_device_type(self):
        return self._device.get_device_type()
    
    def init(self):
        return self._device.init()
    
    def deinit(self):
        return self._device.deinit()
    
    def __str__(self):
        return self._device.__str__()
    

class Stream():
    def __init__(self, device_or_type, stream=None):
        if isinstance(device_or_type, Device):
            device_type = device_or_type.get_device_type()
        else:
            device_type = device_or_type
        self._stream = _C.device.create_stream(device_type, stream)

    def get_device_type(self):
        return self._stream.get_device_type()
    
    def get_device(self):
        return self._stream.get_device()
    
    def synchronize(self):
        return self._stream.synchronize()
    
    def record_event(self, event):
        return self._stream.record_event(event)
    
    def wait_event(self, event):
        return self._stream.wait_event(event)
    
    def on_execution_context_setup(self):
        return self._stream.on_execution_context_setup()
    
    def on_execution_context_teardown(self):
        return self._stream.on_execution_context_teardown()
    
    def get_native_stream(self):
        return self._stream.get_native_stream()
    
    def __str__(self):
        return self._stream.__str__()


class Event():
    def __init__(self, device_or_type):
        if isinstance(device_or_type, Device):
            device_type = device_or_type.get_device_type()
        else:
            device_type = device_or_type
        self._event = _C.device.create_event(device_type)

    def get_device_type(self):
        return self._event.get_device_type()
    
    def get_device(self):
        return self._event.get_device()
    
    def query_done(self):
        return self._event.query_done()
    
    def synchronize(self):
        return self._event.synchronize()
    
    def get_native_event(self):
        return self._event.get_native_event()
    
    def __str__(self):
        return self._event.__str__()
    