
import nndeploy._nndeploy_internal as _C

import nndeploy.base
import nndeploy.base.common as common

# python3 nndeploy/device/device.py

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


class AbstractArchitecture(_C.device.Architecture):
    def __init__(self, device_type_code: nndeploy.base.common.DeviceTypeCode):
        super().__init__(device_type_code)

    def checkDevice(self, device_id: int = 0, library_path: str = ""):
        raise NotImplementedError()

    def enableDevice(self, device_id: int = 0, library_path: str = ""):
        raise NotImplementedError()

    def getDevice(self, device_id: int):
        raise NotImplementedError()

    def getDeviceInfo(self, library_path: str = ""):
        raise NotImplementedError()

    def getDeviceTypeCode(self):
        return nndeploy.base.DeviceTypeCode(super().getDeviceTypeCode())

    def disableDevice(self):
        raise NotImplementedError()
    
class RiscvArchitecture(AbstractArchitecture):
    def __init__(self):
        super().__init__(nndeploy.base.common.DeviceTypeCode.riscv)

    def checkDevice(self, device_id: int = 0, library_path: str = ""):
        return super().checkDevice(device_id, library_path)

    def enableDevice(self, device_id: int = 0, library_path: str = ""):
        return super().enableDevice(device_id, library_path)

    def getDevice(self, device_id: int):
        return super().getDevice(device_id)

    def getDeviceInfo(self, library_path: str = ""):
        return super().getDeviceInfo(library_path)
    
_C.device.registerArchitecture(common.DeviceTypeCode.riscv, RiscvArchitecture())

class Architecture():
    def __init__(self, device_type_code: nndeploy.base.common.DeviceTypeCode):
        self._architecture = _C.device.getArchitecture(device_type_code)

    def check_device(self, device_id: int = 0, library_path: str = ""):
        return self._architecture.checkDevice(device_id, library_path)

    def enable_device(self, device_id: int = 0, library_path: str = ""):
        return self._architecture.enableDevice(device_id, library_path)

    def disable_device(self):
        return self._architecture.disableDevice()

    def get_device(self, device_id: int):
        return self._architecture.getDevice(device_id)

    def get_device_info(self, library_path: str = ""):
        return [DeviceInfo(info) for info in self._architecture.getDeviceInfo(library_path)]

    def get_device_type_code(self):
        # return nndeploy.base.common.DeviceTypeCode(self._architecture.getDeviceTypeCode())
        return self._architecture.getDeviceTypeCode()

    def __str__(self):
        return self._architecture.__str__()


if __name__ == "__main__":
    architecture = Architecture(nndeploy.base.common.DeviceTypeCode.riscv)
    print(architecture.get_device_type_code())
    # print(architecture.get_device_info())
