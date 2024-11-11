import nndeploy._nndeploy_internal as _C

device_name_to_code = {
    "cpu": _C.base.DeviceTypeCode.cpu,
    "cuda": _C.base.DeviceTypeCode.cuda,
    "arm": _C.base.DeviceTypeCode.arm,
    "x86": _C.base.DeviceTypeCode.x86,
    "ascendcl": _C.base.DeviceTypeCode.ascendcl,
    "opencl": _C.base.DeviceTypeCode.opencl,
    "opengl": _C.base.DeviceTypeCode.opengl,
    "metal": _C.base.DeviceTypeCode.metal,
    "vulkan": _C.base.DeviceTypeCode.vulkan,
    "applenpu": _C.base.DeviceTypeCode.applenpu,
}

class DeviceType(_C.base.DeviceType):
    def __init__(self, device_name = "cpu", device_id =0):
        super().__init__()
        self.code_ = device_name_to_code[device_name]
        self.device_id_ = device_id