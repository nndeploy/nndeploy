import nndeploy._nndeploy_internal as _C

device_name_to_code = {
    "cpu": _C.base.DeviceTypeCode.kDeviceTypeCodeCpu,
    "cuda": _C.base.DeviceTypeCode.kDeviceTypeCodeCuda,
    "arm": _C.base.DeviceTypeCode.kDeviceTypeCodeArm,
    "x86": _C.base.DeviceTypeCode.kDeviceTypeCodeX86,
    "ascendcl": _C.base.DeviceTypeCode.kDeviceTypeCodeAscendCL,
    "opencl": _C.base.DeviceTypeCode.kDeviceTypeCodeOpenCL,
    "opengl": _C.base.DeviceTypeCode.kDeviceTypeCodeOpenGL,
    "metal": _C.base.DeviceTypeCode.kDeviceTypeCodeMetal,
    "vulkan": _C.base.DeviceTypeCode.kDeviceTypeCodeVulkan,
    "applenpu": _C.base.DeviceTypeCode.kDeviceTypeCodeAppleNpu,
}