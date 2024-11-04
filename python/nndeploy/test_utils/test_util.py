import nndeploy
import numpy as np

str_to_np_data_types = {
    'float32': np.float32,
    'float16': np.float16
}


device_name_to_code = {
    'cpu': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeCpu,
    'arm': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeArm,
    'x86': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeX86,
    'riscv': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeRiscV,
    'cuda': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeCuda,
    'rocm': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeRocm,
    'sycl': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeSyCL,
    'opencl': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeOpenCL,
    'opengl': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeOpenGL,
    'metal': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeMetal,
    'vulkan': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeVulkan,
    'hexagon': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeHexagon,
    'mtkvpu': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeMtkVpu,
    'ascendcl': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeAscendCL,
    'applenpu': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeAppleNpu,
    'rknpu': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeRkNpu,
    'qualcomnpu': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeQualcommNpu,
    'mtknpu': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeMtkNpu,
    'sophonnpu': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeSophonNpu,
    'notsupport': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeNotSupport}


# 从numpy array返回一个Tensor
def createTensorFromNumpy(np_data):

    tensor = nndeploy.device.Tensor(np_data, device_name_to_code["cpu"])
    return tensor

# 从Tensor返回一个numpy array
def createNumpyFromTensor(tensor):
    return np.array(tensor.to(device_name_to_code["cpu"]))
