import nndeploy
import numpy as np

str_to_np_data_types = {
    'float32':np.float32,
    'float16':np.float16
}


device_name_to_code = {
    'cpu': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeCpu,
    'cuda': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeCuda,
    'arm': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeArm,
    'x86': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeX86,
    'ascendcl': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeAscendCL,
    'opencl': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeOpenCL,
    'opengl': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeOpenGL,
    'metal': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeMetal,
    'vulkan': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeVulkan,
    'applenpu': nndeploy.base.DeviceTypeCode.kDeviceTypeCodeAppleNpu}



def createTensorFromNumpy(np_data, device):

    tensor = nndeploy.device.Tensor(np_data, device_name_to_code[device])
    return tensor
