#include "nndeploy/device/opencl/opencl_device.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy
{
    namespace device
    {
        TypeArchitectureRegister<OpenCLArchitecture> opencl_architecture_register(
            base::kDeviceTypeCodeOpenCL);
    } 
}