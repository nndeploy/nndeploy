#include "nndeploy/framework.h"
#include "nndeploy/device/device.h"
// # include "nndeploy/device/opencl/opencl_runtime.h"
// # include "nndeploy/device/opencl/cl.hpp"

using namespace nndeploy;

int main()
{
    // OpenCLRuntime* ocl_rt = new OpenCLRuntime();
    // //ocl_rt->init_done_ = false;
    // ocl_rt->init();
    // delete ocl_rt;
    base::DeviceType device_type = base::kDeviceTypeCodeOpenCL;
    device_type.device_id_ = 0;
    auto device = device::getDevice(device_type);
    
    
}