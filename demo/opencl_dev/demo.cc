//# include "nndeploy/framework.h"
# include "nndeploy/device/opencl/opencl_runtime.h"
# include "nndeploy/device/opencl/cl.hpp"
using nndeploy::device::OpenCLRuntime;

int main()
{
    OpenCLRuntime* ocl_rt = new OpenCLRuntime();
    ocl_rt->init();
    delete ocl_rt;
}