#include "nndeploy/device/opencl/opencl_runtime.h"

namespace nndeploy
{
    namespace device
    {
        OpenCLRuntime::OpenCLRuntime()
        {
            NNDEPLOY_LOGI("opencl runtime start\n");
        }
        
        base::Status OpenCLRuntime::init()
        {
            if(!init_done_)
            {
                NNDEPLOY_LOGI("init opencl rt\n");
                
                #ifdef NNDEPLOY_USE_OPENCL_WRAPPER
                    if(OpenCLSymbols::GetInstance()->LoadOpenCLLibrary() == false)
                    {
                        NNDEPLOY_LOGI("load opencl lib failed!\n");
                        return base::kStatusCodeErrorDeviceOpenCL;
                    }
                #endif
                NNDEPLOY_LOGI("init opencl rt\n");
            }
            return base::kStatusCodeOk;
        }

    } /* device */


} /* nndeploy */