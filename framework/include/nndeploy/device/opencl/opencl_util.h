#ifndef _NNDEPLOY_DEVICE_OPENCL_OPENCL_UTIL_H_
#define _NNDEPLOY_DEVICE_OPENCL_OPENCL_UTIL_H_

#include "nndeploy/device/opencl/opencl_include.h"

namespace nndeploy
{
    namespace device
    {
        // #define NNDEPLOY_OPENCL_CHECK(status)                                 \
        // {                                                                 \
        //     std::stringstream _error;                                       \
        //     if (CL_SUCCESS != status) \
        //     {                                                                \
        //         _error << "OpenCL failure " << status << ": " << (status) << " " \
        //         NNDEPLOY_CUDA_FETAL_ERROR(_error.str());                      \
        //     }                                                               \
        // }

        inline int openCLGetNumDevices()
        {
            
        }
    }
}

#endif