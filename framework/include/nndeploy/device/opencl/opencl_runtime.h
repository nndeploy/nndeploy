#ifndef _NNDEPLOY_DEVICE_OPENCL_OPENCL_RUNTIME_H_
#define _NNDEPLOY_DEVICE_OPENCL_OPENCL_RUNTIME_H_

#include <mutex>
#include <string>

#include "nndeploy/device/opencl/opencl_wrapper.h"

namespace nndeploy
{
    namespace device
    {
        class NNDEPLOY_CC_API OpenCLRuntime
        {
            private:
                bool init_done_ = false;
                cl_uint num_platforms;
                base::Status getPlatformDetails();
                base::Status getDeviceDetails();
                std::unordered_map<int, cl::Device> device_map;
            
            public:
                OpenCLRuntime();
                ~OpenCLRuntime();
                
                /**
                 * Query GPU devices for each OpenCL platform and create a separate
                 * context per device  
                 */
                base::Status init();
        };

    }
}

#endif /* _NNDEPLOY_DEVICE_OPENCL_OPENCL_RUNTIME_H_ */

