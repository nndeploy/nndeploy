#ifndef _NNDEPLOY_DEVICE_OPENCL_OPENCL_RUNTIME_H_
#define _NNDEPLOY_DEVICE_OPENCL_OPENCL_RUNTIME_H_

#include "nndeploy/device/opencl/opencl_wrapper.h"

namespace nndeploy
{
    namespace device
    {
        class OpenCLRuntime
        {
            private:
                static bool init_done_;
            public:
                OpenCLRuntime(/* args */);
                ~OpenCLRuntime();
                base::Status init();

                cl::Context* context();
                uint8_t getPlatforms();
        };

    }
}

#endif /* _NNDEPLOY_DEVICE_OPENCL_OPENCL_RUNTIME_H_ */

