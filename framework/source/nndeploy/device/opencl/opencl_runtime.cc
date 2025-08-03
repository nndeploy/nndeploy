#include "nndeploy/device/opencl/opencl_runtime.h"

namespace nndeploy
{
    namespace device
    {
        // bool OpenCLRuntime::init_done_ = false;
        
        OpenCLRuntime::OpenCLRuntime()
        {
            NNDEPLOY_LOGI("opencl runtime start\n");
        }

        base::Status OpenCLRuntime::getPlatformDetails() {
            cl_int err;

            err = clGetPlatformIDs(0, NULL, &num_platforms);
            if (err != CL_SUCCESS || num_platforms <= 0) {
                return base::kStatusCodeErrorDeviceOpenCL;
            }

            std::vector<cl_platform_id> platforms(num_platforms);
            err = clGetPlatformIDs(num_platforms, platforms.data(), NULL);
            if (err != CL_SUCCESS) {
                return base::kStatusCodeErrorDeviceOpenCL;
            }

            for(cl_uint i = 0; i < num_platforms; i++) {
                cl_uint num_devices;
                err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
                if (err != CL_SUCCESS) {
                    continue;
                }

                std::vector<cl_device_id> devices(num_devices);
                err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, num_devices, devices.data(), NULL);
                if (err != CL_SUCCESS) {
                    continue;
                }

                for(cl_uint j = 0; j < num_devices; j++) {
                    char device_name[64];
                    err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
                    if (err == CL_SUCCESS) {
                        printf("Platform %d, Device %d: %s\n", i, j, device_name);
                    }
                }
            }
            return base::kStatusCodeOk;
        }

        base::Status OpenCLRuntime::getDeviceDetails()
        {
            if(getPlatformDetails() == base::kStatusCodeOk)
            {
                NNDEPLOY_LOGD("Num. of OpenCL platforms: %d\n", num_platforms);
                return base::kStatusCodeOk;
            }
            return base::kStatusCodeErrorDeviceOpenCL;
        }
        
        base::Status OpenCLRuntime::init()
        {
            if(!init_done_)
            {
                NNDEPLOY_LOGD("init opencl rt\n");
                init_done_ = true;
                #ifdef NNDEPLOY_USE_OPENCL_WRAPPER
                    if(OpenCLSymbols::GetInstance()->LoadOpenCLLibrary() == false)
                    {
                        NNDEPLOY_LOGE("load opencl lib failed!\n");
                        return base::kStatusCodeErrorDeviceOpenCL;
                    }
                #endif
                NNDEPLOY_LOGD("init opencl rt\n");
                getDeviceDetails();
            }
            init_done_ = true;
            return base::kStatusCodeOk;
        }

        OpenCLRuntime::~OpenCLRuntime()
        {

        }

    } /* device */


} /* nndeploy */