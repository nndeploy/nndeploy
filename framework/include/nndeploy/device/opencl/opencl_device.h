#ifndef _NNDEPLOY_DEVICE_OPENCL_DEVICE_H_
#define _NNDEPLOY_DEVICE_OPENCL_DEVICE_H_
#include <nndeploy/device/opencl/opencl_include.h>
#include <memory>
namespace nndeploy
{
    namespace device
    {
        /*
        * @todo inherit nndeploy::device::Architecture
        */
        class OpenCLDevice
        {
        private:
            /* data */
            cl_context mContext;
            cl_device_info mFirstGPUDevice;
            cl_command_queue mCommandQueue;
        public:
            cl_uint mPlatformIdCounts;
            OpenCLDevice(/* args */);
            ~OpenCLDevice();

            uint8_t getPlatformIdCounts();
        };
        
        OpenCLDevice::OpenCLDevice(/* args */) {}
        
        OpenCLDevice::~OpenCLDevice() {}

    } /* device */
} /* nndeploy */


#endif 
