#include "nndeploy/device/opencl/opencl_device.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy
{
    namespace device
    {
        TypeArchitectureRegister<OpenCLArchitecture> opencl_architecture_register(
            base::kDeviceTypeCodeOpenCL);
        
        /* this needs some sanitization of device_type_code i think */
        OpenCLArchitecture::OpenCLArchitecture(base::DeviceTypeCode device_type_code) 
        : Architecture(device_type_code) {};

        OpenCLArchitecture::~OpenCLArchitecture() {};

        base::Status OpenCLArchitecture::checkDevice(int device_id,
                                           std::string library_path) 
        {
            return base::kStatusCodeOk;
        }

        base::Status OpenCLArchitecture::enableDevice(int device_id,
                                                      std::string library_path)
        {
            base::DeviceType device_type(base::kDeviceTypeCodeOpenCL, device_id);

            return base::kStatusCodeOk;

        }

        base::Status enableDevice(int device_id = 0, std::string library_path)
        {
            return base::kStatusCodeOk;
        }

        Device* OpenCLArchitecture::getDevice(int device_id)
        {
            Device *device = nullptr;
            if (devices_.find(device_id) != devices_.end()) 
            {
                return devices_[device_id];
            } 
            else 
            {
                base::Status status = this->enableDevice(device_id, "");
                if (status == base::kStatusCodeOk) 
                {
                    device = devices_[device_id];
                } 
                else 
                {
                    NNDEPLOY_LOGE("enable device failed\n");
                }
            }
            return device;
        }

        std::vector<DeviceInfo> getDeviceInfo(std::string library_path)
        {
            
        }

        base::Status OpenCLDevice::init()
        {
            if(OpenCLSymbols::GetInstance()->LoadLibraryFromPath("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.9\\lib\\x64\\OpenCL.lib") == false)
            {
                NNDEPLOY_LOGE("load opencl lib failed!\n");
                return base::kStatusCodeErrorDeviceOpenCL;
            }
        }


    } 
}