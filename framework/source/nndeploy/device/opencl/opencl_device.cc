#include "nndeploy/device/opencl/opencl_device.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy
{
    namespace device
    {
        TypeArchitectureRegister<OpenCLArchitecture> opencl_architecture_register(
            base::kDeviceTypeCodeOpenCL);
        
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
            OpenCLDevice* device = new OpenCLDevice(device_type, library_path);
            device->init();
            devices_.insert({device_id, device});
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

        std::vector<DeviceInfo> OpenCLArchitecture::getDeviceInfo(std::string library_path) 
        {
            std::vector<DeviceInfo> device_info_list;
            return device_info_list;
        }

        /* OpenCLDevice */

        BufferDesc OpenCLDevice::toBufferDesc(const TensorDesc &desc,
                                    const base::IntVector &config)
        {
            NNDEPLOY_LOGE("Not Implemented\n");
            size_t size = 0;
            return BufferDesc(size, config);
        }

        void* OpenCLDevice::allocate(size_t size)
        {
            NNDEPLOY_LOGE("Not Implemented\n");
            return nullptr;
        }

        void* OpenCLDevice::allocate(const BufferDesc &desc)
        {
            NNDEPLOY_LOGE("Not Implemented\n");
            return nullptr;
        }
        
        void OpenCLDevice::deallocate(void *ptr) 
        {
            NNDEPLOY_LOGE("Not Implemented\n");
            if (ptr == nullptr)
            {
                return;
            }
            return;
        }

        base::Status OpenCLDevice::copy(void *src, void *dst, size_t size,
                              Stream *stream) 
        {
            NNDEPLOY_LOGE("Not Implemented\n");
            return base::kStatusCodeOk;
        }

        base::Status OpenCLDevice::download(void *src, void *dst, size_t size,
                                  Stream *stream)
        {
            NNDEPLOY_LOGE("Not Implemented\n");
            return base::kStatusCodeOk;
        }

        base::Status OpenCLDevice::upload(void *src, void *dst, size_t size,
                                Stream *stream) 
        {
            NNDEPLOY_LOGE("Not Implemented\n");
            return base::kStatusCodeOk;
        }

        base::Status OpenCLDevice::copy(Buffer *src, Buffer *dst, Stream *stream)
        {
            NNDEPLOY_LOGE("Not Implemented\n");
            return base::kStatusCodeOk;
        }

        base::Status OpenCLDevice::download(Buffer *src, Buffer *dst, Stream *stream)
        {
            NNDEPLOY_LOGE("Not Implemented\n");
            return base::kStatusCodeOk;
        }

        base::Status OpenCLDevice::upload(Buffer *src, Buffer *dst, Stream *stream)
        {
            NNDEPLOY_LOGE("Not Implemented\n");
            return base::kStatusCodeOk;
        }

        void* OpenCLDevice::getContext() 
        { 
            NNDEPLOY_LOGE("Not Implemented\n");
            return nullptr; 
        }
        
        Stream* OpenCLDevice::createStream() 
        { 
            NNDEPLOY_LOGE("Not Implemented\n");
            return nullptr;
        }

        Stream* OpenCLDevice::createStream(void *stream) 
        {
            NNDEPLOY_LOGE("Not Implemented\n");
            return nullptr;
        }

        base::Status OpenCLDevice::destroyStream(Stream *stream) 
        {
            NNDEPLOY_LOGE("Not Implemented\n");
            if (stream == nullptr) 
            {
                NNDEPLOY_LOGE("stream is nullptr\n");
                return base::kStatusCodeOk;
            }
            return base::kStatusCodeOk;
        }

        Event* OpenCLDevice::createEvent() 
        { 
            NNDEPLOY_LOGE("Not Implemented\n");
            return nullptr; 
        }

        base::Status OpenCLDevice::destroyEvent(Event *event) 
        {
            NNDEPLOY_LOGE("Not Implemented\n");
            if (event == nullptr) {
                NNDEPLOY_LOGE("event is nullptr\n");
                return base::kStatusCodeErrorDeviceOpenCL;
            }
            return base::kStatusCodeOk;
        }

        base::Status OpenCLDevice::createEvents(Event **events, size_t count) 
        {
            NNDEPLOY_LOGE("Not Implemented\n");
            return base::kStatusCodeOk;
        }

        base::Status OpenCLDevice::destroyEvents(Event **events, size_t count) 
        {
            NNDEPLOY_LOGE("Not Implemented\n");
            return base::kStatusCodeOk;
        }

        base::Status OpenCLDevice::init()
        {
            if(OpenCLSymbols::GetInstance()->LoadOpenCLLibrary() == false)
            {
                NNDEPLOY_LOGE("load opencl lib failed!\n");
                return base::kStatusCodeErrorDeviceOpenCL;
            }
            return base::kStatusCodeOk;
        }

        base::Status OpenCLDevice::deinit() { return base::kStatusCodeOk; }
    } 
}