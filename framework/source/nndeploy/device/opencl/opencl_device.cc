#include "nndeploy/device/opencl/opencl_device.h"

#include <cstddef>
#include <iostream>
#include <vector>

#include "CL/cl.h"
#include "cl.hpp"
#include "nndeploy/base/common.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/opencl/opencl_util.h"

namespace nndeploy {
namespace device {
TypeArchitectureRegister<OpenCLArchitecture> opencl_architecture_register(
    base::kDeviceTypeCodeOpenCL);

OpenCLArchitecture::OpenCLArchitecture(base::DeviceTypeCode device_type_code)
    : Architecture(device_type_code) {};

OpenCLArchitecture::~OpenCLArchitecture() {};

base::Status OpenCLArchitecture::checkDevice(int device_id,
                                             std::string library_path) {
  int device_count = clGetNumDevices();
  if (device_id > -1 && device_id < device_count) {
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("device id is invalid, device id: %d, device count: %d\n",
                  device_id, device_count);
    return base::kStatusCodeErrorDeviceOpenCL;
  }
}

base::Status OpenCLArchitecture::enableDevice(int device_id,
                                              std::string library_path) {
  base::DeviceType device_type(base::kDeviceTypeCodeOpenCL, device_id);
  std::lock_guard<std::mutex> lock(mutex_);
  if (devices_.find(device_id) == devices_.end()) {
    OpenCLDevice *device = new OpenCLDevice(device_type, library_path);
    if (device == nullptr) {
      NNDEPLOY_LOGE("device is nullptr\n");
      return base::kStatusCodeErrorOutOfMemory;
    }
    if (device->init() != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("device init failed\n");
      return base::kStatusCodeErrorDeviceOpenCL;
    }
    devices_.insert({device_id, device});
  }
  return base::kStatusCodeOk;
}

Device *OpenCLArchitecture::getDevice(int device_id) {
  Device *device = nullptr;
  if (devices_.find(device_id) != devices_.end()) {
    return devices_[device_id];
  } else {
    base::Status status = this->enableDevice(device_id, "");
    if (status == base::kStatusCodeOk) {
      device = devices_[device_id];
    } else {
      NNDEPLOY_LOGE("enable device failed\n");
    }
  }
  return device;
}

std::vector<DeviceInfo> OpenCLArchitecture::getDeviceInfo(
    std::string library_path) {
  std::vector<DeviceInfo> device_info_list;
  device_info_list.resize(clGetNumDevices());
  for (size_t i = 0; i < device_info_list.size(); ++i) {
    device_info_list[i].device_type_ =
        base::DeviceType(base::kDeviceTypeCodeOpenCL, i);
  }
  return device_info_list;
}

/* OpenCLDevice */
BufferDesc OpenCLDevice::toBufferDesc(const TensorDesc &desc,
                                      const base::IntVector &config) {
  NNDEPLOY_LOGE("Not Implemented\n");
  size_t size = 0;
  return BufferDesc(size, config);
}

void *OpenCLDevice::allocate(size_t size) {
  NNDEPLOY_LOGE("Not Implemented\n");
  return nullptr;
}

void *OpenCLDevice::allocate(const BufferDesc &desc) {
  NNDEPLOY_LOGE("Not Implemented\n");
  return nullptr;
}

void OpenCLDevice::deallocate(void *ptr) {
  NNDEPLOY_LOGE("Not Implemented\n");
  if (ptr == nullptr) {
    return;
  }
  return;
}

base::Status OpenCLDevice::copy(void *src, void *dst, size_t size,
                                Stream *stream) {
  NNDEPLOY_LOGE("Not Implemented\n");
  return base::kStatusCodeOk;
}

base::Status OpenCLDevice::download(void *src, void *dst, size_t size,
                                    Stream *stream) {
  NNDEPLOY_LOGE("Not Implemented\n");
  return base::kStatusCodeOk;
}

base::Status OpenCLDevice::upload(void *src, void *dst, size_t size,
                                  Stream *stream) {
  NNDEPLOY_LOGE("Not Implemented\n");
  return base::kStatusCodeOk;
}

base::Status OpenCLDevice::copy(Buffer *src, Buffer *dst, Stream *stream) {
  NNDEPLOY_LOGE("Not Implemented\n");
  return base::kStatusCodeOk;
}

base::Status OpenCLDevice::download(Buffer *src, Buffer *dst, Stream *stream) {
  NNDEPLOY_LOGE("Not Implemented\n");
  return base::kStatusCodeOk;
}

base::Status OpenCLDevice::upload(Buffer *src, Buffer *dst, Stream *stream) {
  NNDEPLOY_LOGE("Not Implemented\n");
  return base::kStatusCodeOk;
}

void *OpenCLDevice::getContext() { return &context_; }

Stream *OpenCLDevice::createStream() {
  NNDEPLOY_LOGE("Not Implemented\n");
  return nullptr;
}

Stream *OpenCLDevice::createStream(void *stream) {
  NNDEPLOY_LOGE("Not Implemented\n");
  return nullptr;
}

base::Status OpenCLDevice::destroyStream(Stream *stream) {
  NNDEPLOY_LOGE("Not Implemented\n");
  if (stream == nullptr) {
    NNDEPLOY_LOGE("stream is nullptr\n");
    return base::kStatusCodeOk;
  }
  return base::kStatusCodeOk;
}

Event *OpenCLDevice::createEvent() {
  NNDEPLOY_LOGE("Not Implemented\n");
  return nullptr;
}

base::Status OpenCLDevice::destroyEvent(Event *event) {
  NNDEPLOY_LOGE("Not Implemented\n");
  if (event == nullptr) {
    NNDEPLOY_LOGE("event is nullptr\n");
    return base::kStatusCodeErrorDeviceOpenCL;
  }
  return base::kStatusCodeOk;
}

base::Status OpenCLDevice::createEvents(Event **events, size_t count) {
  NNDEPLOY_LOGE("Not Implemented\n");
  return base::kStatusCodeOk;
}

base::Status OpenCLDevice::destroyEvents(Event **events, size_t count) {
  NNDEPLOY_LOGE("Not Implemented\n");
  return base::kStatusCodeOk;
}

base::Status OpenCLDevice::init() {
  if (OpenCLSymbols::GetInstance()->LoadOpenCLLibrary() == false) {
    NNDEPLOY_LOGE("load opencl lib failed!\n");
    return base::kStatusCodeErrorDeviceOpenCL;
  }
  NNDEPLOY_LOGI("opencl loaded successfully!\n");
  std::vector<cl::Platform> platforms;
  NNDEPLOY_OPENCL_CHECK(cl::Platform::get(&platforms));
  if (platforms.size() <= 0) {
    return base::kStatusCodeErrorDeviceOpenCL;
  }
  std::vector<cl::Device> gpu_devices;
  NNDEPLOY_OPENCL_CHECK(
      platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &gpu_devices));
  if (gpu_devices.size() <= 0) {
    return base::kStatusCodeErrorDeviceOpenCL;
  }
  cl_int err;
  context_ = cl::Context(gpu_devices, nullptr, nullptr, nullptr, &err);
  NNDEPLOY_OPENCL_CHECK(err);
  return base::kStatusCodeOk;
}

base::Status OpenCLDevice::deinit() {
  if (OpenCLSymbols::GetInstance()->UnLoadOpenCLLibrary() == false) {
    NNDEPLOY_LOGE("unload opencl lib failed!\n");
    return base::kStatusCodeErrorDeviceOpenCL;
  }
  NNDEPLOY_LOGI("opencl unloaded successfully!\n");
  return base::kStatusCodeOk;
}

OpenCLDevice::~OpenCLDevice() { OpenCLDevice::deinit(); }

/* OpenCLStream */
OpenCLStream::OpenCLStream(Device *device) : Stream(device) {
  auto opencl_device = static_cast<OpenCLDevice *>(device);
  cl::Context *context =
      static_cast<cl::Context *>(opencl_device->getContext());
  auto devices = context->getInfo<CL_CONTEXT_DEVICES>();
  for (auto &device : devices) {
    std::cout << device.getInfo<CL_DEVICE_NAME>() << '\n';
  }
  // cl::CommandQueue stream(opencl_device->getContext(), );
}

/* UNIMPLEMENTED */
OpenCLStream::OpenCLStream(Device *device, void *stream)
    : Stream(device, stream) {}
base::Status OpenCLStream::synchronize() { return base::kStatusCodeOk; }
base::Status OpenCLStream::recordEvent(Event *event) {
  return base::kStatusCodeOk;
}
base::Status OpenCLStream::waitEvent(Event *event) {
  return base::kStatusCodeOk;
}
void *OpenCLStream::getNativeStream() { return nullptr; }
cl::CommandQueue OpenCLStream::getStream() { return cl::CommandQueue(); }

OpenCLStream::~OpenCLStream() {}

}  // namespace device
}  // namespace nndeploy