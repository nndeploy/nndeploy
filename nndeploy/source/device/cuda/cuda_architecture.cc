
#include "nndeploy/source/device/cuda/cuda_architecture.h"

#include "nndeploy/source/device/cuda/cuda_device.h"

namespace nndeploy {
namespace device {

TypeArchitectureRegister<CudaArchitecture> cuda_architecture_register(
    base::kDeviceTypeCodeCuda);

CudaArchitecture::CudaArchitecture(base::DeviceTypeCode device_type_code)
    : Architecture(device_type_code){};

CudaArchitecture::~CudaArchitecture() {
  for (auto iter : devices_) {
    CudaDevice* tmp_device = dynamic_cast<CudaDevice*>(iter.second);
    if (tmp_device->deinit() != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("device deinit failed");
    }
    delete tmp_device;
  }
};

base::Status CudaArchitecture::checkDevice(int32_t device_id,
                                           void* command_queue,
                                           std::string library_path) {
  int32_t device_count = cudaGetNumDevices();
  if (device_id > 0 && device_id < device_count) {
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("device id is invalid, device id: %d, device count: %d",
                  device_id, device_count);
    return base::kStatusCodeErrorDeviceCuda;
  }
}

base::Status CudaArchitecture::enableDevice(int32_t device_id,
                                            void* command_queue,
                                            std::string library_path) {
  base::DeviceType device_type(base::kDeviceTypeCodeCuda, device_id);
  CudaDevice* device = new CudaDevice(device_type, command_queue, library_path);
  if (device == NULL) {
    NNDEPLOY_LOGE("device is NULL");
    return base::kStatusCodeErrorOutOfMemory;
  }

  if (device->init() != base::kStatusCodeOk) {
    delete device;
    NNDEPLOY_LOGE("device init failed");
    return base::kStatusCodeErrorDeviceCuda;
  } else {
    devices_.insert({device_id, device});
    return base::kStatusCodeOk;
  }

  return base::kStatusCodeOk;
}

Device* CudaArchitecture::getDevice(int32_t device_id) {
  Device* device = nullptr;
  if (devices_.find(device_id) != devices_.end()) {
    return devices_[device_id];
  } else {
    base::Status status = this->enableDevice(device_id, nullptr, "");
    if (status == base::kStatusCodeOk) {
      device = devices_[device_id];
    } else {
      NNDEPLOY_LOGE("enable device failed");
    }
  }
  return device;
}

std::vector<DeviceInfo> CudaArchitecture::getDeviceInfo(
    std::string library_path) {
  std::vector<DeviceInfo> device_info_list;
  int32_t device_count = cudaGetNumDevices();
  for (int i = 0; i < device_count; ++i) {
    cudaDeviceProp p = cudaGetDeviceProperty(i);
    DeviceInfo device_info;
    device_info_list.push_back(device_info);
  }
  return device_info_list;
}

}  // namespace device
}  // namespace nndeploy
