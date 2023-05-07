
#include "nndeploy/source/device/cpu/cpu_architecture.h"

#include "nndeploy/source/device/cpu/cpu_device.h"

namespace nndeploy {
namespace device {

TypeArchitectureRegister<CpuArchitecture> cpu_architecture_register(
    base::kDeviceTypeCodeCpu);

CpuArchitecture::CpuArchitecture(base::DeviceTypeCode device_type_code)
    : Architecture(device_type_code){};

CpuArchitecture::~CpuArchitecture() {
  for (auto iter : devices_) {
    CpuDevice* tmp_device = dynamic_cast<CpuDevice*>(iter.second);
    if (tmp_device->deinit() != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("device deinit failed");
    }
    delete tmp_device;
  }
};

base::Status CpuArchitecture::checkDevice(int32_t device_id,
                                          void* command_queue,
                                          std::string library_path) {
  return base::kStatusCodeOk;
}

base::Status CpuArchitecture::enableDevice(int32_t device_id,
                                           void* command_queue,
                                           std::string library_path) {
  device_id = 0;
  base::DeviceType device_type(base::kDeviceTypeCodeCpu, device_id);
  CpuDevice* device = new CpuDevice(device_type, command_queue, library_path);
  if (device == NULL) {
    NNDEPLOY_LOGE("device is NULL");
    return base::kStatusCodeErrorOutOfMemory;
  }
  if (device->init() != base::kStatusCodeOk) {
    delete device;
    NNDEPLOY_LOGE("device init failed");
    return base::kStatusCodeErrorDeviceCpu;
  } else {
    devices_.insert({device_id, device});
    return base::kStatusCodeOk;
  }

  return base::kStatusCodeOk;
}

Device* CpuArchitecture::getDevice(int32_t device_id) {
  device_id = 0;
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

std::vector<DeviceInfo> CpuArchitecture::getDeviceInfo(
    std::string library_path) {
  std::vector<DeviceInfo> device_info_list;
  return device_info_list;
}

}  // namespace device
}  // namespace nndeploy
