
#include "nndeploy/source/device/x86/x86_architecture.h"

#include "nndeploy/source/device/x86/x86_device.h"

namespace nndeploy {
namespace device {

TypeArchitectureRegister<X86Architecture> x86_architecture_register(
    base::kDeviceTypeCodeX86);

X86Architecture::X86Architecture(base::DeviceTypeCode device_type_code)
    : Architecture(device_type_code){};

X86Architecture::~X86Architecture() {
  for (auto iter : devices_) {
    X86Device* tmp_device = dynamic_cast<X86Device*>(iter.second);
    if (tmp_device->deinit() != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("device deinit failed");
    }
    delete tmp_device;
  }
};

base::Status X86Architecture::checkDevice(int32_t device_id,
                                          void* command_queue,
                                          std::string library_path) {
  return base::kStatusCodeOk;
}

base::Status X86Architecture::enableDevice(int32_t device_id,
                                           void* command_queue,
                                           std::string library_path) {
  device_id = 0;
  std::lock_guard<std::mutex> lock(mutex_);
  if (devices_.find(device_id) == devices_.end()) {
    base::DeviceType device_type(base::kDeviceTypeCodeX86, device_id);
    X86Device* device = new X86Device(device_type, command_queue, library_path);
    if (device == NULL) {
      NNDEPLOY_LOGE("device is NULL");
      return base::kStatusCodeErrorOutOfMemory;
    }
    if (device->init() != base::kStatusCodeOk) {
      delete device;
      NNDEPLOY_LOGE("device init failed");
      return base::kStatusCodeErrorDeviceX86;
    } else {
      devices_.insert({device_id, device});
      return base::kStatusCodeOk;
    }
  }

  return base::kStatusCodeOk;
}

Device* X86Architecture::getDevice(int32_t device_id) {
  device_id = 0;
  Device* device = nullptr;
  if (devices_.find(device_id) != devices_.end()) {
    device = devices_[device_id];
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

std::vector<DeviceInfo> X86Architecture::getDeviceInfo(
    std::string library_path) {
  std::vector<DeviceInfo> device_info_list;
  return device_info_list;
}

}  // namespace device
}  // namespace nndeploy
