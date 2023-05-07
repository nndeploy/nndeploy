
#include "nndeploy/source/device/arm/arm_architecture.h"

#include "nndeploy/source/device/arm/arm_device.h"

namespace nndeploy {
namespace device {

TypeArchitectureRegister<ArmArchitecture> arm_architecture_register(
    base::kDeviceTypeCodeArm);

ArmArchitecture::ArmArchitecture(base::DeviceTypeCode device_type_code)
    : Architecture(device_type_code){};

ArmArchitecture::~ArmArchitecture() {
  for (auto iter : devices_) {
    ArmDevice* tmp_device = dynamic_cast<ArmDevice*>(iter.second);
    if (tmp_device->deinit() != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("device deinit failed");
    }
    delete tmp_device;
  }
};

base::Status ArmArchitecture::checkDevice(int32_t device_id,
                                          void* command_queue,
                                          std::string library_path) {
  return base::kStatusCodeOk;
}

base::Status ArmArchitecture::enableDevice(int32_t device_id,
                                           void* command_queue,
                                           std::string library_path) {
  device_id = 0;
  base::DeviceType device_type(base::kDeviceTypeCodeArm, device_id);
  ArmDevice* device = new ArmDevice(device_type, command_queue, library_path);
  if (device == NULL) {
    NNDEPLOY_LOGE("device is NULL");
    return base::kStatusCodeErrorOutOfMemory;
  }
  if (device->init() != base::kStatusCodeOk) {
    delete device;
    NNDEPLOY_LOGE("device init failed");
    return base::kStatusCodeErrorDeviceArm;
  } else {
    devices_.insert({device_id, device});
    return base::kStatusCodeOk;
  }

  return base::kStatusCodeOk;
}

Device* ArmArchitecture::getDevice(int32_t device_id) {
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

std::vector<DeviceInfo> ArmArchitecture::getDeviceInfo(
    std::string library_path) {
  std::vector<DeviceInfo> device_info_list;
  return device_info_list;
}

}  // namespace device
}  // namespace nndeploy
