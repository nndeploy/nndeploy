
#include "nndeploy/source/device/arm/arm_architecture.h"
#include "nndeploy/source/device/arm/arm_device.h"

namespace nndeploy {
namespace device {

TypeArchitectureRegister<ArmArchitecture> arm_architecture_register(
    base::kDeviceTypeCodeArm);

ArmArchitecture::ArmArchitecture(base::DeviceTypeCode device_type_code)
    : Architecture(device_type_code){};

ArmArchitecture::~ArmArchitecture(){};

base::Status ArmArchitecture::checkDevice(int32_t device_id,
                                          void* command_queue,
                                          std::string library_path) {
  return base::kStatusCodeOk;
}

Device* ArmArchitecture::createDevice(int32_t device_id, void* command_queue,
                                      std::string library_path) {
  ArmDevice* device = new ArmDevice(device_id, command_queue, library_path);
  if (device == NULL) {
    NNDEPLOY_LOGE("device is NULL");
    return NULL;
  }

  if (device->init() != base::kStatusCodeOk) {
    delete device;
    return NULL;
  } else {
    return dynamic_cast<Device*>(device);
  }
}

base::Status ArmArchitecture::destoryDevice(Device* device) {
  if (device == NULL) {
    NNDEPLOY_LOGE("device is NULL");
    return base::kStatusCodeErrorNullParam;
  }

  ArmDevice* tmp_device = dynamic_cast<ArmDevice*>(device);

  base::Status status = base::kStatusCodeOk;
  if (tmp_device->deinit() != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("device deinit failed");
    status = base::kStatusCodeErrorDeviceArm;
  }
  delete tmp_device;

  return status;
}

std::vector<DeviceInfo> ArmArchitecture::getDeviceInfo(
    std::string library_path) {
  std::vector<DeviceInfo> device_info_list;
  return device_info_list;
}

}  // namespace device
}  // namespace nndeploy
