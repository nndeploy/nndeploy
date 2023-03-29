
#include "nndeploy/source/device/x86/x86_architecture.h"

#include "nndeploy/source/device/x86/x86_device.h"

namespace nndeploy {
namespace device {

TypeArchitectureRegister<X86Architecture> x86_architecture_register(
    base::kDeviceTypeCodeX86);

X86Architecture::X86Architecture(base::DeviceTypeCode device_type_code)
    : Architecture(device_type_code){};

X86Architecture::~X86Architecture(){};

base::Status X86Architecture::checkDevice(int32_t device_id,
                                          void* command_queue,
                                          std::string library_path) {
  return base::kStatusCodeOk;
}

Device* X86Architecture::createDevice(int32_t device_id, void* command_queue,
                                      std::string library_path) {
  X86Device* device = new X86Device(device_id, command_queue, library_path);
  if (device == NULL) {
    // TODO: log
    return NULL;
  }

  if (device->init() != base::kStatusCodeOk) {
    delete device;
    return NULL;
  } else {
    return dynamic_cast<Device*>(device);
  }
}

base::Status X86Architecture::destoryDevice(Device* device) {
  if (device == NULL) {
    // TODO: log
    return base::kStatusCodeErrorNullParam;
  }

  X86Device* tmp_device = dynamic_cast<X86Device*>(device);

  base::Status status = base::kStatusCodeOk;
  if (tmp_device->deinit() != base::kStatusCodeOk) {
    // TODO: log
    status = base::kStatusCodeErrorDeviceX86;
  }
  delete tmp_device;

  return status;
}

std::vector<DeviceInfo> X86Architecture::getDeviceInfo(
    std::string library_path) {
  std::vector<DeviceInfo> device_info_list;
  return device_info_list;
}

}  // namespace device
}  // namespace nndeploy
