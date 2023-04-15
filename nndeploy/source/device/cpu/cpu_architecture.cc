
#include "nndeploy/source/device/cpu/cpu_architecture.h"

#include "nndeploy/source/device/cpu/cpu_device.h"

namespace nndeploy {
namespace device {

TypeArchitectureRegister<CpuArchitecture> cpu_architecture_register(
    base::kDeviceTypeCodeCpu);

CpuArchitecture::CpuArchitecture(base::DeviceTypeCode device_type_code)
    : Architecture(device_type_code){};

CpuArchitecture::~CpuArchitecture(){};

base::Status CpuArchitecture::checkDevice(int32_t device_id,
                                          void* command_queue,
                                          std::string library_path) {
  return base::kStatusCodeOk;
}

Device* CpuArchitecture::createDevice(int32_t device_id, void* command_queue,
                                      std::string library_path) {
  CpuDevice* device = new CpuDevice(device_id, command_queue, library_path);
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

base::Status CpuArchitecture::destoryDevice(Device* device) {
  if (device == NULL) {
    NNDEPLOY_LOGE("device is NULL");
    return base::kStatusCodeErrorNullParam;
  }

  CpuDevice* tmp_device = dynamic_cast<CpuDevice*>(device);

  base::Status status = base::kStatusCodeOk;
  if (tmp_device->deinit() != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("device deinit failed");
    status = base::kStatusCodeErrorDeviceCpu;
  }
  delete tmp_device;

  return status;
}

std::vector<DeviceInfo> CpuArchitecture::getDeviceInfo(
    std::string library_path) {
  std::vector<DeviceInfo> device_info_list;
  return device_info_list;
}

}  // namespace device
}  // namespace nndeploy
