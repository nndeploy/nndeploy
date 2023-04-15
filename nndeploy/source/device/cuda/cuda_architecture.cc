
#include "nndeploy/source/device/cuda/cuda_architecture.h"

#include "nndeploy/source/device/cuda/cuda_device.h"

namespace nndeploy {
namespace device {

TypeArchitectureRegister<CudaArchitecture> cuda_architecture_register(
    base::kDeviceTypeCodeCuda);

CudaArchitecture::CudaArchitecture(base::DeviceTypeCode device_type_code)
    : Architecture(device_type_code){};

CudaArchitecture::~CudaArchitecture(){};

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

Device* CudaArchitecture::createDevice(int32_t device_id, void* command_queue,
                                       std::string library_path) {
  CudaDevice* device = new CudaDevice(device_id, command_queue, library_path);
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

base::Status CudaArchitecture::destoryDevice(Device* device) {
  if (device == NULL) {
    NNDEPLOY_LOGE("device is NULL");
    return base::kStatusCodeErrorNullParam;
  }

  CudaDevice* tmp_device = dynamic_cast<CudaDevice*>(device);

  base::Status status = base::kStatusCodeOk;
  if (tmp_device->deinit() != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("device deinit failed");
    status = base::kStatusCodeErrorDeviceCuda;
  }
  delete tmp_device;

  return status;
}

std::vector<DeviceInfo> CudaArchitecture::getDeviceInfo(
    std::string library_path) {
  std::vector<DeviceInfo> device_info_list;
  int32_t device_count = cudaGetNumDevices();
  for (int i = 0; i < device_count; ++i) {
    cudaDeviceProp p = cudaGetDeviceProperties(i);
    DeviceInfo device_info;
    device_info_list.push_back(device_info);
  }
  return device_info_list;
}

}  // namespace device
}  // namespace nndeploy
