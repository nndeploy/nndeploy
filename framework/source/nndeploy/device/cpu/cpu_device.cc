
#include "nndeploy/device/cpu/cpu_device.h"

#include "nndeploy/device/buffer.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace device {

TypeArchitectureRegister<CpuArchitecture> cpu_architecture_register(
    base::kDeviceTypeCodeCpu);

CpuArchitecture::CpuArchitecture(base::DeviceTypeCode device_type_code)
    : Architecture(device_type_code) {};

CpuArchitecture::~CpuArchitecture() {
  for (auto iter : devices_) {
    CpuDevice *tmp_device = dynamic_cast<CpuDevice *>(iter.second);
    if (tmp_device->deinit() != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("device deinit failed");
    }
    delete tmp_device;
  }
};

base::Status CpuArchitecture::checkDevice(int device_id, void *command_queue,
                                          std::string library_path) {
  return base::kStatusCodeOk;
}

base::Status CpuArchitecture::enableDevice(int device_id, void *command_queue,
                                           std::string library_path) {
  device_id = 0;
  std::lock_guard<std::mutex> lock(mutex_);
  if (devices_.find(device_id) == devices_.end()) {
    base::DeviceType device_type(base::kDeviceTypeCodeCpu, device_id);
    CpuDevice *device = new CpuDevice(device_type, command_queue, library_path);
    if (device == nullptr) {
      NNDEPLOY_LOGE("device is nullptr");
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
  }

  return base::kStatusCodeOk;
}

Device *CpuArchitecture::getDevice(int device_id) {
  device_id = 0;
  Device *device = nullptr;
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

/**
 * @brief
 *
 * @param desc
 * @param config
 * @return BufferDesc
 * @note:
 * 通过stride_替代了data_format_，stride_的第一个元素表示的是整个tensor的大小
 * 意味着在TensorDesc的构造函数要花很多心思来计算stride_
 */
BufferDesc CpuDevice::toBufferDesc(const TensorDesc &desc,
                                   const base::IntVector &config) {
  size_t size = desc.data_type_.size();
  if (desc.stride_.empty()) {
    for (int i = 0; i < desc.shape_.size(); ++i) {
      size *= desc.shape_[i];
    }
  } else {
    size = desc.stride_[0];
  }
  return BufferDesc(size, config);
}

void *CpuDevice::allocate(size_t size) {
  void *data = malloc(size);
  if (data == nullptr) {
    NNDEPLOY_LOGE("allocate buffer failed");
    return nullptr;
  }
  return data;
}
void *CpuDevice::allocate(const BufferDesc &desc) {
  void *data = malloc(desc.getRealSize());
  if (data == nullptr) {
    NNDEPLOY_LOGE("allocate buffer failed");
    return nullptr;
  }
  return data;
}
void CpuDevice::deallocate(void *ptr) {
  if (ptr == nullptr) {
    return;
  }
  free(ptr);
}

base::Status CpuDevice::copy(void *src, void *dst, size_t size, int index) {
  if (src != nullptr && dst != nullptr) {
    memcpy(dst, src, size);
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("copy buffer failed");
    return base::kStatusCodeErrorOutOfMemory;
  }
}
base::Status CpuDevice::download(void *src, void *dst, size_t size, int index) {
  if (src != nullptr && dst != nullptr) {
    memcpy(dst, src, size);
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("copy buffer failed");
    return base::kStatusCodeErrorOutOfMemory;
  }
}
base::Status CpuDevice::upload(void *src, void *dst, size_t size, int index) {
  if (src != nullptr && dst != nullptr) {
    memcpy(dst, src, size);
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("copy buffer failed");
    return base::kStatusCodeErrorOutOfMemory;
  }
}

base::Status CpuDevice::copy(Buffer *src, Buffer *dst, int index) {
  if (src != nullptr && dst != nullptr && dst->getDesc() >= src->getDesc()) {
    memcpy(dst->getData(), src->getData(), src->getSize());
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("copy buffer failed");
    return base::kStatusCodeErrorOutOfMemory;
  }
}
base::Status CpuDevice::download(Buffer *src, Buffer *dst, int index) {
  if (src != nullptr && dst != nullptr && dst->getDesc() >= src->getDesc()) {
    memcpy(dst->getData(), src->getData(), src->getSize());
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("download buffer failed");
    return base::kStatusCodeErrorOutOfMemory;
  }
}
base::Status CpuDevice::upload(Buffer *src, Buffer *dst, int index) {
  if (src != nullptr && dst != nullptr && dst->getDesc() >= src->getDesc()) {
    memcpy(dst->getData(), src->getData(), src->getSize());
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("upload buffer failed");
    return base::kStatusCodeErrorOutOfMemory;
  }
}

base::Status CpuDevice::init() { return base::kStatusCodeOk; }
base::Status CpuDevice::deinit() { return base::kStatusCodeOk; }

}  // namespace device
}  // namespace nndeploy