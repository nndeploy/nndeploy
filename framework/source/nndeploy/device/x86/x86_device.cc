#include "nndeploy/device/x86/x86_device.h"

#include "nndeploy/device/buffer.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace device {

TypeArchitectureRegister<X86Architecture> x86_architecture_register(
    base::kDeviceTypeCodeX86);

X86Architecture::X86Architecture(base::DeviceTypeCode device_type_code)
    : Architecture(device_type_code) {};

X86Architecture::~X86Architecture() {};

base::Status X86Architecture::checkDevice(int device_id,
                                          std::string library_path) {
  return base::kStatusCodeOk;
}

base::Status X86Architecture::enableDevice(int device_id,
                                           std::string library_path) {
  // device_id = 0;
  std::lock_guard<std::mutex> lock(mutex_);
  if (devices_.find(device_id) == devices_.end()) {
    base::DeviceType device_type(base::kDeviceTypeCodeX86, device_id);
    X86Device *device = new X86Device(device_type, library_path);
    if (device == nullptr) {
      NNDEPLOY_LOGE("device is nullptr\n");
      return base::kStatusCodeErrorOutOfMemory;
    }
    if (device->init() != base::kStatusCodeOk) {
      delete device;
      NNDEPLOY_LOGE("device init failed\n");
      return base::kStatusCodeErrorDeviceX86;
    } else {
      devices_.insert({device_id, device});
      return base::kStatusCodeOk;
    }
  }

  return base::kStatusCodeOk;
}

Device *X86Architecture::getDevice(int device_id) {
  // device_id = 0;
  Device *device = nullptr;
  if (devices_.find(device_id) != devices_.end()) {
    device = devices_[device_id];
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

std::vector<DeviceInfo> X86Architecture::getDeviceInfo(
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
BufferDesc X86Device::toBufferDesc(const TensorDesc &desc,
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

void *X86Device::allocate(size_t size) {
  void *data = malloc(size);
  if (data == nullptr) {
    NNDEPLOY_LOGE("allocate buffer failed\n");
    return nullptr;
  }
  return data;
}
void *X86Device::allocate(const BufferDesc &desc) {
  void *data = malloc(desc.getRealSize());
  if (data == nullptr) {
    NNDEPLOY_LOGE("allocate buffer failed\n");
    return nullptr;
  }
  return data;
}
void X86Device::deallocate(void *ptr) {
  if (ptr == nullptr) {
    return;
  }
  free(ptr);
}

base::Status X86Device::copy(void *src, void *dst, size_t size,
                             Stream *stream) {
  if (src != nullptr && dst != nullptr) {
    memcpy(dst, src, size);
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("copy buffer failed\n");
    return base::kStatusCodeErrorOutOfMemory;
  }
}

base::Status X86Device::download(void *src, void *dst, size_t size,
                                 Stream *stream) {
  if (src != nullptr && dst != nullptr) {
    memcpy(dst, src, size);
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("download buffer failed\n");
    return base::kStatusCodeErrorOutOfMemory;
  }
}

base::Status X86Device::upload(void *src, void *dst, size_t size,
                               Stream *stream) {
  if (src != nullptr && dst != nullptr) {
    memcpy(dst, src, size);
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("upload buffer failed\n");
    return base::kStatusCodeErrorOutOfMemory;
  }
}

base::Status X86Device::copy(Buffer *src, Buffer *dst, Stream *stream) {
  size_t dst_size = dst->getSize();
  size_t src_size = src->getSize();
  size_t size = std::min(dst_size, src_size);
  if (src != nullptr && dst != nullptr) {
    memcpy(dst->getData(), src->getData(), size);
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("copy buffer failed\n");
    return base::kStatusCodeErrorOutOfMemory;
  }
}

base::Status X86Device::download(Buffer *src, Buffer *dst, Stream *stream) {
  size_t dst_size = dst->getSize();
  size_t src_size = src->getSize();
  size_t size = std::min(dst_size, src_size);
  if (src != nullptr && dst != nullptr) {
    memcpy(dst->getData(), src->getData(), size);
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("download buffer failed\n");
    return base::kStatusCodeErrorOutOfMemory;
  }
}

base::Status X86Device::upload(Buffer *src, Buffer *dst, Stream *stream) {
  size_t dst_size = dst->getSize();
  size_t src_size = src->getSize();
  size_t size = std::min(dst_size, src_size);
  if (src != nullptr && dst != nullptr) {
    memcpy(dst->getData(), src->getData(), size);
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("upload buffer failed\n");
    return base::kStatusCodeErrorOutOfMemory;
  }
}

base::Status X86Device::init() { return base::kStatusCodeOk; }
base::Status X86Device::deinit() { return base::kStatusCodeOk; }

}  // namespace device
}  // namespace nndeploy