
#include "nndeploy/device/x86/x86_device.h"

#include "nndeploy/device/buffer.h"
#include "nndeploy/device/mat.h"
#include "nndeploy/device/tensor.h"

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

BufferDesc X86Device::toBufferDesc(const MatDesc& desc,
                                   const base::IntVector& config) {
  BufferDesc buffer_desc;
  buffer_desc.config_ = config;
  size_t size = desc.data_type_.size();
  if (desc.stride_.empty()) {
    for (int i = 0; i < desc.shape_.size(); ++i) {
      size *= desc.shape_[i];
    }
  } else {
    size = desc.stride_[0];
  }
  buffer_desc.size_.push_back(size);
  return buffer_desc;
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
BufferDesc X86Device::toBufferDesc(const TensorDesc& desc,
                                   const base::IntVector& config) {
  BufferDesc buffer_desc;
  buffer_desc.config_ = config;
  size_t size = desc.data_type_.size();
  if (desc.stride_.empty()) {
    for (int i = 0; i < desc.shape_.size(); ++i) {
      size *= desc.shape_[i];
    }
  } else {
    size = desc.stride_[0];
  }
  buffer_desc.size_.push_back(size);
  return buffer_desc;
}

Buffer* X86Device::allocate(size_t size) {
  BufferDesc desc(size);
  return this->allocate(desc);
}
Buffer* X86Device::allocate(const BufferDesc& desc) {
  void* data = malloc(desc.size_[0]);
  Buffer* buffer = Device::create(desc, data, kBufferSourceTypeAllocate);
  return buffer;
}
void X86Device::deallocate(Buffer* buffer) {
  if (buffer == nullptr) {
    return;
  }
  if (buffer->subRef() > 1) {
    return;
  }
  BufferSourceType buffer_source_type = buffer->getBufferSourceType();
  if (buffer_source_type == kBufferSourceTypeNone ||
      buffer_source_type == kBufferSourceTypeExternal) {
    Device::destory(buffer);
  } else if (buffer_source_type == kBufferSourceTypeAllocate) {
    if (buffer->getPtr() != nullptr) {
      void* data = buffer->getPtr();
      free(data);
    }
    Device::destory(buffer);
  } else if (buffer_source_type == kBufferSourceTypeMapped) {
    return;
  } else {
    return;
  }
}

base::Status X86Device::copy(Buffer* src, Buffer* dst) {
  BufferDescCompareStatus status =
      compareBufferDesc(dst->getDesc(), src->getDesc());
  if (status >= kBufferDescCompareStatusConfigEqualSizeEqual) {
    memcpy(dst->getPtr(), src->getPtr(), src->getDesc().size_[0]);
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("copy buffer failed");
    return base::kStatusCodeErrorOutOfMemory;
  }
}
base::Status X86Device::download(Buffer* src, Buffer* dst) {
  BufferDescCompareStatus status =
      compareBufferDesc(dst->getDesc(), src->getDesc());
  if (status >= kBufferDescCompareStatusConfigEqualSizeEqual) {
    memcpy(dst->getPtr(), src->getPtr(), src->getDesc().size_[0]);
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("download buffer failed");
    return base::kStatusCodeErrorOutOfMemory;
  }
}
base::Status X86Device::upload(Buffer* src, Buffer* dst) {
  BufferDescCompareStatus status =
      compareBufferDesc(dst->getDesc(), src->getDesc());
  if (status >= kBufferDescCompareStatusConfigEqualSizeEqual) {
    memcpy(dst->getPtr(), src->getPtr(), src->getDesc().size_[0]);
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("upload buffer failed");
    return base::kStatusCodeErrorOutOfMemory;
  }
}

base::Status X86Device::init() { return base::kStatusCodeOk; }
base::Status X86Device::deinit() { return base::kStatusCodeOk; }

}  // namespace device
}  // namespace nndeploy