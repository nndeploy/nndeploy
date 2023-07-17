
#include "nndeploy/include/device/cpu/cpu_device.h"

#include "nndeploy/include/device/buffer.h"
#include "nndeploy/include/device/mat.h"
#include "nndeploy/include/device/tensor.h"

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
  std::lock_guard<std::mutex> lock(mutex_);
  if (devices_.find(device_id) == devices_.end()) {
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

BufferDesc CpuDevice::toBufferDesc(const MatDesc& desc,
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
BufferDesc CpuDevice::toBufferDesc(const TensorDesc& desc,
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

Buffer* CpuDevice::allocate(size_t size) {
  BufferDesc desc(size);
  return this->allocate(desc);
}
Buffer* CpuDevice::allocate(const BufferDesc& desc) {
  void* data = malloc(desc.size_[0]);
  Buffer* buffer = Device::create(desc, data, kBufferSourceTypeAllocate);
  return buffer;
}
void CpuDevice::deallocate(Buffer* buffer) {
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

base::Status CpuDevice::copy(Buffer* src, Buffer* dst) {
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
base::Status CpuDevice::download(Buffer* src, Buffer* dst) {
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
base::Status CpuDevice::upload(Buffer* src, Buffer* dst) {
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

base::Status CpuDevice::init() { return base::kStatusCodeOk; }
base::Status CpuDevice::deinit() { return base::kStatusCodeOk; }

}  // namespace device
}  // namespace nndeploy