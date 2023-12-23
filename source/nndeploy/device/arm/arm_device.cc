
#include "nndeploy/device/arm/arm_device.h"

#include "nndeploy/device/buffer.h"
#include "nndeploy/device/mat.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace device {

TypeArchitectureRegister<ArmArchitecture> arm_architecture_register(
    base::kDeviceTypeCodeArm);

ArmArchitecture::ArmArchitecture(base::DeviceTypeCode device_type_code)
    : Architecture(device_type_code){};

ArmArchitecture::~ArmArchitecture() {
  for (auto iter : devices_) {
    ArmDevice *tmp_device = dynamic_cast<ArmDevice *>(iter.second);
    if (tmp_device->deinit() != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("device deinit failed");
    }
    delete tmp_device;
  }
};

base::Status ArmArchitecture::checkDevice(int device_id, void *command_queue,
                                          std::string library_path) {
  return base::kStatusCodeOk;
}

base::Status ArmArchitecture::enableDevice(int device_id, void *command_queue,
                                           std::string library_path) {
  device_id = 0;
  std::lock_guard<std::mutex> lock(mutex_);
  if (devices_.find(device_id) == devices_.end()) {
    base::DeviceType device_type(base::kDeviceTypeCodeArm, device_id);
    ArmDevice *device = new ArmDevice(device_type, command_queue, library_path);
    if (device == nullptr) {
      NNDEPLOY_LOGE("device is nullptr");
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
  }

  return base::kStatusCodeOk;
}

Device *ArmArchitecture::getDevice(int device_id) {
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

std::vector<DeviceInfo> ArmArchitecture::getDeviceInfo(
    std::string library_path) {
  std::vector<DeviceInfo> device_info_list;
  return device_info_list;
}

BufferDesc ArmDevice::toBufferDesc(const MatDesc &desc,
                                   const base::IntVector &config) {
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
  buffer_desc.size_.emplace_back(size);
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
BufferDesc ArmDevice::toBufferDesc(const TensorDesc &desc,
                                   const base::IntVector &config) {
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
  buffer_desc.size_.emplace_back(size);
  return buffer_desc;
}

Buffer *ArmDevice::allocate(size_t size) {
  BufferDesc desc(size);
  return this->allocate(desc);
}
Buffer *ArmDevice::allocate(const BufferDesc &desc) {
  void *data = malloc(desc.size_[0]);
  Buffer *buffer = Device::create(desc, data, kBufferSourceTypeAllocate);
  return buffer;
}
void ArmDevice::deallocate(Buffer *buffer) {
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
      void *data = buffer->getPtr();
      free(data);
    }
    Device::destory(buffer);
  } else if (buffer_source_type == kBufferSourceTypeMapped) {
    return;
  } else {
    return;
  }
}

base::Status ArmDevice::copy(Buffer *src, Buffer *dst) {
  if (compareBufferDesc(dst->getDesc(), src->getDesc()) >= 0) {
    memcpy(dst->getPtr(), src->getPtr(), src->getDesc().size_[0]);
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("copy buffer failed");
    return base::kStatusCodeErrorOutOfMemory;
  }
}
base::Status ArmDevice::download(Buffer *src, Buffer *dst) {
  if (compareBufferDesc(dst->getDesc(), src->getDesc()) >= 0) {
    memcpy(dst->getPtr(), src->getPtr(), src->getDesc().size_[0]);
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("download buffer failed");
    return base::kStatusCodeErrorOutOfMemory;
  }
}
base::Status ArmDevice::upload(Buffer *src, Buffer *dst) {
  if (compareBufferDesc(dst->getDesc(), src->getDesc()) >= 0) {
    memcpy(dst->getPtr(), src->getPtr(), src->getDesc().size_[0]);
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("upload buffer failed");
    return base::kStatusCodeErrorOutOfMemory;
  }
}

base::Status ArmDevice::init() { return base::kStatusCodeOk; }
base::Status ArmDevice::deinit() { return base::kStatusCodeOk; }

}  // namespace device
}  // namespace nndeploy