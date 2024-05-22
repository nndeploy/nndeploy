
#include "nndeploy/device/ascend_cl/ascend_cl_device.h"

#include "nndeploy/device/ascend_cl/ascend_cl_util.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace device {

TypeArchitectureRegister<AscendCLArchitecture> ascend_cl_architecture_register(
    base::kDeviceTypeCodeAscendCL);

AscendCLArchitecture::AscendCLArchitecture(
    base::DeviceTypeCode device_type_code)
    : Architecture(device_type_code){};

AscendCLArchitecture::~AscendCLArchitecture() {
  for (auto iter : devices_) {
    AscendCLDevice *tmp_device = dynamic_cast<AscendCLDevice *>(iter.second);
    if (tmp_device->deinit() != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("device deinit failed");
    }
    delete tmp_device;
  }
};

base::Status AscendCLArchitecture::checkDevice(int device_id,
                                               void *command_queue,
                                               std::string library_path) {
  int device_count = ascendCLGetNumDevices();
  if (device_id > -1 && device_id < device_count) {
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("device id is invalid, device id: %d, device count: %d",
                  device_id, device_count);
    return base::kStatusCodeErrorDeviceAscendCL;
  }
}

base::Status AscendCLArchitecture::enableDevice(int device_id,
                                                void *command_queue,
                                                std::string library_path) {
  base::DeviceType device_type(base::kDeviceTypeCodeAscendCL, device_id);
  std::lock_guard<std::mutex> lock(mutex_);
  if (devices_.find(device_id) == devices_.end()) {
    AscendCLDevice *device =
        new AscendCLDevice(device_type, command_queue, library_path);
    if (device == nullptr) {
      NNDEPLOY_LOGE("device is nullptr");
      return base::kStatusCodeErrorOutOfMemory;
    }

    if (device->init() != base::kStatusCodeOk) {
      delete device;
      NNDEPLOY_LOGE("device init failed");
      return base::kStatusCodeErrorDeviceAscendCL;
    } else {
      devices_.insert({device_id, device});
      return base::kStatusCodeOk;
    }
  }

  return base::kStatusCodeOk;
}

Device *AscendCLArchitecture::getDevice(int device_id) {
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

std::vector<DeviceInfo> AscendCLArchitecture::getDeviceInfo(
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
 * @note: 暂未考虑锁业的情况
 */
BufferDesc AscendCLDevice::toBufferDesc(const MatDesc &desc,
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
BufferDesc AscendCLDevice::toBufferDesc(const TensorDesc &desc,
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

Buffer *AscendCLDevice::allocate(size_t size) {
  BufferDesc desc(size);
  return this->allocate(desc);
}
Buffer *AscendCLDevice::allocate(const BufferDesc &desc) {
  void *data = nullptr;
  aclError status =
      aclrtMalloc(&data, desc.size_[0], ACL_MEM_MALLOC_NORMAL_ONLY);
  if (ACL_SUCCESS != status) {
    NNDEPLOY_LOGE("ascend_cl alloc failed with size %lu for %p, status:%d\n",
                  desc.size_[0], data, status);
    return nullptr;
  }
  if (data == nullptr) {
    NNDEPLOY_LOGE("ascend_cl alloc got nullptr\n");
    return nullptr;
  }
  Buffer *buffer = Device::create(desc, data, kMemoryTypeAllocate);
  return buffer;
}
void AscendCLDevice::deallocate(Buffer *buffer) {
  if (buffer == nullptr) {
    return;
  }
  if (buffer->subRef() > 1) {
    return;
  }
  MemoryType buffer_source_type = buffer->getMemoryType();
  if (buffer_source_type == kMemoryTypeNone ||
      buffer_source_type == kMemoryTypeExternal) {
    Device::destory(buffer);
  } else if (buffer_source_type == kMemoryTypeAllocate) {
    if (buffer->getData() != nullptr) {
      void *data = buffer->getData();
      // NNDEPLOY_CUDA_CHECK(cudaFree(data));
      aclError ret = aclrtFree(data);
      if (ret != ACL_SUCCESS) {
        NNDEPLOY_LOGE("deallocate fuction: aclrtFree failed, errorCode is %d",
                      ret);
        return;
      }
    }
    Device::destory(buffer);
  } else if (buffer_source_type == kMemoryTypeMapped) {
    return;
  } else {
    return;
  }
}

base::Status AscendCLDevice::copy(Buffer *src, Buffer *dst) {
  if (compareBufferDesc(dst->getDesc(), src->getDesc()) >= 0) {
    aclError ret = aclrtMemcpyAsync(dst->getData(), dst->getDesc().size_[0],
                                    src->getData(), src->getDesc().size_[0],
                                    ACL_MEMCPY_DEVICE_TO_DEVICE, stream_);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("copy fuction: aclrtMemcpyAsync failed, errorCode is %d",
                    ret);
      return base::kStatusCodeErrorDeviceAscendCL;
    }
    ret = aclrtSynchronizeStream(stream_);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE(
          "copy fuction: aclrtSynchronizeStream failed, errorCode is %d", ret);
      return base::kStatusCodeErrorDeviceAscendCL;
    }
    return base::kStatusCodeOk;
  } else {
    // TODO: add log
    return base::kStatusCodeErrorOutOfMemory;
  }
}
base::Status AscendCLDevice::download(Buffer *src, Buffer *dst) {
  if (compareBufferDesc(dst->getDesc(), src->getDesc()) >= 0) {
    aclError ret = aclrtMemcpyAsync(dst->getData(), dst->getDesc().size_[0],
                                    src->getData(), src->getDesc().size_[0],
                                    ACL_MEMCPY_DEVICE_TO_HOST, stream_);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE(
          "download fuction: aclrtMemcpyAsync failed, errorCode is %d", ret);
      return base::kStatusCodeErrorDeviceAscendCL;
    }
    ret = aclrtSynchronizeStream(stream_);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE(
          "download fuction: aclrtSynchronizeStream failed, errorCode is %d",
          ret);
      return base::kStatusCodeErrorDeviceAscendCL;
    }
    return base::kStatusCodeOk;
  } else {
    // TODO: add log
    return base::kStatusCodeErrorOutOfMemory;
  }
}
base::Status AscendCLDevice::upload(Buffer *src, Buffer *dst) {
  if (compareBufferDesc(dst->getDesc(), src->getDesc()) >= 0) {
    aclError ret = aclrtMemcpyAsync(dst->getData(), dst->getDesc().size_[0],
                                    src->getData(), src->getDesc().size_[0],
                                    ACL_MEMCPY_HOST_TO_DEVICE, stream_);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("upload fuction: aclrtMemcpyAsync failed, errorCode is %d",
                    ret);
      return base::kStatusCodeErrorDeviceAscendCL;
    }
    ret = aclrtSynchronizeStream(stream_);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE(
          "upload fuction: aclrtSynchronizeStream failed, errorCode is %d",
          ret);
      return base::kStatusCodeErrorDeviceAscendCL;
    }
    return base::kStatusCodeOk;
  } else {
    // TODO: add log
    return base::kStatusCodeErrorOutOfMemory;
  }
}

base::Status AscendCLDevice::synchronize() {
  aclError ret = aclrtSynchronizeStream(stream_);
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE(
        "synchronize fuction: aclrtSynchronizeStream failed, errorCode is %d",
        ret);
    return base::kStatusCodeErrorDeviceAscendCL;
  }
  return base::kStatusCodeOk;
}
void *AscendCLDevice::getContext() { return (void *)context_; }
void *AscendCLDevice::getCommandQueue() { return (void *)stream_; }

base::Status AscendCLDevice::init() {
  if (external_command_queue_ == nullptr) {
    aclError ret = aclrtSetDevice(device_type_.device_id_);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclrtSetDevice failed, errorCode is %d", ret);
      return base::kStatusCodeErrorDeviceAscendCL;
    }

    ret = aclrtCreateContext(&context_, device_type_.device_id_);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclrtCreateContext failed, errorCode is %d", ret);
      return base::kStatusCodeErrorDeviceAscendCL;
    }

    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclrtCreateStream failed, errorCode is %d", ret);
      return base::kStatusCodeErrorDeviceAscendCL;
    }
  } else {
    stream_ = (aclrtStream)(external_command_queue_);
  }
  return base::kStatusCodeOk;
}

// Always：修改了deinit函数，分为两种情况
// 1. 共享了外部的command_queue，此时不需要释放stream_和context_
// 2. 没有共享外部的command_queue，此时需要释放stream_和context_
base::Status AscendCLDevice::deinit() {
  if (external_command_queue_ != nullptr) {
    aclError ret;
    if (stream_ != nullptr) {
      ret = aclrtDestroyStream(stream_);
      if (ret != ACL_SUCCESS) {
        NNDEPLOY_LOGE("aclrtDestroyStream failed, errorCode is %d", ret);
        return base::kStatusCodeErrorDeviceAscendCL;
      }
      stream_ = nullptr;
    }

    if (context_ != nullptr) {
      ret = aclrtDestroyContext(context_);
      if (ret != ACL_SUCCESS) {
        NNDEPLOY_LOGE("aclrtDestroyContext failed, errorCode is %d", ret);
        return base::kStatusCodeErrorDeviceAscendCL;
      }
      context_ = nullptr;

      ret = aclrtResetDevice(device_type_.device_id_);
      if (ret != ACL_SUCCESS) {
        NNDEPLOY_LOGE("aclrtResetDevice failed, errorCode is %d", ret);
        return base::kStatusCodeErrorDeviceAscendCL;
      }
    }
  }
  return base::kStatusCodeOk;
}

}  // namespace device
}  // namespace nndeploy