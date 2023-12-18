
#include "nndeploy/device/ascend_cl/ascend_cl_device.h"

#include "nndeploy/device/buffer.h"
#include "nndeploy/device/mat.h"
#include "nndeploy/device/ascend_cl/ascend_cl_util.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace device {

TypeArchitectureRegister<AscendclArchitecture> mdc_architecture_register(
    base::kDeviceTypeCodeASCENDCL);

AscendclArchitecture::AscendclArchitecture(base::DeviceTypeCode device_type_code)
    : Architecture(device_type_code){};

AscendclArchitecture::~AscendclArchitecture() {
  for (auto iter : devices_) {
    AscendclDevice *tmp_device = dynamic_cast<AscendclDevice *>(iter.second);
    if (tmp_device->deinit() != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("device deinit failed");
    }
    delete tmp_device;
  }
};

base::Status AscendclArchitecture::checkDevice(int device_id, void *command_queue,
                                          std::string library_path) {
  int device_count = mdcGetNumDevices();
  if (device_id > -1 && device_id < device_count) {
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("device id is invalid, device id: %d, device count: %d",
                  device_id, device_count);
    return base::kStatusCodeErrorDeviceASCENDCL;
  }
}

base::Status AscendclArchitecture::enableDevice(int device_id, void *command_queue,
                                           std::string library_path) {
  base::DeviceType device_type(base::kDeviceTypeCodeASCENDCL, device_id);
  std::lock_guard<std::mutex> lock(mutex_);
  if (devices_.find(device_id) == devices_.end()) {
    AscendclDevice *device = new AscendclDevice(device_type, command_queue, library_path);
    if (device == nullptr) {
      NNDEPLOY_LOGE("device is nullptr");
      return base::kStatusCodeErrorOutOfMemory;
    }

    if (device->init() != base::kStatusCodeOk) {
      delete device;
      NNDEPLOY_LOGE("device init failed");
      return base::kStatusCodeErrorDeviceASCENDCL;
    } else {
      devices_.insert({device_id, device});
      return base::kStatusCodeOk;
    }
  }

  return base::kStatusCodeOk;
}

Device *AscendclArchitecture::getDevice(int device_id) {
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

std::vector<DeviceInfo> AscendclArchitecture::getDeviceInfo(
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
BufferDesc AscendclDevice::toBufferDesc(const MatDesc &desc,
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
BufferDesc AscendclDevice::toBufferDesc(const TensorDesc &desc,
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

Buffer *AscendclDevice::allocate(size_t size) {
  BufferDesc desc(size);
  return this->allocate(desc);
}
Buffer *AscendclDevice::allocate(const BufferDesc &desc) {
  void *data = nullptr;
  aclError status =
      aclrtMalloc(&data, desc.size_[0], ACL_MEM_MALLOC_NORMAL_ONLY);
  if (ACL_SUCCESS != status) {
    NNDEPLOY_LOGE("mdc alloc failed with size %lu for %p, status:%d\n",
                  desc.size_[0], data, status);
    return nullptr;
  }
  if (data == nullptr) {
    NNDEPLOY_LOGE("mdc alloc got nullptr\n");
    return nullptr;
  }
  Buffer *buffer = Device::create(desc, data, kBufferSourceTypeAllocate);
  return buffer;
}
void AscendclDevice::deallocate(Buffer *buffer) {
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
      // NNDEPLOY_CUDA_CHECK(cudaFree(data));
      aclError ret = aclrtFree(data);
      if (ret != ACL_SUCCESS) {
        NNDEPLOY_LOGE("deallocate fuction: aclrtFree failed, errorCode is %d",
                      ret);
        return;
      }
    }
    Device::destory(buffer);
  } else if (buffer_source_type == kBufferSourceTypeMapped) {
    return;
  } else {
    return;
  }
}

base::Status AscendclDevice::copy(Buffer *src, Buffer *dst) {
  if (compareBufferDesc(dst->getDesc(), src->getDesc()) >= 0) {
    aclError ret = aclrtMemcpyAsync(dst->getPtr(), dst->getDesc().size_[0],
                                    src->getPtr(), src->getDesc().size_[0],
                                    ACL_MEMCPY_DEVICE_TO_DEVICE, stream_);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("copy fuction: aclrtMemcpyAsync failed, errorCode is %d",
                    ret);
      return base::kStatusCodeErrorDeviceASCENDCL;
    }
    ret = aclrtSynchronizeStream(stream_);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE(
          "copy fuction: aclrtSynchronizeStream failed, errorCode is %d", ret);
      return base::kStatusCodeErrorDeviceASCENDCL;
    }
    return base::kStatusCodeOk;
  } else {
    // TODO: add log
    return base::kStatusCodeErrorOutOfMemory;
  }
}
base::Status AscendclDevice::download(Buffer *src, Buffer *dst) {
  if (compareBufferDesc(dst->getDesc(), src->getDesc()) >= 0) {
    aclError ret = aclrtMemcpyAsync(dst->getPtr(), dst->getDesc().size_[0],
                                    src->getPtr(), src->getDesc().size_[0],
                                    ACL_MEMCPY_DEVICE_TO_HOST, stream_);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE(
          "download fuction: aclrtMemcpyAsync failed, errorCode is %d", ret);
      return base::kStatusCodeErrorDeviceASCENDCL;
    }
    ret = aclrtSynchronizeStream(stream_);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE(
          "download fuction: aclrtSynchronizeStream failed, errorCode is %d",
          ret);
      return base::kStatusCodeErrorDeviceASCENDCL;
    }
    return base::kStatusCodeOk;
  } else {
    // TODO: add log
    return base::kStatusCodeErrorOutOfMemory;
  }
}
base::Status AscendclDevice::upload(Buffer *src, Buffer *dst) {
  if (compareBufferDesc(dst->getDesc(), src->getDesc()) >= 0) {
    aclError ret = aclrtMemcpyAsync(dst->getPtr(), dst->getDesc().size_[0],
                                    src->getPtr(), src->getDesc().size_[0],
                                    ACL_MEMCPY_HOST_TO_DEVICE, stream_);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("upload fuction: aclrtMemcpyAsync failed, errorCode is %d",
                    ret);
      return base::kStatusCodeErrorDeviceASCENDCL;
    }
    ret = aclrtSynchronizeStream(stream_);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE(
          "upload fuction: aclrtSynchronizeStream failed, errorCode is %d",
          ret);
      return base::kStatusCodeErrorDeviceASCENDCL;
    }
    return base::kStatusCodeOk;
  } else {
    // TODO: add log
    return base::kStatusCodeErrorOutOfMemory;
  }
}

base::Status AscendclDevice::synchronize() {
  aclError ret = aclrtSynchronizeStream(stream_);
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE(
        "synchronize fuction: aclrtSynchronizeStream failed, errorCode is %d",
        ret);
    return base::kStatusCodeErrorDeviceASCENDCL;
  }
  return base::kStatusCodeOk;
}
void *AscendclDevice::getCommandQueue() { return context_; }

base::Status AscendclDevice::init() {
  if (external_command_queue_ == nullptr) {
    aclError ret = aclrtSetDevice(device_type_.device_id_);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclrtSetDevice failed, errorCode is %d", ret);
      return base::kStatusCodeErrorDeviceASCENDCL;
    }

    ret = aclrtCreateContext(&context_, device_type_.device_id_);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclrtCreateContext failed, errorCode is %d", ret);
      return base::kStatusCodeErrorDeviceASCENDCL;
    }

    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclrtCreateStream failed, errorCode is %d", ret);
      return base::kStatusCodeErrorDeviceASCENDCL;
    }
  } else {
    stream_ = (aclrtStream)(external_command_queue_);
  }
  return base::kStatusCodeOk;
}

// Always：修改了deinit函数，分为两种情况
// 1. 共享了外部的command_queue，此时不需要释放stream_和context_
// 2. 没有共享外部的command_queue，此时需要释放stream_和context_
base::Status AscendclDevice::deinit() {
  if (external_command_queue_ != nullptr) {
    aclError ret;
    if (stream_ != nullptr) {
      ret = aclrtDestroyStream(stream_);
      if (ret != ACL_SUCCESS) {
        NNDEPLOY_LOGE("aclrtDestroyStream failed, errorCode is %d", ret);
        return base::kStatusCodeErrorDeviceASCENDCL;
      }
      stream_ = nullptr;
    }

    if (context_ != nullptr) {
      ret = aclrtDestroyContext(context_);
      if (ret != ACL_SUCCESS) {
        NNDEPLOY_LOGE("aclrtDestroyContext failed, errorCode is %d", ret);
        return base::kStatusCodeErrorDeviceASCENDCL;
      }
      context_ = nullptr;

      ret = aclrtResetDevice(device_type_.device_id_);
      if (ret != ACL_SUCCESS) {
        NNDEPLOY_LOGE("aclrtResetDevice failed, errorCode is %d", ret);
        return base::kStatusCodeErrorDeviceASCENDCL;
      }
    }
  }
  return base::kStatusCodeOk;
}

}  // namespace device
}  // namespace nndeploy