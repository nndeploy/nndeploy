
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
    : Architecture(device_type_code) {};

AscendCLArchitecture::~AscendCLArchitecture() {
  for (auto iter : devices_) {
    AscendCLDevice *tmp_device = dynamic_cast<AscendCLDevice *>(iter.second);
    if (tmp_device->deinit() != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("device deinit failed");
    }
    delete tmp_device;
  }
};

void AscendCLArchitecture::setAclConfigPath(
    int device_id, const std::string &acl_config_path) {
  acl_config_path_map_[device_id] = acl_config_path;
}

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

    if (acl_config_path_map_.find(device_id) != acl_config_path_map_.end()) {
      device->setAclConfigPath(acl_config_path_map_[device_id]);
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

void *AscendCLDevice::allocate(size_t size) {
  BufferDesc desc(size);
  return this->allocate(desc);
}
void *AscendCLDevice::allocate(const BufferDesc &desc) {
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
  return data;
}
void AscendCLDevice::deallocate(void *ptr) {
  if (ptr == nullptr) {
    return;
  }
  aclError ret = aclrtFree(ptr);
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("deallocate fuction: aclrtFree failed, errorCode is %d", ret);
    return;
  }
}

base::Status AscendCLDevice::copy(void *src, void *dst, size_t size,
                                  int index) {
  aclrtStream stream = (aclrtStream)(this->getCommandQueue(index));
  if (src != nullptr && dst != nullptr) {
    aclError ret = aclrtMemcpyAsync(dst, size, src, size,
                                    ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("copy fuction: aclrtMemcpyAsync failed, errorCode is %d",
                    ret);
      return base::kStatusCodeErrorDeviceAscendCL;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE(
          "copy fuction: aclrtSynchronizeStream failed, errorCode is %d", ret);
      return base::kStatusCodeErrorDeviceAscendCL;
    }
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("copy buffer failed");
    return base::kStatusCodeErrorOutOfMemory;
  }
}
base::Status AscendCLDevice::download(void *src, void *dst, size_t size,
                                      int index) {
  aclrtStream stream = (aclrtStream)(this->getCommandQueue(index));
  if (src != nullptr && dst != nullptr) {
    aclError ret = aclrtMemcpyAsync(dst, size, src, size,
                                    ACL_MEMCPY_DEVICE_TO_HOST, stream);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE(
          "download fuction: aclrtMemcpyAsync failed, errorCode is %d", ret);
      return base::kStatusCodeErrorDeviceAscendCL;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE(
          "download fuction: aclrtSynchronizeStream failed, errorCode is %d",
          ret);
      return base::kStatusCodeErrorDeviceAscendCL;
    }
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("copy buffer failed");
    return base::kStatusCodeErrorOutOfMemory;
  }
}
base::Status AscendCLDevice::upload(void *src, void *dst, size_t size,
                                    int index) {
  aclrtStream stream = (aclrtStream)(this->getCommandQueue(index));
  if (src != nullptr && dst != nullptr) {
    aclError ret = aclrtMemcpyAsync(dst, size, src, size,
                                    ACL_MEMCPY_HOST_TO_DEVICE, stream);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("upload fuction: aclrtMemcpyAsync failed, errorCode is %d",
                    ret);
      return base::kStatusCodeErrorDeviceAscendCL;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE(
          "upload fuction: aclrtSynchronizeStream failed, errorCode is %d",
          ret);
      return base::kStatusCodeErrorDeviceAscendCL;
    }
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("copy buffer failed");
    return base::kStatusCodeErrorOutOfMemory;
  }
}

base::Status AscendCLDevice::copy(Buffer *src, Buffer *dst, int index) {
  if (src != nullptr && dst != nullptr && dst->getDesc() >= src->getDesc()) {
    aclrtStream stream = (aclrtStream)(this->getCommandQueue(index));
    aclError ret = aclrtMemcpyAsync(dst->getData(), src->getDesc().size_[0],
                                    src->getData(), src->getDesc().size_[0],
                                    ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("copy fuction: aclrtMemcpyAsync failed, errorCode is %d",
                    ret);
      return base::kStatusCodeErrorDeviceAscendCL;
    }
    ret = aclrtSynchronizeStream(stream);
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
base::Status AscendCLDevice::download(Buffer *src, Buffer *dst, int index) {
  if (src != nullptr && dst != nullptr && dst->getDesc() >= src->getDesc()) {
    aclrtStream stream = (aclrtStream)(this->getCommandQueue(index));
    aclError ret = aclrtMemcpyAsync(dst->getData(), src->getDesc().size_[0],
                                    src->getData(), src->getDesc().size_[0],
                                    ACL_MEMCPY_DEVICE_TO_HOST, stream);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE(
          "download fuction: aclrtMemcpyAsync failed, errorCode is %d", ret);
      return base::kStatusCodeErrorDeviceAscendCL;
    }
    ret = aclrtSynchronizeStream(stream);
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
base::Status AscendCLDevice::upload(Buffer *src, Buffer *dst, int index) {
  if (src != nullptr && dst != nullptr && dst->getDesc() >= src->getDesc()) {
    aclrtStream stream = (aclrtStream)(this->getCommandQueue(index));
    aclError ret = aclrtMemcpyAsync(dst->getData(), src->getDesc().size_[0],
                                    src->getData(), src->getDesc().size_[0],
                                    ACL_MEMCPY_HOST_TO_DEVICE, stream);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("upload fuction: aclrtMemcpyAsync failed, errorCode is %d",
                    ret);
      return base::kStatusCodeErrorDeviceAscendCL;
    }
    ret = aclrtSynchronizeStream(stream);
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

void *AscendCLDevice::getContext() { return (void *)context_; }

base::Status AscendCLDevice::newCommandQueue() {
  AclStreamWrapper acl_stream_wrapper;
  acl_stream_wrapper.external_command_queue_ = nullptr;
  aclError status = aclrtCreateStream(&acl_stream_wrapper.stream_);
  if (ACL_SUCCESS != status) {
    NNDEPLOY_LOGE("acl stream create failed\n");
    return base::kStatusCodeErrorDeviceAscendCL;
  }
  acl_stream_wrapper_.emplace_back(acl_stream_wrapper);
  return base::kStatusCodeOk;
}
base::Status AscendCLDevice::deleteCommandQueue(int index) {
  if (index < 0) {
    index = acl_stream_wrapper_.size() - 1;
  }
  if (index < acl_stream_wrapper_.size()) {
    base::Status status = synchronize(index);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "synchronize failed");
    if (acl_stream_wrapper_[index].external_command_queue_ == nullptr) {
      aclError ret = aclrtDestroyStream(acl_stream_wrapper_[index].stream_);
      if (ACL_SUCCESS != ret) {
        NNDEPLOY_LOGE("acl stream destroy failed\n");
        return base::kStatusCodeErrorDeviceAscendCL;
      }
    }
    acl_stream_wrapper_.erase(acl_stream_wrapper_.begin() + index);
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("index[%d] is error\n", index);
    return base::kStatusCodeErrorDeviceAscendCL;
  }
}
base::Status AscendCLDevice::deleteCommandQueue(void *command_queue) {
  for (int i = 0; i < acl_stream_wrapper_.size(); ++i) {
    if (command_queue == (void *)(acl_stream_wrapper_[i].stream_)) {
      return this->deleteCommandQueue(i);
    }
  }
  NNDEPLOY_LOGE("command queue is not found\n");
  return base::kStatusCodeOk;
}
base::Status AscendCLDevice::setCommandQueue(void *command_queue) {
  if (command_queue == nullptr) {
    NNDEPLOY_LOGE("command queue is nullptr\n");
    return base::kStatusCodeErrorDeviceAscendCL;
  }
  AclStreamWrapper acl_stream_wrapper;
  acl_stream_wrapper.external_command_queue_ = command_queue;
  acl_stream_wrapper.stream_ = (aclrtStream)command_queue;
  acl_stream_wrapper_.emplace_back(acl_stream_wrapper);
  return base::kStatusCodeOk;
}

void *AscendCLDevice::getCommandQueue(int index) {
  if (index < 0) {
    index = acl_stream_wrapper_.size() - 1;
  }
  if (index < acl_stream_wrapper_.size()) {
    return (void *)(acl_stream_wrapper_[index].stream_);
  } else {
    return nullptr;
  }
}

base::Status AscendCLDevice::synchronize(int index) {
  aclrtStream stream = (aclrtStream)(this->getCommandQueue(index));
  aclError ret = aclrtSynchronizeStream(stream);
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE(
        "synchronize fuction: aclrtSynchronizeStream failed, errorCode is %d",
        ret);
    return base::kStatusCodeErrorDeviceAscendCL;
  }
  return base::kStatusCodeOk;
}

void AscendCLDevice::setAclConfigPath(const std::string &acl_config_path) {
  acl_config_path_ = acl_config_path;
}

base::Status AscendCLDevice::init() {
  if (acl_stream_wrapper_[0].external_command_queue_ == nullptr) {
    // 初始化
    aclError ret = aclInit(acl_config_path_.c_str());
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclInit failed, errorCode is %d", ret);
    }
    // 选择设备
    ret = aclrtSetDevice(device_type_.device_id_);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclrtSetDevice failed, errorCode is %d", ret);
      return base::kStatusCodeErrorDeviceAscendCL;
    }
    // Always: 当共享外部的command_queue时，不需要创建context吗？
    ret = aclrtCreateContext(&context_, device_type_.device_id_);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclrtCreateContext failed, errorCode is %d", ret);
      return base::kStatusCodeErrorDeviceAscendCL;
    }
    // 创建流
    ret = aclrtCreateStream(&acl_stream_wrapper_[0].stream_);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclrtCreateStream failed, errorCode is %d", ret);
      return base::kStatusCodeErrorDeviceAscendCL;
    }
  } else {
    acl_stream_wrapper_[0].stream_ =
        (aclrtStream)(acl_stream_wrapper_[0].external_command_queue_);
  }
  return base::kStatusCodeOk;
}

// Always：修改了deinit函数，分为两种情况
// 1. 共享了外部的command_queue，此时不需要释放stream_和context_
// 2. 没有共享外部的command_queue，此时需要释放stream_和context_
base::Status AscendCLDevice::deinit() {
  for (auto iter : acl_stream_wrapper_) {
    aclError ret = aclrtSynchronizeStream(iter.stream_);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE(
          "synchronize fuction: aclrtSynchronizeStream failed, errorCode is %d",
          ret);
      return base::kStatusCodeErrorDeviceAscendCL;
    }
    if (iter.external_command_queue_ != nullptr) {
      aclError ret;
      if (iter.stream_ != nullptr) {
        ret = aclrtDestroyStream(iter.stream_);
        if (ret != ACL_SUCCESS) {
          NNDEPLOY_LOGE("aclrtDestroyStream failed, errorCode is %d", ret);
          return base::kStatusCodeErrorDeviceAscendCL;
        }
        iter.stream_ = nullptr;
      }
    }
  }
  if (context_ != nullptr) {
    aclError ret = aclrtDestroyContext(context_);
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

    ret = aclFinalize();
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclFinalize failed, errorCode is %d", ret);
      return base::kStatusCodeErrorDeviceAscendCL;
    }
  }
  return base::kStatusCodeOk;
}

}  // namespace device
}  // namespace nndeploy