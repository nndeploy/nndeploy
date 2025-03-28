#include "nndeploy/device/ascend_cl/ascend_cl_device.h"

#include <climits>

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
  // std::once_flag once;
  // std::call_once(once, [this]() {
  //   aclError ret = aclFinalize();
  //   if (ret != ACL_SUCCESS) {
  //     NNDEPLOY_LOGE("aclFinalize failed, errorCode is %d\n", ret);
  //   }
  // });
};

void AscendCLArchitecture::setAclConfigPath(
    const std::string &acl_config_path) {
  acl_config_path_ = acl_config_path;
}

base::Status AscendCLArchitecture::checkDevice(int device_id,
                                               std::string library_path) {
  int device_count = ascendCLGetNumDevices();
  if (device_id > -1 && device_id < device_count) {
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("device id is invalid, device id: %d, device count: %d\n",
                  device_id, device_count);
    return base::kStatusCodeErrorDeviceAscendCL;
  }
}

base::Status AscendCLArchitecture::enableDevice(int device_id,
                                                std::string library_path) {
  base::DeviceType device_type(base::kDeviceTypeCodeAscendCL, device_id);
  std::lock_guard<std::mutex> lock(mutex_);
  if (devices_.find(device_id) == devices_.end()) {
    AscendCLDevice *device = new AscendCLDevice(device_type, library_path);
    if (device == nullptr) {
      NNDEPLOY_LOGE("device is nullptr\n");
      return base::kStatusCodeErrorOutOfMemory;
    }

    if (device->init() != base::kStatusCodeOk) {
      delete device;
      NNDEPLOY_LOGE("device init failed\n");
      return base::kStatusCodeErrorDeviceAscendCL;
    } else {
      devices_.insert({device_id, device});
      return base::kStatusCodeOk;
    }
  }

  return base::kStatusCodeOk;
}

Device *AscendCLArchitecture::getDevice(int device_id) {
  static std::once_flag once;
  std::call_once(once, [this]() {
    aclError ret = aclInit(acl_config_path_.c_str());
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclInit failed, errorCode is %d\n", ret);
    };
  });
  Device *device = nullptr;
  if (devices_.find(device_id) != devices_.end()) {
    return devices_[device_id];
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

void *AscendCLDevice::allocate(size_t size) {
  BufferDesc desc(size);
  return this->allocate(desc);
}
void *AscendCLDevice::allocate(const BufferDesc &desc) {
  aclError ret = aclrtSetCurrentContext(context_);
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("aclrtSetCurrentContext failed, errorCode is %d\n", ret);
    return nullptr;
  }
  void *data = nullptr;
  aclError status =
      aclrtMalloc(&data, desc.getRealSize(), ACL_MEM_MALLOC_NORMAL_ONLY);
  if (ACL_SUCCESS != status) {
    NNDEPLOY_LOGE("ascend_cl alloc failed with size %lu for %p, status:%d\n",
                  desc.getRealSize(), data, status);
    return nullptr;
  }
  if (data == nullptr) {
    NNDEPLOY_LOGE("ascend_cl alloc got nullptr\n");
    return nullptr;
  }
  return data;
}
void AscendCLDevice::deallocate(void *ptr) {
  aclError ret = aclrtSetCurrentContext(context_);
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("aclrtSetCurrentContext failed, errorCode is %d\n", ret);
    return;
  }
  if (ptr == nullptr) {
    return;
  }
  ret = aclrtFree(ptr);
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("deallocate fuction: aclrtFree failed, errorCode is %d\n",
                  ret);
    return;
  }
}

base::Status AscendCLDevice::copy(void *src, void *dst, size_t size,
                                  Stream *stream) {
  aclError ret = aclrtSetCurrentContext(context_);
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("aclrtSetCurrentContext failed, errorCode is %d\n", ret);
    return base::kStatusCodeErrorDeviceAscendCL;
  }
  if (stream == nullptr) {
    aclError ret =
        aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE(
          "aclrtMemcpy device to device copy failed, error code is %d\n", ret);
      return base::kStatusCodeErrorDeviceAscendCL;
    }
  } else {
    aclrtStream acl_stream =
        (aclrtStream)stream->as<AscendCLStream>()->getStream();
    aclError ret = aclrtMemcpyAsync(dst, size, src, size,
                                    ACL_MEMCPY_DEVICE_TO_DEVICE, acl_stream);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE(
          "aclrtMemcpyAsync device to device copy failed, error code is %d\n",
          ret);
      return base::kStatusCodeErrorDeviceAscendCL;
    }
  }
  return base::kStatusCodeOk;
}

base::Status AscendCLDevice::download(void *src, void *dst, size_t size,
                                      Stream *stream) {
  aclError ret = aclrtSetCurrentContext(context_);
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("aclrtSetCurrentContext failed, errorCode is %d\n", ret);
    return base::kStatusCodeErrorDeviceAscendCL;
  }
  if (stream == nullptr) {
    NNDEPLOY_ASCEND_CL_CHECK(
        aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST));
  } else {
    aclrtStream acl_stream =
        (aclrtStream)stream->as<AscendCLStream>()->getStream();
    NNDEPLOY_ASCEND_CL_CHECK(aclrtMemcpyAsync(
        dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST, acl_stream));
  }

  return base::kStatusCodeOk;
}

base::Status AscendCLDevice::upload(void *src, void *dst, size_t size,
                                    Stream *stream) {
  aclError ret = aclrtSetCurrentContext(context_);
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("aclrtSetCurrentContext failed, errorCode is %d\n", ret);
    return base::kStatusCodeErrorDeviceAscendCL;
  }
  if (stream == nullptr) {
    aclError ret = aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclrtMemcpy failed, errorCode is %d\n", ret);
      return base::kStatusCodeErrorDeviceAscendCL;
    }
  } else {
    aclrtStream acl_stream =
        (aclrtStream)stream->as<AscendCLStream>()->getStream();
    aclError ret = aclrtMemcpyAsync(dst, size, src, size,
                                    ACL_MEMCPY_HOST_TO_DEVICE, acl_stream);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclrtMemcpyAsync failed, errorCode is %d\n", ret);
      return base::kStatusCodeErrorDeviceAscendCL;
    }
  }

  return base::kStatusCodeOk;
}

base::Status AscendCLDevice::copy(Buffer *src, Buffer *dst, Stream *stream) {
  size_t dst_size = dst->getSize();
  size_t src_size = src->getSize();
  size_t size = std::min(dst_size, src_size);
  return this->copy(src->getData(), dst->getData(), size, stream);
}

base::Status AscendCLDevice::download(Buffer *src, Buffer *dst,
                                      Stream *stream) {
  size_t dst_size = dst->getSize();
  size_t src_size = src->getSize();
  size_t size = std::min(dst_size, src_size);
  return this->download(src->getData(), dst->getData(), size, stream);
}

base::Status AscendCLDevice::upload(Buffer *src, Buffer *dst, Stream *stream) {
  size_t dst_size = dst->getSize();
  size_t src_size = src->getSize();
  size_t size = std::min(dst_size, src_size);
  return this->upload(src->getData(), dst->getData(), size, stream);
}

void *AscendCLDevice::getContext() { return (void *)context_; }

base::Status AscendCLDevice::bindThread() {
  aclError ret = aclrtSetCurrentContext((aclrtContext)context_);
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE(
        "aclrtSetCurrentContext failed, errorCode is %d, ACL_SUCCESS = %d\n",
        ret, ACL_SUCCESS);
    return base::kStatusCodeErrorDeviceAscendCL;
  }
  return base::kStatusCodeOk;
}

base::Status AscendCLDevice::init() {
  // 初始化
  // aclError ret = aclInit(acl_config_path_.c_str());
  // if (ret != ACL_SUCCESS) {
  //   NNDEPLOY_LOGE("aclInit failed, errorCode is %d\n", ret);
  // }
  // 选择设备
  NNDEPLOY_LOGI("set device %d\n", device_type_.device_id_);
  aclError ret = aclrtSetDevice(device_type_.device_id_);
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("aclrtSetDevice failed, errorCode is %d\n", ret);
    return base::kStatusCodeErrorDeviceAscendCL;
  }
  // Always: 如何共享外部context
  ret = aclrtCreateContext(&context_, device_type_.device_id_);
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("aclrtCreateContext failed, errorCode is %d\n", ret);
    return base::kStatusCodeErrorDeviceAscendCL;
  }
  return base::kStatusCodeOk;
}

base::Status AscendCLDevice::deinit() {
  if (context_ != nullptr) {
    aclError ret = aclrtDestroyContext(context_);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclrtDestroyContext failed, errorCode is %d\n", ret);
      return base::kStatusCodeErrorDeviceAscendCL;
    }
    context_ = nullptr;

    ret = aclrtResetDevice(device_type_.device_id_);
    if (ret != ACL_SUCCESS) {
      NNDEPLOY_LOGE("aclrtResetDevice failed, errorCode is %d\n", ret);
      return base::kStatusCodeErrorDeviceAscendCL;
    }

    static std::once_flag once;
    std::call_once(once, [this]() {
      aclError ret = aclFinalize();
      if (ret != ACL_SUCCESS) {
        NNDEPLOY_LOGE("aclFinalize failed, errorCode is %d\n", ret);
      }
    });

    // ret = aclFinalize();
    // if (ret != ACL_SUCCESS) {
    //   NNDEPLOY_LOGE("aclFinalize failed, errorCode is %d\n", ret);
    //   return base::kStatusCodeErrorDeviceAscendCL;
    // }
  }
  return base::kStatusCodeOk;
}

Stream *AscendCLDevice::createStream() { return new AscendCLStream(this); }

Stream *AscendCLDevice::createStream(void *stream) {
  return new AscendCLStream(this, stream);
}

base::Status AscendCLDevice::destroyStream(Stream *stream) {
  if (stream == nullptr) {
    NNDEPLOY_LOGE("stream is nullptr\n");
    return base::kStatusCodeOk;
  }
  delete stream;
  stream = nullptr;
  return base::kStatusCodeOk;
}

Event *AscendCLDevice::createEvent() { return new AscendCLEvent(this); }
base::Status AscendCLDevice::destroyEvent(Event *event) {
  if (event == nullptr) {
    NNDEPLOY_LOGE("event is nullptr\n");
    return base::kStatusCodeErrorDeviceAscendCL;
  }
  delete event;
  event = nullptr;
  return base::kStatusCodeOk;
}
base::Status AscendCLDevice::createEvents(Event **events, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    Event *event = this->createEvent();
    if (event == nullptr) {
      return base::kStatusCodeErrorDeviceAscendCL;
    }
  }
  return base::kStatusCodeOk;
}
base::Status AscendCLDevice::destroyEvents(Event **events, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    this->destroyEvent(events[i]);
  }
  return base::kStatusCodeOk;
}

AscendCLStream::AscendCLStream(Device *device) : Stream(device) {
  aclError ret = aclrtSetCurrentContext((aclrtContext)(device_->getContext()));
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("aclrtSetCurrentContext failed, errorCode is %d\n", ret);
    return;
  }
  aclrtStream stream;
  ret = aclrtCreateStream(&stream);
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("aclrtCreateStream failed, errorCode is %d\n", ret);
  }
  stream_ = stream;
}
AscendCLStream::AscendCLStream(Device *device, void *stream)
    : Stream(device, stream), stream_((aclrtStream)stream) {}
AscendCLStream::~AscendCLStream() {
  aclError ret = aclrtSetCurrentContext((aclrtContext)(device_->getContext()));
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("aclrtSetCurrentContext failed, errorCode is %d\n", ret);
    return;
  }
  if (is_external_) {
    return;
  }
  ret = aclrtDestroyStream(stream_);
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("aclrtDestroyStream failed, errorCode is %d\n", ret);
  }
}

base::Status AscendCLStream::synchronize() {
  aclError ret = aclrtSetCurrentContext((aclrtContext)(device_->getContext()));
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("aclrtSetCurrentContext failed, errorCode is %d\n", ret);
    return base::kStatusCodeErrorDeviceAscendCL;
  }
  ret = aclrtSynchronizeStream(stream_);
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("aclrtSynchronizeStream failed, errorCode is %d\n", ret);
    return base::kStatusCodeErrorDeviceAscendCL;
  }
  return base::kStatusCodeOk;
}
base::Status AscendCLStream::recordEvent(Event *event) {
  aclError ret = aclrtSetCurrentContext((aclrtContext)(device_->getContext()));
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("aclrtSetCurrentContext failed, errorCode is %d\n", ret);
    return base::kStatusCodeErrorDeviceAscendCL;
  }
  if (event == nullptr) {
    NNDEPLOY_LOGE("event is nullptr\n");
    return base::kStatusCodeErrorDeviceAscendCL;
  }
  ret = aclrtRecordEvent(event->as<AscendCLEvent>()->getEvent(), stream_);
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("aclrtRecordStream failed, errorCode is %d\n", ret);
  }
  return base::kStatusCodeOk;
}
base::Status AscendCLStream::waitEvent(Event *event) {
  aclError ret = aclrtSetCurrentContext((aclrtContext)(device_->getContext()));
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("aclrtSetCurrentContext failed, errorCode is %d\n", ret);
    return base::kStatusCodeErrorDeviceAscendCL;
  }
  if (event == nullptr) {
    NNDEPLOY_LOGE("event is nullptr\n");
    return base::kStatusCodeErrorDeviceAscendCL;
  }
  ret = aclrtStreamWaitEvent(stream_, event->as<AscendCLEvent>()->getEvent());
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("aclrtStreamWaitEvent failed, errorCode is %d\n", ret);
  }
  return base::kStatusCodeOk;
}

void *AscendCLStream::getNativeStream() { return (void *)stream_; }
aclrtStream AscendCLStream::getStream() { return stream_; }

// event
AscendCLEvent::AscendCLEvent(Device *device) : Event(device) {
  aclError ret = aclrtSetCurrentContext((aclrtContext)(device_->getContext()));
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("aclrtSetCurrentContext failed, errorCode is %d\n", ret);
    return;
  }
  ret = aclrtCreateEvent(&event_);
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("aclrtCreateEvent failed, errorCode is %d\n", ret);
  }
}
AscendCLEvent::~AscendCLEvent() {
  aclError ret = aclrtSetCurrentContext((aclrtContext)(device_->getContext()));
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("aclrtSetCurrentContext failed, errorCode is %d\n", ret);
    return;
  }
  ret = aclrtDestroyEvent(event_);
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("aclrtDestroyEvent failed, errorCode is %d\n", ret);
  }
}

bool AscendCLEvent::queryDone() {
  aclError ret = aclrtSetCurrentContext((aclrtContext)(device_->getContext()));
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("aclrtSetCurrentContext failed, errorCode is %d\n", ret);
    return false;
  }
  aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
  ret = aclrtQueryEventStatus(event_, &status);
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("aclrtQueryEvent failed, errorCode is %d\n", ret);
    return false;
  } else {
    return status == ACL_EVENT_RECORDED_STATUS_COMPLETE;
  }
}
base::Status AscendCLEvent::synchronize() {
  aclError ret = aclrtSetCurrentContext((aclrtContext)(device_->getContext()));
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("aclrtSetCurrentContext failed, errorCode is %d\n", ret);
    return base::kStatusCodeErrorDeviceAscendCL;
  }
  ret = aclrtSynchronizeEvent(event_);
  if (ret != ACL_SUCCESS) {
    NNDEPLOY_LOGE("aclrtSynchronizeEvent failed, errorCode is %d\n", ret);
  }
  return base::kStatusCodeOk;
}

aclrtEvent AscendCLEvent::getEvent() { return event_; }

}  // namespace device
}  // namespace nndeploy