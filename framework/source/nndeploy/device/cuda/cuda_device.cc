#include "nndeploy/device/cuda/cuda_device.h"

#include "nndeploy/device/buffer.h"
#include "nndeploy/device/cuda/cuda_util.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace device {

TypeArchitectureRegister<CudaArchitecture> cuda_architecture_register(
    base::kDeviceTypeCodeCuda);

CudaArchitecture::CudaArchitecture(base::DeviceTypeCode device_type_code)
    : Architecture(device_type_code) {};

CudaArchitecture::~CudaArchitecture() {};

base::Status CudaArchitecture::checkDevice(int device_id,
                                           std::string library_path) {
  int device_count = cudaGetNumDevices();
  if (device_id > -1 && device_id < device_count) {
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("device id is invalid, device id: %d, device count: %d\n",
                  device_id, device_count);
    return base::kStatusCodeErrorDeviceCuda;
  }
}

base::Status CudaArchitecture::enableDevice(int device_id,
                                            std::string library_path) {
  base::DeviceType device_type(base::kDeviceTypeCodeCuda, device_id);
  std::lock_guard<std::mutex> lock(mutex_);
  if (devices_.find(device_id) == devices_.end()) {
    CudaDevice *device = new CudaDevice(device_type, library_path);
    if (device == nullptr) {
      NNDEPLOY_LOGE("device is nullptr\n");
      return base::kStatusCodeErrorOutOfMemory;
    }

    if (device->init() != base::kStatusCodeOk) {
      delete device;
      NNDEPLOY_LOGE("device init failed\n");
      return base::kStatusCodeErrorDeviceCuda;
    } else {
      devices_.insert({device_id, device});
      return base::kStatusCodeOk;
    }
  }

  return base::kStatusCodeOk;
}

Device *CudaArchitecture::getDevice(int device_id) {
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

std::vector<DeviceInfo> CudaArchitecture::getDeviceInfo(
    std::string library_path) {
  std::vector<DeviceInfo> device_info_list;
  int device_count = cudaGetNumDevices();
  for (int i = 0; i < device_count; ++i) {
    cudaDeviceProp p = cudaGetDeviceProperty(i);
    DeviceInfo device_info;
    device_info_list.emplace_back(device_info);
  }
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
BufferDesc CudaDevice::toBufferDesc(const TensorDesc &desc,
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

void *CudaDevice::allocate(size_t size) {
  BufferDesc desc(size);
  return this->allocate(desc);
}
void *CudaDevice::allocate(const BufferDesc &desc) {
  void *data = nullptr;
  cudaError_t status = cudaMalloc(&data, desc.getRealSize());
  if (cudaSuccess != status) {
    NNDEPLOY_LOGE("cuda alloc failed with size %lu for %p, status:%d\n",
                  desc.getRealSize(), data, status);
    return nullptr;
  }
  if (data == nullptr) {
    NNDEPLOY_LOGE("cuda alloc got nullptr\n");
    return nullptr;
  }
  return data;
}
void CudaDevice::deallocate(void *ptr) {
  if (ptr == nullptr) {
    return;
  }
  NNDEPLOY_CUDA_CHECK(cudaFree(ptr));
}

base::Status CudaDevice::copy(void *src, void *dst, size_t size,
                              Stream *stream) {
  if (stream == nullptr) {
    NNDEPLOY_CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
  } else {
    cudaStream_t cuda_stream =
        (cudaStream_t)stream->as<CudaStream>()->getStream();
    NNDEPLOY_CUDA_CHECK(
        cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, cuda_stream));
  }

  return base::kStatusCodeOk;
}

base::Status CudaDevice::download(void *src, void *dst, size_t size,
                                  Stream *stream) {
  if (stream == nullptr) {
    NNDEPLOY_CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
  } else {
    cudaStream_t cuda_stream =
        (cudaStream_t)stream->as<CudaStream>()->getStream();
    NNDEPLOY_CUDA_CHECK(
        cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, cuda_stream));
  }

  return base::kStatusCodeOk;
}

base::Status CudaDevice::upload(void *src, void *dst, size_t size,
                                Stream *stream) {
  cudaStream_t cuda_stream = nullptr;
  if (stream == nullptr) {
    NNDEPLOY_CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
  } else {
    cuda_stream = (cudaStream_t)stream->as<CudaStream>()->getStream();
    NNDEPLOY_CUDA_CHECK(
        cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, cuda_stream));
  }

  return base::kStatusCodeOk;
}

base::Status CudaDevice::copy(Buffer *src, Buffer *dst, Stream *stream) {
  size_t dst_size = dst->getSize();
  size_t src_size = src->getSize();
  size_t size = std::min(dst_size, src_size);
  return this->copy(src->getData(), dst->getData(), size, stream);
}

base::Status CudaDevice::download(Buffer *src, Buffer *dst, Stream *stream) {
  size_t dst_size = dst->getSize();
  size_t src_size = src->getSize();
  size_t size = std::min(dst_size, src_size);
  return this->download(src->getData(), dst->getData(), size, stream);
}

base::Status CudaDevice::upload(Buffer *src, Buffer *dst, Stream *stream) {
  size_t dst_size = dst->getSize();
  size_t src_size = src->getSize();
  size_t size = std::min(dst_size, src_size);
  return this->upload(src->getData(), dst->getData(), size, stream);
}

void *CudaDevice::getContext() { return nullptr; }

base::Status CudaDevice::init() { return base::kStatusCodeOk; }
base::Status CudaDevice::deinit() { return base::kStatusCodeOk; }

Stream *CudaDevice::createStream() { return new CudaStream(this); }

Stream *CudaDevice::createStream(void *stream) {
  return new CudaStream(this, stream);
}

base::Status CudaDevice::deleteStream(Stream *stream) {
  if (stream == nullptr) {
    NNDEPLOY_LOGE("stream is nullptr\n");
    return base::kStatusCodeOk;
  }
  delete stream;
  stream = nullptr;
  return base::kStatusCodeOk;
}

Event *CudaDevice::createEvent() { return new CudaEvent(this); }
base::Status CudaDevice::destroyEvent(Event *event) {
  if (event == nullptr) {
    NNDEPLOY_LOGE("event is nullptr\n");
    return base::kStatusCodeErrorDeviceCuda;
  }
  delete event;
  event = nullptr;
  return base::kStatusCodeOk;
}
base::Status CudaDevice::createEvents(Event **events, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    Event *event = this->createEvent();
    if (event == nullptr) {
      return base::kStatusCodeErrorDeviceCuda;
    }
  }
  return base::kStatusCodeOk;
}
base::Status CudaDevice::destroyEvents(Event **events, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    this->destroyEvent(events[i]);
  }
  return base::kStatusCodeOk;
}

CudaStream::CudaStream(Device *device) : Stream(device) {
  cudaStream_t stream;
  cudaError_t ret = cudaStreamCreate(&stream);
  if (ret != cudaSuccess) {
    NNDEPLOY_LOGE("cudaStreamCreate failed, errorCode is %d", ret);
  }
  stream_ = stream;
}
CudaStream::CudaStream(Device *device, void *stream)
    : Stream(device, stream), stream_((cudaStream_t)stream) {}
CudaStream::~CudaStream() {
  if (is_external_) {
    return;
  }
  cudaError_t ret = cudaStreamDestroy(stream_);
  if (ret != cudaSuccess) {
    NNDEPLOY_LOGE("cudaStreamDestroy failed, errorCode is %d", ret);
  }
}

base::Status CudaStream::synchronize() {
  cudaError_t ret = cudaStreamSynchronize(stream_);
  if (ret != cudaSuccess) {
    NNDEPLOY_LOGE("cudaStreamSynchronize failed, errorCode is %d", ret);
  }
  return base::kStatusCodeOk;
}
base::Status CudaStream::recordEvent(Event *event) {
  if (event == nullptr) {
    NNDEPLOY_LOGE("event is nullptr\n");
    return base::kStatusCodeErrorDeviceCuda;
  }
  cudaError_t ret =
      cudaEventRecord(event->as<CudaEvent>()->getEvent(), stream_);
  if (ret != cudaSuccess) {
    NNDEPLOY_LOGE("cudaRecordStream failed, errorCode is %d", ret);
  }
  return base::kStatusCodeOk;
}
base::Status CudaStream::waitEvent(Event *event) {
  if (event == nullptr) {
    NNDEPLOY_LOGE("event is nullptr\n");
    return base::kStatusCodeErrorDeviceCuda;
  }
  cudaError_t ret =
      cudaStreamWaitEvent(stream_, event->as<CudaEvent>()->getEvent(), 0);
  if (ret != cudaSuccess) {
    NNDEPLOY_LOGE("cudaStreamWaitEvent failed, errorCode is %d", ret);
  }
  return base::kStatusCodeOk;
}

cudaStream_t CudaStream::getStream() { return stream_; }

// event
CudaEvent::CudaEvent(Device *device) : Event(device) {
  cudaError_t ret = cudaEventCreate(&event_);
  if (ret != cudaSuccess) {
    NNDEPLOY_LOGE("cudaEventCreate failed, errorCode is %d", ret);
  }
}
CudaEvent::~CudaEvent() {
  cudaError_t ret = cudaEventDestroy(event_);
  if (ret != cudaSuccess) {
    NNDEPLOY_LOGE("cudaEventDestroy failed, errorCode is %d", ret);
  }
}

bool CudaEvent::queryDone() {
  cudaError_t ret = cudaEventQuery(event_);
  if (ret != cudaSuccess) {
    NNDEPLOY_LOGE("cudaEventQuery failed, errorCode is %d", ret);
    return false;
  } else {
    return true;
  }
}
base::Status CudaEvent::synchronize() {
  cudaError_t ret = cudaEventSynchronize(event_);
  if (ret != cudaSuccess) {
    NNDEPLOY_LOGE("cudaEventSynchronize failed, errorCode is %d", ret);
  }
  return base::kStatusCodeOk;
}

cudaEvent_t CudaEvent::getEvent() { return event_; }

}  // namespace device
}  // namespace nndeploy
