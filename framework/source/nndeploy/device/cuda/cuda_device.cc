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
  cudaStream_t cuda_stream = nullptr;
  if (stream != nullptr) {
    cuda_stream = (cudaStream_t)stream->as<CudaStream>()->getStream();
  }
  NNDEPLOY_CUDA_CHECK(
      cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, cuda_stream));
  return base::kStatusCodeOk;
}

base::Status CudaDevice::download(void *src, void *dst, size_t size,
                                  Stream *stream) {
  cudaStream_t cuda_stream = nullptr;
  if (stream != nullptr) {
    cuda_stream = (cudaStream_t)stream->as<CudaStream>()->getStream();
  }
  NNDEPLOY_CUDA_CHECK(
      cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, cuda_stream));
  return base::kStatusCodeOk;
}

base::Status CudaDevice::upload(void *src, void *dst, size_t size,
                                Stream *stream) {
  cudaStream_t cuda_stream = nullptr;
  if (stream != nullptr) {
    cuda_stream = (cudaStream_t)stream->as<CudaStream>()->getStream();
  }
  NNDEPLOY_CUDA_CHECK(
      cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, cuda_stream));
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

int CudaDevice::newCommandQueue() {
  if (cuda_stream_wrapper_.size() == INT_MAX) {
    NNDEPLOY_LOGE("stream index is out of range\n");
    return -1;
  }
  CudaStreamWrapper cuda_stream_wrapper;
  cuda_stream_wrapper.external_command_queue_ = nullptr;
  cudaError_t status = cudaStreamCreate(&cuda_stream_wrapper.stream_);
  if (cudaSuccess != status) {
    NNDEPLOY_CUDA_CHECK(status);
    NNDEPLOY_LOGE("cuda stream create failed\n");
    return base::kStatusCodeErrorDeviceCuda;
  }
  base::Status nndp_status =
      insertStream(stream_index_, cuda_stream_wrapper_, cuda_stream_wrapper);
  NNDEPLOY_RETURN_VALUE_ON_NEQ(nndp_status, base::kStatusCodeOk, -1,
                               "insertStream failed\n");
  return stream_index_;
}
base::Status CudaDevice::deleteCommandQueue(int index) {
  // if (index < 0) {
  //   index = stream_index_;
  // }
  if (index <= 0) {
    NNDEPLOY_LOGE("index[%d] is error\n", index);
    return base::kStatusCodeErrorDeviceAscendCL;
  }
  // 更新stream_index_
  stream_index_ = updateStreamIndex(cuda_stream_wrapper_);
  if (cuda_stream_wrapper_.find(index) != cuda_stream_wrapper_.end()) {
    base::Status status = synchronize(index);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "synchronize failed");
    if (cuda_stream_wrapper_[index].external_command_queue_ == nullptr) {
      cudaError_t status =
          cudaStreamDestroy(cuda_stream_wrapper_[index].stream_);
      if (cudaSuccess != status) {
        NNDEPLOY_LOGE("cuda stream destroy failed\n");
        return base::kStatusCodeErrorDeviceCuda;
      }
    }
    cuda_stream_wrapper_.erase(index);
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("index[%d] is error\n", index);
    return base::kStatusCodeErrorDeviceCuda;
  }
}
base::Status CudaDevice::deleteCommandQueue(void *stream) {
  for (auto iter : cuda_stream_wrapper_) {
    if (stream == (void *)(iter.second.stream_)) {
      return this->deleteCommandQueue(iter.first);
    }
  }
  NNDEPLOY_LOGE("command queue is not found\n");
  return base::kStatusCodeOk;
}
int CudaDevice::setCommandQueue(void *stream, bool is_external) {
  if (stream == nullptr) {
    NNDEPLOY_LOGE("command queue is nullptr\n");
    return base::kStatusCodeErrorDeviceCuda;
  }
  CudaStreamWrapper cuda_stream_wrapper;
  if (is_external) {
    cuda_stream_wrapper.external_command_queue_ = stream;
  } else {
    cuda_stream_wrapper.external_command_queue_ = nullptr;
  }
  cuda_stream_wrapper.stream_ = (cudaStream_t)stream;
  base::Status status =
      insertStream(stream_index_, cuda_stream_wrapper_, cuda_stream_wrapper);
  NNDEPLOY_RETURN_VALUE_ON_NEQ(status, base::kStatusCodeOk, -1,
                               "insertStream failed\n");
  return stream_index_;
}

void *CudaDevice::getCommandQueue(int index) {
  // if (index < 0) {
  //   index = stream_index_;
  // }
  if (cuda_stream_wrapper_.find(index) == cuda_stream_wrapper_.end()) {
    NNDEPLOY_LOGE("getCommandQueue failed\n");
    return nullptr;
  } else {
    return (void *)cuda_stream_wrapper_[index].stream_;
  }
}

base::Status CudaDevice::synchronize(int index) {
  cudaStream_t stream = (cudaStream_t)(this->getCommandQueue(index));
  cudaError_t status = cudaStreamSynchronize(stream);
  if (cudaSuccess != status) {
    NNDEPLOY_LOGE("cuda stream synchronize failed\n");
    return base::kStatusCodeErrorDeviceCuda;
  }
  return base::kStatusCodeOk;
}

base::Status CudaDevice::init() {
  if (cuda_stream_wrapper_[0].external_command_queue_ == nullptr) {
    NNDEPLOY_CUDA_CHECK(cudaSetDevice(device_type_.device_id_));
    NNDEPLOY_CUDA_CHECK(cudaStreamCreate(&cuda_stream_wrapper_[0].stream_));
  }
  return base::kStatusCodeOk;
}
base::Status CudaDevice::deinit() {
  for (auto iter : cuda_stream_wrapper_) {
    cudaError_t status = cudaStreamSynchronize(iter.second.stream_);
    if (cudaSuccess != status) {
      NNDEPLOY_CUDA_CHECK(status);
      NNDEPLOY_LOGE("cuda stream synchronize failed\n");
      return base::kStatusCodeErrorDeviceCuda;
    }
    if (iter.second.external_command_queue_ == nullptr) {
      NNDEPLOY_CUDA_CHECK(cudaStreamDestroy(iter.second.stream_));
    }
  }
  return base::kStatusCodeOk;
}

Stream *CudaDevice::createStream() {
  cudaStream_t stream;
  NNDEPLOY_CUDA_CHECK(cudaStreamCreate(&stream));
  return new CudaStream(this, stream);
}

Stream *CudaDevice::createStream(void *stream) {
  return new CudaStream(this, stream);
}

base::Status CudaDevice::deleteStream(Stream *stream) {
  if (stream == nullptr) {
    return base::kStatusCodeOk;
  }
  if (!stream->isExternal()) {
    NNDEPLOY_CUDA_CHECK(
        cudaStreamDestroy((cudaStream_t)stream->as<CudaStream>()->getStream()));
  }
  delete stream;
  return base::kStatusCodeOk;
}

}  // namespace device
}  // namespace nndeploy