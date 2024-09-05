
#include "nndeploy/device/cuda/cuda_device.h"

#include "nndeploy/device/buffer.h"
#include "nndeploy/device/cuda/cuda_util.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace device {

TypeArchitectureRegister<CudaArchitecture> cuda_architecture_register(
    base::kDeviceTypeCodeCuda);

CudaArchitecture::CudaArchitecture(base::DeviceTypeCode device_type_code)
    : Architecture(device_type_code){};

CudaArchitecture::~CudaArchitecture() {
  for (auto iter : devices_) {
    CudaDevice *tmp_device = dynamic_cast<CudaDevice *>(iter.second);
    if (tmp_device->deinit() != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("device deinit failed");
    }
    delete tmp_device;
  }
};

base::Status CudaArchitecture::checkDevice(int device_id, void *command_queue,
                                           std::string library_path) {
  int device_count = cudaGetNumDevices();
  if (device_id > -1 && device_id < device_count) {
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("device id is invalid, device id: %d, device count: %d",
                  device_id, device_count);
    return base::kStatusCodeErrorDeviceCuda;
  }
}

base::Status CudaArchitecture::enableDevice(int device_id, void *command_queue,
                                            std::string library_path) {
  base::DeviceType device_type(base::kDeviceTypeCodeCuda, device_id);
  std::lock_guard<std::mutex> lock(mutex_);
  if (devices_.find(device_id) == devices_.end()) {
    CudaDevice *device =
        new CudaDevice(device_type, command_queue, library_path);
    if (device == nullptr) {
      NNDEPLOY_LOGE("device is nullptr");
      return base::kStatusCodeErrorOutOfMemory;
    }

    if (device->init() != base::kStatusCodeOk) {
      delete device;
      NNDEPLOY_LOGE("device init failed");
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
    base::Status status = this->enableDevice(device_id, nullptr, "");
    if (status == base::kStatusCodeOk) {
      device = devices_[device_id];
    } else {
      NNDEPLOY_LOGE("enable device failed");
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

void *CudaDevice::allocate(size_t size) {
  BufferDesc desc(size);
  return this->allocate(desc);
}
void *CudaDevice::allocate(const BufferDesc &desc) {
  void *data = nullptr;
  cudaError_t status = cudaMalloc(&data, desc.size_[0]);
  if (cudaSuccess != status) {
    NNDEPLOY_LOGE("cuda alloc failed with size %lu for %p, status:%d\n",
                  desc.size_[0], data, status);
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

base::Status CudaDevice::copy(void *src, void *dst, size_t size, int index) {
  cudaStream_t stream = (cudaStream_t)(this->getCommandQueue(index));
  if (src != nullptr && dst != nullptr) {
    cudaError_t status =
        cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
    NNDEPLOY_CUDA_CHECK(status);
    NNDEPLOY_CUDA_CHECK(cudaStreamSynchronize(stream));
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("copy buffer failed");
    return base::kStatusCodeErrorOutOfMemory;
  }
}
base::Status CudaDevice::download(void *src, void *dst, size_t size,
                                  int index) {
  cudaStream_t stream = (cudaStream_t)(this->getCommandQueue(index));
  if (src != nullptr && dst != nullptr) {
    cudaError_t status =
        cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
    NNDEPLOY_CUDA_CHECK(status);
    NNDEPLOY_CUDA_CHECK(cudaStreamSynchronize(stream));
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("copy buffer failed");
    return base::kStatusCodeErrorOutOfMemory;
  }
}
base::Status CudaDevice::upload(void *src, void *dst, size_t size, int index) {
  cudaStream_t stream = (cudaStream_t)(this->getCommandQueue(index));
  if (src != nullptr && dst != nullptr) {
    cudaError_t status =
        cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
    NNDEPLOY_CUDA_CHECK(status);
    NNDEPLOY_CUDA_CHECK(cudaStreamSynchronize(stream));
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("copy buffer failed");
    return base::kStatusCodeErrorOutOfMemory;
  }
}

base::Status CudaDevice::copy(Buffer *src, Buffer *dst, int index) {
  cudaStream_t stream = (cudaStream_t)(this->getCommandQueue(index));
  if (src != nullptr && dst != nullptr && dst->getDesc() >= src->getDesc()) {
    cudaError_t status =
        cudaMemcpyAsync(dst->getData(), src->getData(), src->getDesc().size_[0],
                        cudaMemcpyDeviceToDevice, stream);
    NNDEPLOY_CUDA_CHECK(status);
    NNDEPLOY_CUDA_CHECK(cudaStreamSynchronize(stream));
    return base::kStatusCodeOk;
  } else {
    // TODO: add log
    return base::kStatusCodeErrorOutOfMemory;
  }
}
base::Status CudaDevice::download(Buffer *src, Buffer *dst, int index) {
  cudaStream_t stream = (cudaStream_t)(this->getCommandQueue(index));
  if (src != nullptr && dst != nullptr && dst->getDesc() >= src->getDesc()) {
    cudaError_t status =
        cudaMemcpyAsync(dst->getData(), src->getData(), src->getDesc().size_[0],
                        cudaMemcpyDeviceToHost, stream);
    NNDEPLOY_CUDA_CHECK(status);
    NNDEPLOY_CUDA_CHECK(cudaStreamSynchronize(stream));
    return base::kStatusCodeOk;
  } else {
    // TODO: add log
    return base::kStatusCodeErrorOutOfMemory;
  }
}
base::Status CudaDevice::upload(Buffer *src, Buffer *dst, int index) {
  cudaStream_t stream = (cudaStream_t)(this->getCommandQueue(index));
  if (src != nullptr && dst != nullptr && dst->getDesc() >= src->getDesc()) {
    cudaError_t status =
        cudaMemcpyAsync(dst->getData(), src->getData(), src->getDesc().size_[0],
                        cudaMemcpyHostToDevice, stream);

    NNDEPLOY_CUDA_CHECK(status);
    NNDEPLOY_CUDA_CHECK(cudaStreamSynchronize(stream));
    return base::kStatusCodeOk;
  } else {
    // TODO: add log
    return base::kStatusCodeErrorOutOfMemory;
  }
}

void *CudaDevice::getContext() { return nullptr; }

base::Status CudaDevice::newCommandQueue() {
  CudaStreamWrapper cuda_stream_wrapper;
  cuda_stream_wrapper.external_command_queue_ = nullptr;
  cudaError_t status = cudaStreamCreate(&cuda_stream_wrapper.stream_);
  if (cudaSuccess != status) {
    NNDEPLOY_CUDA_CHECK(status);
    NNDEPLOY_LOGE("cuda stream create failed\n");
    return base::kStatusCodeErrorDeviceCuda;
  }
  cuda_stream_wrapper_.emplace_back(cuda_stream_wrapper);
  return base::kStatusCodeOk;
}
base::Status CudaDevice::deleteCommandQueue(int index) {
  if (index < 0) {
    index = cuda_stream_wrapper_.size() - 1;
  }
  if (index < cuda_stream_wrapper_.size()) {
    base::Status status = synchronize(index);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "synchronize failed");
    if (cuda_stream_wrapper_[index].external_command_queue_ == nullptr) {
      cudaError_t status =
          cudaStreamDestroy(cuda_stream_wrapper_[index].stream_);
      if (cudaSuccess != status) {
        NNDEPLOY_CUDA_CHECK(status);
        NNDEPLOY_LOGE("cuda stream destroy failed\n");
        return base::kStatusCodeErrorDeviceCuda;
      }
    }
    cuda_stream_wrapper_.erase(cuda_stream_wrapper_.begin() + index);
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("index[%d] is error\n", index);
    return base::kStatusCodeErrorDeviceCuda;
  }
}
base::Status CudaDevice::deleteCommandQueue(void *command_queue) {
  for (int i = 0; i < cuda_stream_wrapper_.size(); ++i) {
    if (command_queue == (void *)(cuda_stream_wrapper_[i].stream_)) {
      return this->deleteCommandQueue(i);
    }
  }
  NNDEPLOY_LOGE("command queue is not found\n");
  return base::kStatusCodeOk;
}
base::Status CudaDevice::setCommandQueue(void *command_queue) {
  if (command_queue == nullptr) {
    NNDEPLOY_LOGE("command queue is nullptr\n");
    return base::kStatusCodeErrorDeviceCuda;
  }
  CudaStreamWrapper cuda_stream_wrapper;
  cuda_stream_wrapper.external_command_queue_ = command_queue;
  cuda_stream_wrapper.stream_ = (cudaStream_t)command_queue;
  cuda_stream_wrapper_.emplace_back(cuda_stream_wrapper);
  return base::kStatusCodeOk;
}

void *CudaDevice::getCommandQueue(int index) {
  if (index < 0) {
    index = cuda_stream_wrapper_.size() - 1;
  }
  if (index < cuda_stream_wrapper_.size()) {
    return (void *)(cuda_stream_wrapper_[index].stream_);
  } else {
    return nullptr;
  }
}

base::Status CudaDevice::synchronize(int index) {
  cudaStream_t stream = (cudaStream_t)(this->getCommandQueue(index));
  cudaError_t status = cudaStreamSynchronize(stream);
  if (cudaSuccess != status) {
    NNDEPLOY_CUDA_CHECK(status);
    NNDEPLOY_LOGE("cuda stream synchronize failed\n");
    return base::kStatusCodeErrorDeviceCuda;
  }
  return base::kStatusCodeOk;
}

base::Status CudaDevice::init() {
  if (cuda_stream_wrapper_[0].external_command_queue_ == nullptr) {
    NNDEPLOY_CUDA_CHECK(cudaSetDevice(device_type_.device_id_));
    NNDEPLOY_CUDA_CHECK(cudaStreamCreate(&cuda_stream_wrapper_[0].stream_));
  } else {
    cuda_stream_wrapper_[0].stream_ =
        (cudaStream_t)(cuda_stream_wrapper_[0].external_command_queue_);
  }
  return base::kStatusCodeOk;
}
base::Status CudaDevice::deinit() {
  for (auto iter : cuda_stream_wrapper_) {
    cudaError_t status = cudaStreamSynchronize(iter.stream_);
    if (cudaSuccess != status) {
      NNDEPLOY_CUDA_CHECK(status);
      NNDEPLOY_LOGE("cuda stream synchronize failed\n");
      return base::kStatusCodeErrorDeviceCuda;
    }
    if (iter.external_command_queue_ == nullptr) {
      NNDEPLOY_CUDA_CHECK(cudaStreamDestroy(iter.stream_));
    }
  }
  return base::kStatusCodeOk;
}

}  // namespace device
}  // namespace nndeploy