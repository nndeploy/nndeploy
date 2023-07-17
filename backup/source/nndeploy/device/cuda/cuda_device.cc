
#include "nndeploy/include/device/cuda/cuda_device.h"

#include "nndeploy/include/device/buffer.h"
#include "nndeploy/include/device/cuda/cuda_util.h"
#include "nndeploy/include/device/mat.h"
#include "nndeploy/include/device/tensor.h"

namespace nndeploy {
namespace device {

TypeArchitectureRegister<CudaArchitecture> cuda_architecture_register(
    base::kDeviceTypeCodeCuda);

CudaArchitecture::CudaArchitecture(base::DeviceTypeCode device_type_code)
    : Architecture(device_type_code){};

CudaArchitecture::~CudaArchitecture() {
  for (auto iter : devices_) {
    CudaDevice* tmp_device = dynamic_cast<CudaDevice*>(iter.second);
    if (tmp_device->deinit() != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("device deinit failed");
    }
    delete tmp_device;
  }
};

base::Status CudaArchitecture::checkDevice(int32_t device_id,
                                           void* command_queue,
                                           std::string library_path) {
  int32_t device_count = cudaGetNumDevices();
  if (device_id > 0 && device_id < device_count) {
    return base::kStatusCodeOk;
  } else {
    NNDEPLOY_LOGE("device id is invalid, device id: %d, device count: %d",
                  device_id, device_count);
    return base::kStatusCodeErrorDeviceCuda;
  }
}

base::Status CudaArchitecture::enableDevice(int32_t device_id,
                                            void* command_queue,
                                            std::string library_path) {
  base::DeviceType device_type(base::kDeviceTypeCodeCuda, device_id);
  std::lock_guard<std::mutex> lock(mutex_);
  if (devices_.find(device_id) == devices_.end()) {
    CudaDevice* device =
        new CudaDevice(device_type, command_queue, library_path);
    if (device == NULL) {
      NNDEPLOY_LOGE("device is NULL");
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

Device* CudaArchitecture::getDevice(int32_t device_id) {
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

std::vector<DeviceInfo> CudaArchitecture::getDeviceInfo(
    std::string library_path) {
  std::vector<DeviceInfo> device_info_list;
  int32_t device_count = cudaGetNumDevices();
  for (int i = 0; i < device_count; ++i) {
    cudaDeviceProp p = cudaGetDeviceProperty(i);
    DeviceInfo device_info;
    device_info_list.push_back(device_info);
  }
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
BufferDesc CudaDevice::toBufferDesc(const MatDesc& desc,
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
BufferDesc CudaDevice::toBufferDesc(const TensorDesc& desc,
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

Buffer* CudaDevice::allocate(size_t size) {
  BufferDesc desc(size);
  return this->allocate(desc);
}
Buffer* CudaDevice::allocate(const BufferDesc& desc) {
  void* data = nullptr;
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
  Buffer* buffer = Device::create(desc, data, kBufferSourceTypeAllocate);
  return buffer;
}
void CudaDevice::deallocate(Buffer* buffer) {
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
      NNDEPLOY_CUDA_CHECK(cudaFree(data));
    }
    Device::destory(buffer);
  } else if (buffer_source_type == kBufferSourceTypeMapped) {
    return;
  } else {
    return;
  }
}

base::Status CudaDevice::copy(Buffer* src, Buffer* dst) {
  BufferDescCompareStatus status =
      compareBufferDesc(dst->getDesc(), src->getDesc());
  if (status >= kBufferDescCompareStatusConfigEqualSizeEqual) {
    cudaError_t status =
        cudaMemcpyAsync(dst->getPtr(), src->getPtr(), src->getDesc().size_[0],
                        cudaMemcpyDeviceToDevice, stream_);
    NNDEPLOY_CUDA_CHECK(status);
    NNDEPLOY_CUDA_CHECK(cudaStreamSynchronize(stream_));
    return base::kStatusCodeOk;
  } else {
    // TODO: add log
    return base::kStatusCodeErrorOutOfMemory;
  }
}
base::Status CudaDevice::download(Buffer* src, Buffer* dst) {
  BufferDescCompareStatus status =
      compareBufferDesc(dst->getDesc(), src->getDesc());
  if (status >= kBufferDescCompareStatusConfigEqualSizeEqual) {
    cudaError_t status =
        cudaMemcpyAsync(dst->getPtr(), src->getPtr(), src->getDesc().size_[0],
                        cudaMemcpyDeviceToHost, stream_);
    NNDEPLOY_CUDA_CHECK(status);
    NNDEPLOY_CUDA_CHECK(cudaStreamSynchronize(stream_));
    return base::kStatusCodeOk;
  } else {
    // TODO: add log
    return base::kStatusCodeErrorOutOfMemory;
  }
}
base::Status CudaDevice::upload(Buffer* src, Buffer* dst) {
  BufferDescCompareStatus status =
      compareBufferDesc(dst->getDesc(), src->getDesc());
  if (status >= kBufferDescCompareStatusConfigEqualSizeEqual) {
    cudaError_t status =
        cudaMemcpyAsync(dst->getPtr(), src->getPtr(), src->getDesc().size_[0],
                        cudaMemcpyHostToDevice, stream_);
    NNDEPLOY_CUDA_CHECK(status);
    NNDEPLOY_CUDA_CHECK(cudaStreamSynchronize(stream_));
    return base::kStatusCodeOk;
  } else {
    // TODO: add log
    return base::kStatusCodeErrorOutOfMemory;
  }
}

base::Status CudaDevice::synchronize() {
  cudaError_t status = cudaStreamSynchronize(stream_);
  if (cudaSuccess != status) {
    NNDEPLOY_LOGE("cuda stream synchronize failed\n");
    return base::kStatusCodeErrorDeviceCuda;
  }
  return base::kStatusCodeOk;
}
void* CudaDevice::getCommandQueue() { return (void*)(stream_); }

base::Status CudaDevice::init() {
  if (external_command_queue_ == nullptr) {
    NNDEPLOY_CUDA_CHECK(cudaSetDevice(device_type_.device_id_));
    NNDEPLOY_CUDA_CHECK(cudaStreamCreate(&stream_));
  } else {
    stream_ = (cudaStream_t)(external_command_queue_);
  }
  return base::kStatusCodeOk;
}
base::Status CudaDevice::deinit() {
  if (external_command_queue_ != nullptr) {
    NNDEPLOY_CUDA_CHECK(cudaStreamDestroy(stream_));
  }
  return base::kStatusCodeOk;
}

}  // namespace device
}  // namespace nndeploy