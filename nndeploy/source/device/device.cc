
#include "nndeploy/include/device/device.h"

namespace nndeploy {
namespace device {

BufferDesc Device::toBufferDesc(const MatDesc& desc_,
                                const base::IntVector& config) {
  NNDEPLOY_LOGI("this device[%d, %d] can't toBufferDesc!\n", device_type_.code_,
                device_type_.device_id_);
  return BufferDesc();
}

BufferDesc Device::toBufferDesc(const TensorDesc& desc_,
                                const base::IntVector& config) {
  NNDEPLOY_LOGI("this device[%d, %d] can't toBufferDesc!\n", device_type_.code_,
                device_type_.device_id_);
  return BufferDesc();
}

Buffer* Device::create(size_t size, void* ptr) {
  BufferDesc desc;
  desc.size_.push_back(size);
  Buffer* buffer = new Buffer(this, desc, ptr, kBufferSourceTypeExternal);
  return buffer;
}

Buffer* Device::create(const BufferDesc& desc, void* ptr) {
  Buffer* buffer = new Buffer(this, desc, ptr, kBufferSourceTypeExternal);
  return buffer;
}

Buffer* Device::create(size_t size, int32_t id) {
  BufferDesc desc;
  desc.size_.push_back(size);
  Buffer* buffer = new Buffer(this, desc, id, kBufferSourceTypeExternal);
  return buffer;
}

Buffer* Device::create(const BufferDesc& desc, int32_t id) {
  Buffer* buffer = new Buffer(this, desc, id, kBufferSourceTypeExternal);
  return buffer;
}

BufferPool* Device::createBufferPool(base::BufferPoolType buffer_pool_type) {
  NNDEPLOY_LOGI("this device[%d, %d] can't createBufferPool!\n",
                device_type_.code_, device_type_.device_id_);
  return nullptr;
}

BufferPool* Device::createBufferPool(base::BufferPoolType buffer_pool_type,
                                     size_t limit_size) {
  NNDEPLOY_LOGI("this device[%d, %d] can't createBufferPool!\n",
                device_type_.code_, device_type_.device_id_);
  return nullptr;
}

BufferPool* Device::createBufferPool(base::BufferPoolType buffer_pool_type,
                                     Buffer* buffer) {
  NNDEPLOY_LOGI("this device[%d, %d] can't createBufferPool!\n",
                device_type_.code_, device_type_.device_id_);
  return nullptr;
}

base::Status Device::synchronize() {
  NNDEPLOY_LOGI("this device[%d, %d] can't synchronize!\n", device_type_.code_,
                device_type_.device_id_);
  return base::kStatusCodeOk;
}

void* Device::getCommandQueue() {
  NNDEPLOY_LOGI("this device[%d, %d] can't getCommandQueue!\n",
                device_type_.code_, device_type_.device_id_);
  return nullptr;
}

base::DeviceType Device::getDeviceType() { return device_type_; }

}  // namespace device
}  // namespace nndeploy