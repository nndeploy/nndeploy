
#include "nndeploy/include/device/device.h"

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/log.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/device/buffer.h"
#include "nndeploy/include/device/buffer_pool.h"

namespace nndeploy {
namespace device {

Buffer* Device::create(size_t size, void* ptr) {
  Buffer* buffer = new Buffer(this, size, ptr, true);
  return buffer;
}

Buffer* Device::create(BufferDesc& desc, void* ptr) {
  Buffer* buffer = new Buffer(this, desc, ptr, true);
  return buffer;
}

Buffer* Device::create(size_t size, int32_t id) {
  Buffer* buffer = new Buffer(this, size, id, true);
  return buffer;
}

Buffer* Device::create(BufferDesc& desc, int32_t id) {
  Buffer* buffer = new Buffer(this, desc, id, true);
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