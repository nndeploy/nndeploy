
#include "nncore/include/device/device.h"

#include "nncore/include/base/log.h"
#include "nncore/include/base/macro.h"
#include "nncore/include/base/object.h"
#include "nncore/include/base/status.h"
#include "nncore/include/base/type.h"
#include "nncore/include/device/buffer.h"
#include "nncore/include/device/memory_pool.h"


namespace nncore {
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

MemoryPool* Device::createMemoryPool(base::MemoryPoolType memory_pool_type) {
  NNCORE_LOGI("this device[%d, %d] can't createMemoryPool!\n",
                device_type_.code_, device_type_.device_id_);
  return nullptr;
}

MemoryPool* Device::createMemoryPool(base::MemoryPoolType memory_pool_type,
                                     size_t limit_size) {
  NNCORE_LOGI("this device[%d, %d] can't createMemoryPool!\n",
                device_type_.code_, device_type_.device_id_);
  return nullptr;
}

MemoryPool* Device::createMemoryPool(base::MemoryPoolType memory_pool_type,
                                     Buffer* buffer) {
  NNCORE_LOGI("this device[%d, %d] can't createMemoryPool!\n",
                device_type_.code_, device_type_.device_id_);
  return nullptr;
}

base::Status Device::synchronize() {
  NNCORE_LOGI("this device[%d, %d] can't synchronize!\n", device_type_.code_,
                device_type_.device_id_);
  return base::NNCORE_OK;
}

void* Device::getCommandQueue() {
  NNCORE_LOGI("this device[%d, %d] can't getCommandQueue!\n",
                device_type_.code_, device_type_.device_id_);
  return nullptr;
}

base::DeviceType Device::getDeviceType() { return device_type_; }

}  // namespace device
}  // namespace nncore