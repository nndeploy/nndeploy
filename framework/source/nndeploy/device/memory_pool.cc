
#include "nndeploy/device/memory_pool.h"

#include "nndeploy/device/buffer.h"

namespace nndeploy {
namespace device {

MemoryPool::MemoryPool(Device *device, base::MemoryPoolType memory_pool_type)
    : device_(device), memory_pool_type_(memory_pool_type) {}
MemoryPool::~MemoryPool() {}

base::Status MemoryPool::init() {
  NNDEPLOY_LOGI("this memory_pool_type[%d] can't init!\n", memory_pool_type_);
  return base::kStatusCodeOk;
}
base::Status MemoryPool::init(size_t size) {
  NNDEPLOY_LOGI("this memory_pool_type[%d] can't init!\n", memory_pool_type_);
  return base::kStatusCodeOk;
}
base::Status MemoryPool::init(void *ptr, size_t size) {
  NNDEPLOY_LOGI("this memory_pool_type[%d] can't init!\n", memory_pool_type_);
  return base::kStatusCodeOk;
}
base::Status MemoryPool::init(Buffer *buffer) {
  NNDEPLOY_LOGI("this memory_pool_type[%d] can't init!\n", memory_pool_type_);
  return base::kStatusCodeOk;
}

Device *MemoryPool::getDevice() { return device_; }
base::MemoryPoolType MemoryPool::getMemoryPoolType() {
  return memory_pool_type_;
}

}  // namespace device
}  // namespace nndeploy
