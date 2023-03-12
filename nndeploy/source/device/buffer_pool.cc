
#include "nndeploy/include/device/buffer_pool.h"

namespace nndeploy {
namespace device {

Device* BufferPool::getDevice() { return device_; }
base::BufferPoolType BufferPool::getBufferPoolType() {
  return buffer_pool_type_;
}

BufferPool::BufferPool(Device* device, base::BufferPoolType buffer_pool_type)
    : device_(device), buffer_pool_type_(buffer_pool_type) {}
BufferPool::~BufferPool() {}

base::Status BufferPool::init() {
  NNDEPLOY_LOGI("this buffer_pool_type[%d] can't init!\n", buffer_pool_type_);
  return base::kStatusCodeOk;
}
base::Status BufferPool::init(size_t limit_size) {
  NNDEPLOY_LOGI("this buffer_pool_type[%d] can't init!\n", buffer_pool_type_);
  return base::kStatusCodeOk;
}
base::Status BufferPool::init(Buffer* buffer) {
  NNDEPLOY_LOGI("this buffer_pool_type[%d] can't init!\n", buffer_pool_type_);
  return base::kStatusCodeOk;
}

}  // namespace device
}  // namespace nndeploy
