
#include "nndeploy/device/buffer_pool.h"

#include "nndeploy/device/buffer.h"

namespace nndeploy {
namespace device {

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

Buffer* BufferPool::create(size_t size, void* ptr,
                           BufferSourceType buffer_source_type) {
  BufferDesc desc;
  desc.size_.push_back(size);
  Buffer* buffer = new Buffer(this, desc, ptr, buffer_source_type);
  return buffer;
}

Buffer* BufferPool::create(const BufferDesc& desc, void* ptr,
                           BufferSourceType buffer_source_type) {
  Buffer* buffer = new Buffer(this, desc, ptr, buffer_source_type);
  return buffer;
}

Buffer* BufferPool::create(size_t size, int id,
                           BufferSourceType buffer_source_type) {
  BufferDesc desc;
  desc.size_.push_back(size);
  Buffer* buffer = new Buffer(this, desc, id, buffer_source_type);
  return buffer;
}

Buffer* BufferPool::create(const BufferDesc& desc, int id,
                           BufferSourceType buffer_source_type) {
  Buffer* buffer = new Buffer(this, desc, id, buffer_source_type);
  return buffer;
}

Device* BufferPool::getDevice() { return device_; }
base::BufferPoolType BufferPool::getBufferPoolType() {
  return buffer_pool_type_;
}

void BufferPool::destory(Buffer* buffer) { delete buffer; }

}  // namespace device
}  // namespace nndeploy
