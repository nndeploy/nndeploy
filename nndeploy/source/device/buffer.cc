#include "nndeploy/include/device/buffer.h"

#include "nndeploy/include/base/status.h"
#include "nndeploy/include/base/type.h"
#include "nndeploy/include/device/device.h"
#include "nndeploy/include/device/memory_pool.h"


namespace nndeploy {
namespace device {

Buffer::~Buffer() {
  if (device_) {
    device_->free(this);
  }

  if (memory_pool_) {
    memory_pool_->free(this);
  }
}

Buffer::Buffer(size_t size, void *ptr) {
  desc_.size_.push_back(size);
  data_ptr_ = ptr;
}

Buffer::Buffer(size_t size, int32_t id) {
  desc_.size_.push_back(size);
  data_id_ = id;
}

Buffer::Buffer(BufferDesc desc, void *ptr) {
  desc_ = desc;
  data_ptr_ = ptr;
}

Buffer::Buffer(BufferDesc desc, int32_t id) {
  desc_ = desc;
  data_id_ = id;
}

Buffer::Buffer(Device *device, size_t size) {}

Buffer::Buffer(Device *device, BufferDesc desc) {}

Buffer::Buffer(MemoryPool *pool, size_t size) {}

Buffer::Buffer(MemoryPool *pool, BufferDesc desc) {}

// get
bool Buffer::empty() { return desc.size_.empty(); }

base::DeviceType Buffer::getDeviceType() {
  if (device_) {
    return device_->getDeviceType();
  }

  if (memory_pool_) {
    return memory_pool_->getDeviceType();
  }
}

Device *Buffer::getDevice() { return device_; }

MemoryPool *Buffer::getMemoryPool() { return memory_pool_; }

BufferDesc Buffer::getDesc() { return desc_; }

base::MemoryBufferType getMemoryBufferType() { return desc_.memory_type_; }

size_t getSize() {
  if (desc_.size_.empty()) {
    return 0;
  }
  size_t size = 1;
  for (auto iter : desc_.size_) {
    size *= iter;,
  }
  return size;
}

base::SizeVector getSizeVector() { return desc_.size_; }

base::IntVector getConfig() { return desc_.config_; }

void *getPtr() { return data_ptr_; }

int32_t getId()[return data_id_;]

}  // namespace device
}  // namespace nndeploy