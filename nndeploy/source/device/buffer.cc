
#include "nndeploy/include/device/buffer.h"

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/device/buffer_pool.h"
#include "nndeploy/include/device/device.h"

namespace nndeploy {
namespace device {

// get
bool Buffer::empty() { return desc_.size_.empty(); }

base::DeviceType Buffer::getDeviceType() { return device_->getDeviceType(); }

Device *Buffer::getDevice() { return device_; }

BufferPool *Buffer::getBufferPool() { return buffer_pool_; }

bool Buffer::isBufferPool() {
  if (buffer_pool_ == nullptr) {
    return false;
  } else {
    return true;
  }
}

BufferDesc Buffer::getDesc() { return desc_; }

size_t Buffer::getSize() {
  if (desc_.size_.empty()) {
    return 0;
  }
  size_t size = 1;
  for (auto iter : desc_.size_) {
    size *= iter;
  }
  return size;
}

base::SizeVector Buffer::getSizeVector() { return desc_.size_; }

base::IntVector Buffer::getConfig() { return desc_.config_; }

void *Buffer::getPtr() { return data_ptr_; }

int32_t Buffer::getId() { return data_id_; }

BufferSourceType Buffer::getBufferSourceType() { return buffer_source_type_; }

int32_t Buffer::getRef() { return ref_count_; }

void Buffer::addRef() { ref_count_++; }

void Buffer::subRef() { ref_count_--; }

Buffer::Buffer(Device *device, const BufferDesc &desc, void *ptr,
               BufferSourceType buffer_source_type)
    : device_(device),
      buffer_pool_(nullptr),
      desc_(desc),
      data_ptr_(ptr),
      data_id_(-1),
      buffer_source_type_(buffer_source_type),
      ref_count_(1) {}
Buffer::Buffer(Device *device, const BufferDesc &desc, int32_t id,
               BufferSourceType buffer_source_type)
    : device_(device),
      buffer_pool_(nullptr),
      desc_(desc),
      data_ptr_(nullptr),
      data_id_(id),
      buffer_source_type_(buffer_source_type),
      ref_count_(1) {}
Buffer::Buffer(BufferPool *buffer_pool, const BufferDesc &desc, void *ptr,
               BufferSourceType buffer_source_type)
    : device_(buffer_pool->getDevice()),
      buffer_pool_(buffer_pool),
      desc_(desc),
      data_ptr_(ptr),
      data_id_(-1),
      buffer_source_type_(buffer_source_type),
      ref_count_(1) {}
Buffer::Buffer(BufferPool *buffer_pool, const BufferDesc &desc, int32_t id,
               BufferSourceType buffer_source_type)
    : device_(buffer_pool->getDevice()),
      buffer_pool_(buffer_pool),
      desc_(desc),
      data_ptr_(nullptr),
      data_id_(id),
      buffer_source_type_(buffer_source_type),
      ref_count_(1) {}

Buffer::~Buffer(){};

}  // namespace device
}  // namespace nndeploy