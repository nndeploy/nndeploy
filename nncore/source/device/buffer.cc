#include "nncore/include/device/buffer.h"

#include "nncore/include/base/status.h"
#include "nncore/include/base/type.h"
#include "nncore/include/device/device.h"
#include "nncore/include/device/memory_pool.h"


namespace nncore {
namespace device {

// get
bool Buffer::empty() { return desc_.size_.empty(); }

base::DeviceType Buffer::getDeviceType() {
  if (device_) {
    return device_->getDeviceType();
  }

  if (memory_pool_) {
    return memory_pool_->getDevice()->getDeviceType();
  }
}

Device *Buffer::getDevice() { return device_; }

MemoryPool *Buffer::getMemoryPool() { return memory_pool_; }

bool Buffer::isMemoryPool() {
  if (memory_pool_ == nullptr) {
    return false;
  } else {
    return true;
  }
}

bool Buffer::isExternal() { return is_external_; }

BufferDesc Buffer::getDesc() { return desc_; }

base::MemoryBufferType Buffer::getMemoryBufferType() {
  return desc_.memory_buffer_type_;
}

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

}  // namespace device
}  // namespace nncore