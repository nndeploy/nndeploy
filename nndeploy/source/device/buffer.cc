
#include "nndeploy/include/device/buffer.h"

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/device/buffer_pool.h"
#include "nndeploy/include/device/device.h"

namespace nndeploy {
namespace device {

// get
bool Buffer::empty() { return desc_.size_.empty(); }

base::DeviceType Buffer::getDeviceType() {
  if (device_) {
    return device_->getDeviceType();
  }

  if (buffer_pool_) {
    return buffer_pool_->getDevice()->getDeviceType();
  }

  base::DeviceType device_type(base::kDeviceTypeCodeNone, -1);

  return device_type;
}

Device *Buffer::getDevice() { return device_; }

BufferPool *Buffer::getBufferPool() { return buffer_pool_; }

bool Buffer::isBufferPool() {
  if (buffer_pool_ == nullptr) {
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
}  // namespace nndeploy