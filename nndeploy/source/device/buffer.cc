
#include "nndeploy/source/device/buffer.h"

#include "nndeploy/source/device/buffer_pool.h"
#include "nndeploy/source/device/device.h"

namespace nndeploy {
namespace device {

BufferDescCompareStatus compareBufferDesc(const BufferDesc &desc1,
                                          const BufferDesc &desc2) {
  bool config_equal = true;
  if (desc1.config_.size() != desc2.config_.size()) {
    config_equal = false;
  } else {
    for (size_t i = 0; i < desc1.config_.size(); i++) {
      if (desc1.config_[i] != desc2.config_[i]) {
        // config not equal
        // TODO: add log
        config_equal = false;
      }
    }
  }
  if (config_equal) {
    size_t size_1 = 1;
    size_t size_2 = 1;
    for (size_t i = 0; i < desc1.size_.size(); i++) {
      size_1 *= desc1.size_[i];
      size_2 *= desc2.size_[i];
    }
    if (size_1 < size_2) {
      return kBufferDescCompareStatusConfigEqualSizeLess;
    } else if (size_1 == size_2) {
      return kBufferDescCompareStatusConfigEqualSizeEqual;
    } else {
      return kBufferDescCompareStatusConfigEqualSizeGreater;
    }
  } else {
    if (desc1.size_.size() != desc2.size_.size()) {
      return kBufferDescCompareStatusConfigNotEqualSizeNotEqual;
    } else {
      size_t size_1 = 1;
      size_t size_2 = 1;
      for (size_t i = 0; i < desc1.size_.size(); i++) {
        size_1 *= desc1.size_[i];
        size_2 *= desc2.size_[i];
      }
      if (size_1 < size_2) {
        return kBufferDescCompareStatusConfigNotEqualSizeLess;
      } else if (size_1 == size_2) {
        return kBufferDescCompareStatusConfigNotEqualSizeEqual;
      } else {
        return kBufferDescCompareStatusConfigNotEqualSizeGreater;
      }
    }
  }
}

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

}  // namespace device
}  // namespace nndeploy