
#include "nndeploy/device/buffer.h"

#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"

namespace nndeploy {
namespace device {

BufferDesc::BufferDesc() {}
BufferDesc::BufferDesc(size_t size) { size_.emplace_back(size); }
BufferDesc::BufferDesc(size_t *size, size_t len) {
  for (int i = 0; i < len; ++i) {
    size_.emplace_back(size[i]);
  }
}
BufferDesc::BufferDesc(const base::SizeVector &size,
                       const base::IntVector &config)
    : size_(size), config_(config) {}
BufferDesc::BufferDesc(size_t *size, size_t len, const base::IntVector &config)
    : config_(config) {
  for (int i = 0; i < len; ++i) {
    size_.emplace_back(size[i]);
  }
}

BufferDesc::BufferDesc(const BufferDesc &desc) {
  size_ = desc.size_;
  config_ = desc.config_;
}
BufferDesc::BufferDesc &operator=(const BufferDesc &desc) {
  size_ = std::move(desc.size_);
  config_ = std::move(desc.config_);
  size_ = desc.size_;
  config_ = desc.config_;
  return *this;
}

BufferDesc::BufferDesc(BufferDesc &&desc) {
  size_ = std::move(desc.size_);
  config_ = std::move(desc.config_);
}
BufferDesc::BufferDesc &operator=(BufferDesc &&desc) {
  size_ = std::move(desc.size_);
  config_ = std::move(desc.config_);
  return *this;
}

BufferDesc::~BufferDesc(){};

bool BufferDesc::isSameConfig(const BufferDesc &desc) {
  if (config_.size() != desc.config_.size()) {
    return false;
  }
  for (int i = 0; i < config_.size(); ++i) {
    if (config_[i] != desc.config_[i]) {
      return false;
    }
  }
  return true;
}
bool BufferDesc::isSameDim(const BufferDesc &desc) {
  if (size_.size() == desc.size_.size()) {
    return true;
  } else {
    return false;
  }
}
bool BufferDesc::is1D() { return size_.size() == 1; }

bool BufferDesc::operator>(const BufferDesc &other) {
  bool flag = true;
  for (int i = 0; i < size_.size(); ++i) {
    if (size_[i] <= other.size_[i]) {
      flag = false;
      break;
    }
  }
  return flag;
}
bool BufferDesc::operator>=(const BufferDesc &other) {
  bool flag = true;
  for (int i = 0; i < size_.size(); ++i) {
    if (size_[i] < other.size_[i]) {
      flag = false;
      break;
    }
  }
  return flag;
}
bool BufferDesc::operator==(const BufferDesc &other) {
  bool flag = true;
  for (int i = 0; i < size_.size(); ++i) {
    if (size_[i] != other.size_[i]) {
      flag = false;
      break;
    }
  }
  return flag;
}
bool BufferDesc::operator!=(const BufferDesc &other) {
  return !(*this == other);
}

Buffer::Buffer(Device *device, const BufferDesc &desc, void *ptr,
               BufferSourceType buffer_source_type)
    : device_(device),
      memory_pool_(nullptr),
      desc_(desc),
      data_ptr_(ptr),
      data_id_(-1),
      buffer_source_type_(buffer_source_type),
      ref_count_(1) {}
Buffer::Buffer(Device *device, const BufferDesc &desc, int id,
               BufferSourceType buffer_source_type)
    : device_(device),
      memory_pool_(nullptr),
      desc_(desc),
      data_ptr_(nullptr),
      data_id_(id),
      buffer_source_type_(buffer_source_type),
      ref_count_(1) {}
Buffer::Buffer(MemoryPool *memory_pool, const BufferDesc &desc, void *ptr,
               BufferSourceType buffer_source_type)
    : device_(memory_pool->getDevice()),
      memory_pool_(memory_pool),
      desc_(desc),
      data_ptr_(ptr),
      data_id_(-1),
      buffer_source_type_(buffer_source_type),
      ref_count_(1) {}
Buffer::Buffer(MemoryPool *memory_pool, const BufferDesc &desc, int id,
               BufferSourceType buffer_source_type)
    : device_(memory_pool->getDevice()),
      memory_pool_(memory_pool),
      desc_(desc),
      data_ptr_(nullptr),
      data_id_(id),
      buffer_source_type_(buffer_source_type),
      ref_count_(1) {}

Buffer::Buffer(Device *device, const BufferDesc &desc) {
  device_ = device;
  memory_pool_ = nullptr;
  desc_ = desc;
  data_ = device->allocate(desc);
  buffer_source_type_ = kBufferSourceTypeAllocate;
  ref_count_ = 1;
}
Buffer::Buffer(Device *device, const BufferDesc &desc);
Buffer::Buffer(MemoryPool *memory_pool, const BufferDesc &desc);
Buffer::Buffer(MemoryPool *memory_pool, const BufferDesc &desc);
Buffer::Buffer(Device *device, const BufferDesc &desc, void *ptr);
Buffer::Buffer(Device *device, const BufferDesc &desc, int id);

Buffer::Buffer(const Buffer &buffer);
Buffer &Buffer::operator=(const Buffer &buffer);

Buffer::Buffer(Buffer &&buffer);
Buffer &Buffer::operator=(Buffer &&buffer);

Buffer::~Buffer() {
  if (this->subRef() == 1 && buffer_source_type_ == kBufferSourceTypeAllocate) {
    if (memory_pool_ != nullptr) {
      if (data_ != nullptr) {
        memory_pool_->deallocate(data_);
      }
    } else {
      Device *device = device_;
      device->deallocate(data_);
    }
  }
};

// get
bool Buffer::empty() { return false; }

base::DeviceType Buffer::getDeviceType() { return device_->getDeviceType(); }

Device *Buffer::getDevice() { return device_; }

MemoryPool *Buffer::getMemoryPool() { return memory_pool_; }

bool Buffer::isMemoryPool() {
  if (memory_pool_ == nullptr) {
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

void *Buffer::getData() { return data_; }

BufferSourceType Buffer::getBufferSourceType() { return buffer_source_type_; }

void destoryBuffer(Buffer *buffer) {
  if (buffer->isMemoryPool()) {
    MemoryPool *pool = buffer->getMemoryPool();
    pool->deallocate(buffer);
  } else {
    Device *device = buffer->getDevice();
    device->deallocate(buffer);
  }
}

base::Status deepCopyBuffer(Buffer *src, Buffer *dst) {
  Device *src_device = src->getDevice();
  base::DeviceType src_device_type = src_device->getDeviceType();
  Device *dst_device = dst->getDevice();
  base::DeviceType dst_device_type = dst_device->getDeviceType();
  if (src_device_type == dst_device_type) {
    return src_device->copy(src, dst);
  } else if (isHostDeviceType(src_device_type) &&
             !isHostDeviceType(dst_device_type)) {
    return src_device->upload(src, dst);
  } else if (!isHostDeviceType(src_device_type) &&
             isHostDeviceType(dst_device_type)) {
    return src_device->download(src, dst);
  } else {
    return base::kStatusCodeErrorNotImplement;
  }
}

Buffer *getDeepCopyBuffer(Buffer *src) {
  Device *src_device = src->getDevice();
  base::DeviceType src_device_type = src_device->getDeviceType();
  Buffer *dst = src_device->allocate(src->getDesc());
  src_device->copy(src, dst);
  return dst;
}

}  // namespace device
}  // namespace nndeploy