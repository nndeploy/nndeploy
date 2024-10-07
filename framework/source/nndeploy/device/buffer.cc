
#include "nndeploy/device/buffer.h"

namespace nndeploy {
namespace device {

Buffer::Buffer(Device *device, size_t size) {
  device_ = device;
  memory_pool_ = nullptr;
  desc_ = size;
  memory_type_ = base::kMemoryTypeAllocate;
  ref_count_ = new int(1);
  data_ = device->allocate(desc_);
}
Buffer::Buffer(Device *device, const BufferDesc &desc) {
  device_ = device;
  memory_pool_ = nullptr;
  desc_ = desc;
  memory_type_ = base::kMemoryTypeAllocate;
  ref_count_ = new int(1);
  data_ = device->allocate(desc_);
}

Buffer::Buffer(Device *device, size_t size, void *ptr) {
  device_ = device;
  memory_pool_ = nullptr;
  desc_ = size;
  memory_type_ = base::kMemoryTypeExternal;
  ref_count_ = new int(1);
  data_ = ptr;
}
Buffer::Buffer(Device *device, const BufferDesc &desc, void *ptr) {
  device_ = device;
  memory_pool_ = nullptr;
  desc_ = desc;
  memory_type_ = base::kMemoryTypeExternal;
  ref_count_ = new int(1);
  data_ = ptr;
}

Buffer::Buffer(Device *device, size_t size, void *ptr,
               base::MemoryType memory_type) {
  device_ = device;
  memory_pool_ = nullptr;
  desc_ = size;
  memory_type_ = memory_type;
  ref_count_ = new int(1);
  data_ = ptr;
}
Buffer::Buffer(Device *device, const BufferDesc &desc, void *ptr,
               base::MemoryType memory_type) {
  device_ = device;
  memory_pool_ = nullptr;
  desc_ = desc;
  memory_type_ = memory_type;
  ref_count_ = new int(1);
  data_ = ptr;
}

Buffer::Buffer(MemoryPool *memory_pool, size_t size) {
  device_ = memory_pool->getDevice();
  memory_pool_ = memory_pool;
  desc_ = size;
  memory_type_ = base::kMemoryTypeAllocate;
  ref_count_ = new int(1);
  data_ = memory_pool->allocate(desc_);
}
Buffer::Buffer(MemoryPool *memory_pool, const BufferDesc &desc) {
  device_ = memory_pool->getDevice();
  memory_pool_ = memory_pool;
  desc_ = desc;
  memory_type_ = base::kMemoryTypeAllocate;
  ref_count_ = new int(1);
  data_ = memory_pool->allocate(desc_);
}
Buffer::Buffer(MemoryPool *memory_pool, size_t size, void *ptr,
               base::MemoryType memory_type) {
  device_ = memory_pool->getDevice();
  memory_pool_ = memory_pool;
  desc_ = size;
  memory_type_ = memory_type;
  ref_count_ = new int(1);
  data_ = ptr;
}
Buffer::Buffer(MemoryPool *memory_pool, const BufferDesc &desc, void *ptr,
               base::MemoryType memory_type) {
  device_ = memory_pool->getDevice();
  memory_pool_ = memory_pool;
  desc_ = desc;
  memory_type_ = memory_type;
  ref_count_ = new int(1);
  data_ = ptr;
}

Buffer::Buffer(const Buffer &buffer) {
  if (this == &buffer) {
    return;
  }
  device_ = buffer.device_;
  memory_pool_ = buffer.memory_pool_;
  desc_ = buffer.desc_;
  memory_type_ = buffer.memory_type_;
  ref_count_ = buffer.ref_count_;
  if (ref_count_ != nullptr) {
    buffer.addRef();
  }
  data_ = buffer.data_;
}
Buffer &Buffer::operator=(const Buffer &buffer) {
  if (this == &buffer) {
    return *this;
  }
  device_ = buffer.device_;
  memory_pool_ = buffer.memory_pool_;
  desc_ = buffer.desc_;
  memory_type_ = buffer.memory_type_;
  ref_count_ = buffer.ref_count_;
  if (ref_count_ != nullptr) {
    buffer.addRef();
  }
  data_ = buffer.data_;
  return *this;
}

Buffer::Buffer(Buffer &&buffer) noexcept {
  if (this == &buffer) {
    return;
  }
  device_ = buffer.device_;
  memory_pool_ = buffer.memory_pool_;
  desc_ = std::move(buffer.desc_);
  memory_type_ = buffer.memory_type_;
  ref_count_ = buffer.ref_count_;
  data_ = buffer.data_;
  buffer.clear();
}
Buffer &Buffer::operator=(Buffer &&buffer) noexcept {
  if (this == &buffer) {
    return *this;
  }
  device_ = buffer.device_;
  memory_pool_ = buffer.memory_pool_;
  desc_ = std::move(buffer.desc_);
  memory_type_ = buffer.memory_type_;
  ref_count_ = buffer.ref_count_;
  data_ = buffer.data_;
  buffer.clear();
  return *this;
}

Buffer::~Buffer() {
  if (data_ != nullptr && ref_count_ != nullptr && this->subRef() == 1) {
    if (memory_pool_ != nullptr && memory_type_ == base::kMemoryTypeAllocate) {
      if (data_ != nullptr) {
        memory_pool_->deallocate(data_);
      }
    } else {
      if (data_ != nullptr && memory_type_ == base::kMemoryTypeAllocate) {
        device_->deallocate(data_);
      }
    }
    delete ref_count_;
  }
  this->clear();
};

Buffer *Buffer::clone() {
  Buffer *dst = nullptr;
  if (memory_pool_ != nullptr) {
    dst = new Buffer(memory_pool_, this->getDesc());
  } else {
    dst = new Buffer(device_, this->getDesc());
  }
  device_->copy(this, dst);
  return dst;
}
base::Status Buffer::copyTo(Buffer *dst) {
  Device *src_device = this->getDevice();
  base::DeviceType src_device_type = src_device->getDeviceType();
  Device *dst_device = dst->getDevice();
  base::DeviceType dst_device_type = dst_device->getDeviceType();
  if (src_device_type == dst_device_type) {
    return src_device->copy(this, dst);
  } else if (isHostDeviceType(src_device_type) &&
             !isHostDeviceType(dst_device_type)) {
    return dst_device->upload(this, dst);
  } else if (!isHostDeviceType(src_device_type) &&
             isHostDeviceType(dst_device_type)) {
    return src_device->download(this, dst);
  } else {
    return base::kStatusCodeErrorNotImplement;
  }
}

bool Buffer::justModify(const size_t &size) { return desc_.justModify(size); }
bool Buffer::justModify(const base::SizeVector &size) {
  return desc_.justModify(size);
}
bool Buffer::justModify(const BufferDesc &desc) {
  return desc_.justModify(desc);
}

// 序列化buffer为二进制文件
base::Status Buffer::serialize(std::ostream &stream) {
  uint64_t buffer_size = this->getRealSize();
  if (!stream.write(reinterpret_cast<const char *>(&buffer_size),
                    sizeof(buffer_size))) {
    return base::kStatusCodeErrorIO;
  }
  if (!isHostDeviceType(this->getDeviceType())) {
    Device *host_device = getDefaultHostDevice();
    Buffer *host_buffer = new Buffer(host_device, this->getDesc());
    base::Status status = this->copyTo(host_buffer);
    if (status != base::kStatusCodeOk) {
      delete host_buffer;
      return status;
    }
    const char *data = static_cast<const char *>(host_buffer->getData());
    if (!stream.write(data, buffer_size)) {
      return base::kStatusCodeErrorIO;
    }
    delete host_buffer;
    return status;
  } else {
    const char *data = static_cast<const char *>(data_);
    if (!stream.write(data, buffer_size)) {
      return base::kStatusCodeErrorIO;
    }
    return base::kStatusCodeOk;
  }
}
// 从二进制文件反序列化回buffer
base::Status Buffer::deserialize(std::istream &stream) {
  device_ = getDefaultHostDevice();
  memory_pool_ = nullptr;
  memory_type_ = base::kMemoryTypeAllocate;
  ref_count_ = new int(1);
  uint64_t buffer_size = 0;
  if (!stream.read(reinterpret_cast<char *>(&buffer_size),
                   sizeof(buffer_size))) {
    return base::kStatusCodeErrorIO;
  }
  desc_ = buffer_size;
  data_ = device_->allocate(buffer_size);
  if (!stream.read(reinterpret_cast<char *>(data_), buffer_size)) {
    return base::kStatusCodeErrorIO;
  } else {
    return base::kStatusCodeOk;
  }
}

void Buffer::print() {
  std::cout << "Buffer: " << std::endl;
  std::cout << "device type: "
            << base::deviceTypeToString(this->getDeviceType()) << std::endl;
  std::cout << "ref_count: " << ref_count_[0] << std::endl;
  desc_.print();
}

// get
bool Buffer::empty() const { return false; }

base::DeviceType Buffer::getDeviceType() const {
  return device_->getDeviceType();
}

Device *Buffer::getDevice() const { return device_; }

MemoryPool *Buffer::getMemoryPool() const { return memory_pool_; }

bool Buffer::isMemoryPool() const {
  if (memory_pool_ == nullptr) {
    return false;
  } else {
    return true;
  }
}

BufferDesc Buffer::getDesc() const { return desc_; }

size_t Buffer::getSize() const { return desc_.getSize(); }

base::SizeVector Buffer::getSizeVector() const { return desc_.getSizeVector(); }

size_t Buffer::getRealSize() const { return desc_.getRealSize(); }

base::SizeVector Buffer::getRealSizeVector() const {
  return desc_.getRealSizeVector();
}

base::IntVector Buffer::getConfig() const { return desc_.getConfig(); }

void *Buffer::getData() const { return data_; }

base::MemoryType Buffer::getMemoryType() const { return memory_type_; }

void Buffer::clear() {
  device_ = nullptr;
  memory_pool_ = nullptr;
  desc_.clear();
  memory_type_ = base::kMemoryTypeNone;
  ref_count_ = nullptr;
  data_ = nullptr;
}

}  // namespace device
}  // namespace nndeploy