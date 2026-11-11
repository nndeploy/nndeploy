
#include "nndeploy/device/buffer.h"

#include "nndeploy/base/status.h"

namespace nndeploy {
namespace device {

struct Buffer::Storage {
  int refs = 1;
  std::shared_ptr<void> external_owner;
};

int Buffer::addRef() const { return NNDEPLOY_XADD(&storage_->refs, 1); }
int Buffer::subRef() const { return NNDEPLOY_XADD(&storage_->refs, -1); }

void Buffer::release() {
  if (storage_ && subRef() == 1) {
    if (data_ && memory_type_ == base::kMemoryTypeAllocate) {
      if (memory_pool_) memory_pool_->deallocate(data_);
      else if (device_) device_->deallocate(data_);
    }
    delete storage_;
  }
  clear();
}

Buffer::Buffer(Device *device, size_t size) {
  device_ = device;
  memory_pool_ = nullptr;
  desc_ = size;
  memory_type_ = base::kMemoryTypeAllocate;
  storage_ = new Storage();
  data_ = device->allocate(desc_);
}
Buffer::Buffer(Device *device, const BufferDesc &desc) {
  device_ = device;
  memory_pool_ = nullptr;
  desc_ = desc;
  memory_type_ = base::kMemoryTypeAllocate;
  storage_ = new Storage();
  data_ = device->allocate(desc_);
}

Buffer::Buffer(Device *device, size_t size, void *ptr) {
  device_ = device;
  memory_pool_ = nullptr;
  desc_ = size;
  memory_type_ = base::kMemoryTypeExternal;
  storage_ = new Storage();
  data_ = ptr;
}
Buffer::Buffer(Device *device, const BufferDesc &desc, void *ptr) {
  device_ = device;
  memory_pool_ = nullptr;
  desc_ = desc;
  memory_type_ = base::kMemoryTypeExternal;
  storage_ = new Storage();
  data_ = ptr;
}

Buffer::Buffer(Device *device, const BufferDesc &desc, std::shared_ptr<void> storage)
    : Buffer(device, desc, storage.get()) {
  storage_->external_owner = std::move(storage);
}

Buffer::Buffer(Device *device, size_t size, void *ptr,
               base::MemoryType memory_type) {
  device_ = device;
  memory_pool_ = nullptr;
  desc_ = size;
  memory_type_ = memory_type;
  storage_ = new Storage();
  data_ = ptr;
}
Buffer::Buffer(Device *device, const BufferDesc &desc, void *ptr,
               base::MemoryType memory_type) {
  device_ = device;
  memory_pool_ = nullptr;
  desc_ = desc;
  memory_type_ = memory_type;
  storage_ = new Storage();
  data_ = ptr;
}

Buffer::Buffer(MemoryPool *memory_pool, size_t size) {
  device_ = memory_pool->getDevice();
  memory_pool_ = memory_pool;
  desc_ = size;
  memory_type_ = base::kMemoryTypeAllocate;
  storage_ = new Storage();
  data_ = memory_pool->allocate(desc_);
}
Buffer::Buffer(MemoryPool *memory_pool, const BufferDesc &desc) {
  device_ = memory_pool->getDevice();
  memory_pool_ = memory_pool;
  desc_ = desc;
  memory_type_ = base::kMemoryTypeAllocate;
  storage_ = new Storage();
  data_ = memory_pool->allocate(desc_);
}
Buffer::Buffer(MemoryPool *memory_pool, size_t size, void *ptr,
               base::MemoryType memory_type) {
  device_ = memory_pool->getDevice();
  memory_pool_ = memory_pool;
  desc_ = size;
  memory_type_ = memory_type;
  storage_ = new Storage();
  data_ = ptr;
}
Buffer::Buffer(MemoryPool *memory_pool, const BufferDesc &desc, void *ptr,
               base::MemoryType memory_type) {
  device_ = memory_pool->getDevice();
  memory_pool_ = memory_pool;
  desc_ = desc;
  memory_type_ = memory_type;
  storage_ = new Storage();
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
  storage_ = buffer.storage_;
  if (storage_ != nullptr) {
    buffer.addRef();
  }
  data_ = buffer.data_;
}
Buffer &Buffer::operator=(const Buffer &buffer) {
  if (this == &buffer) {
    return *this;
  }
  release();
  device_ = buffer.device_;
  memory_pool_ = buffer.memory_pool_;
  desc_ = buffer.desc_;
  memory_type_ = buffer.memory_type_;
  storage_ = buffer.storage_;
  if (storage_ != nullptr) {
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
  storage_ = buffer.storage_;
  data_ = buffer.data_;
  buffer.clear();
}
Buffer &Buffer::operator=(Buffer &&buffer) noexcept {
  if (this == &buffer) {
    return *this;
  }
  release();
  device_ = buffer.device_;
  memory_pool_ = buffer.memory_pool_;
  desc_ = std::move(buffer.desc_);
  memory_type_ = buffer.memory_type_;
  storage_ = buffer.storage_;
  data_ = buffer.data_;
  buffer.clear();
  return *this;
}

Buffer::~Buffer() {
  release();
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
             isHostDeviceType(dst_device_type)) {
    return src_device->copy(this, dst);
  } else if (isHostDeviceType(src_device_type) &&
             !isHostDeviceType(dst_device_type)) {
    return dst_device->upload(this, dst);
  } else if (!isHostDeviceType(src_device_type) &&
             isHostDeviceType(dst_device_type)) {
    return src_device->download(this, dst);
  } else {
    NNDEPLOY_LOGE("Unsupported device type{%s->%s} for copy operation.",
                  base::deviceTypeToString(src_device_type).c_str(),
                  base::deviceTypeToString(dst_device_type).c_str());
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
base::Status Buffer::serialize(std::string &bin_str) {
  const uint64_t size = getRealSize();
  std::string result(reinterpret_cast<const char *>(&size), sizeof(size));
  if (size > result.max_size() - sizeof(size)) return base::kStatusCodeErrorInvalidValue;
  if (size) {
    if (!device_ || !data_) return base::kStatusCodeErrorNullParam;
    result.resize(sizeof(size) + size);
    char *dst = &result[sizeof(size)];
    if (isHostDeviceType(getDeviceType())) {
      std::memcpy(dst, data_, size);
    } else {
      auto status = device_->download(data_, dst, size);
      if (status != base::kStatusCodeOk) return status;
    }
  }
  bin_str.swap(result);
  return base::kStatusCodeOk;
}

#ifdef ENABLE_NNDEPLOY_SAFETENSORS_CPP
base::Status Buffer::serializeToSafetensors(
    safetensors::safetensors_t &st, const safetensors::tensor_t &tensor) {
  uint64_t buffer_size = this->getRealSize();
  size_t tensor_size = tensor.data_offsets[1] - tensor.data_offsets[0];
  if (buffer_size != tensor_size) {
    NNDEPLOY_LOGE(
        "unsupported buffers' size is different!! buffersize == %zu, "
        "tensor_size == %zu",
        buffer_size, tensor_size);
    return base::kStatusCodeErrorInvalidParam;
  }
  if (!tensor_size) {
    return base::kStatusCodeOk;
  }

  if (!isHostDeviceType(this->getDeviceType())) {
    NNDEPLOY_LOGI("unsupported now");
    return base::kStatusCodeErrorIO;
  } else {
    // copy tensor_size data to meta_data place
    // just copy out
    const void *src = static_cast<const void *>(data_);
    char *dst = reinterpret_cast<char *>(st.storage.data());
    memcpy(dst + tensor.data_offsets[0], src, tensor_size);
    return base::kStatusCodeOk;
  }
}
#endif

// 从二进制文件反序列化回buffer
base::Status Buffer::deserialize(const std::string &bin_str) {
  uint64_t size;
  if (bin_str.size() < sizeof(size)) return base::kStatusCodeErrorIO;
  std::memcpy(&size, bin_str.data(), sizeof(size));
  if (size > bin_str.size() - sizeof(size)) return base::kStatusCodeErrorIO;
  auto *host = getDefaultHostDevice();
  if (!host) return base::kStatusCodeErrorNullParam;
  Buffer replacement(host, static_cast<size_t>(size));
  if (size) {
    if (!replacement.data_) return base::kStatusCodeErrorOutOfMemory;
    std::memcpy(replacement.data_, bin_str.data() + sizeof(size), size);
  }
  *this = std::move(replacement);
  return base::kStatusCodeOk;
}

// base::Status Buffer::serializeFromSafetensors(const char *storage,
//                                                   const size_t &data_size) {

//   memory_pool_ = nullptr;
//   memory_type_ = base::kMemoryTypeMapped;
//   storage_ = new Storage();
//   desc_ = data_size;  // still_need to
//   data_ = (void *)storage;
//   return base::kStatusCodeOk;
// }

void Buffer::print(std::ostream &stream) const {
  stream << "Buffer: " << std::endl;
  stream << "device type: " << base::deviceTypeToString(this->getDeviceType())
         << std::endl;
  stream << "ref_count: " << storage_->refs << std::endl;
  desc_.print(stream);
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
  storage_ = nullptr;
  data_ = nullptr;
}

}  // namespace device
}  // namespace nndeploy
