
#include "nndeploy/source/device/x86/x86_device.h"

#include <cstring>

#include "nndeploy/source/device/device.h"

namespace nndeploy {
namespace device {

BufferDesc X86Device::toBufferDesc(const MatDesc& desc,
                                   const base::IntVector& config) {
  BufferDesc buffer_desc;
  buffer_desc.config_ = config;
  size_t size = desc.data_type_.size();
  if (desc.stride_.empty()) {
    for (int i = 0; i < desc.shape_.size(); ++i) {
      size *= desc.shape_[i];
    }
  } else {
    size = desc.stride_[0];
  }
  buffer_desc.size_.push_back(size);
  return buffer_desc;
}

/**
 * @brief
 *
 * @param desc
 * @param config
 * @return BufferDesc
 * @note:
 * 通过stride_替代了data_format_，stride_的第一个元素表示的是整个tensor的大小
 * 意味着在TensorDesc的构造函数要花很多心思来计算stride_
 */
BufferDesc X86Device::toBufferDesc(const TensorImplDesc& desc,
                                   const base::IntVector& config) {
  BufferDesc buffer_desc;
  buffer_desc.config_ = config;
  size_t size = desc.data_type_.size();
  if (desc.stride_.empty()) {
    for (int i = 0; i < desc.shape_.size(); ++i) {
      size *= desc.shape_[i];
    }
  } else {
    size = desc.stride_[0];
  }
  buffer_desc.size_.push_back(size);
  return buffer_desc;
}

Buffer* X86Device::allocate(size_t size) {
  BufferDesc desc(size);
  return this->allocate(desc);
}
Buffer* X86Device::allocate(const BufferDesc& desc) {
  void* data = malloc(desc.size_[0]);
  Buffer* buffer = Device::create(desc, data, kBufferSourceTypeAllocate);
  return buffer;
}
void X86Device::deallocate(Buffer* buffer) {
  if (buffer == nullptr) {
    return;
  }
  if (buffer->getRef() > 1) {
    return;
  }
  BufferSourceType buffer_source_type = buffer->getBufferSourceType();
  if (buffer_source_type == kBufferSourceTypeNone ||
      buffer_source_type == kBufferSourceTypeExternal) {
    Device::destory(buffer);
  } else if (buffer_source_type == kBufferSourceTypeAllocate) {
    if (buffer->getPtr() != nullptr) {
      void* data = buffer->getPtr();
      free(data);
    }
    Device::destory(buffer);
  } else if (buffer_source_type == kBufferSourceTypeMapped) {
    return;
  } else {
    return;
  }
}

base::Status X86Device::copy(Buffer* src, Buffer* dst) {
  BufferDescCompareStatus status =
      compareBufferDesc(dst->getDesc(), src->getDesc());
  if (status >= kBufferDescCompareStatusConfigEqualSizeEqual) {
    memcpy(dst->getPtr(), src->getPtr(), src->getDesc().size_[0]);
    return base::kStatusCodeOk;
  } else {
    // TODO: add log
    return base::kStatusCodeErrorOutOfMemory;
  }
}
base::Status X86Device::download(Buffer* src, Buffer* dst) {
  BufferDescCompareStatus status =
      compareBufferDesc(dst->getDesc(), src->getDesc());
  if (status >= kBufferDescCompareStatusConfigEqualSizeEqual) {
    memcpy(dst->getPtr(), src->getPtr(), src->getDesc().size_[0]);
    return base::kStatusCodeOk;
  } else {
    // TODO: add log
    return base::kStatusCodeErrorOutOfMemory;
  }
}
base::Status X86Device::upload(Buffer* src, Buffer* dst) {
  BufferDescCompareStatus status =
      compareBufferDesc(dst->getDesc(), src->getDesc());
  if (status >= kBufferDescCompareStatusConfigEqualSizeEqual) {
    memcpy(dst->getPtr(), src->getPtr(), src->getDesc().size_[0]);
    return base::kStatusCodeOk;
  } else {
    // TODO: add log
    return base::kStatusCodeErrorOutOfMemory;
  }
}

base::Status X86Device::init() { return base::kStatusCodeOk; }
base::Status X86Device::deinit() { return base::kStatusCodeOk; }

}  // namespace device
}  // namespace nndeploy