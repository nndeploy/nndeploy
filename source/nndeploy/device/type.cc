
#include "nndeploy/device/type.h"

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
  if (this == &desc) {
    return;
  }
  size_ = desc.size_;
  config_ = desc.config_;
}
BufferDesc &BufferDesc::operator=(const BufferDesc &desc) {
  if (this == &desc) {
    return *this;
  }
  size_ = desc.size_;
  config_ = desc.config_;
  return *this;
}
BufferDesc &BufferDesc::operator=(size_t size) {
  size_.emplace_back(size);
  return *this;
}

BufferDesc::BufferDesc(BufferDesc &&desc) noexcept {
  if (this == &desc) {
    return;
  }
  size_ = std::move(desc.size_);
  config_ = std::move(desc.config_);
}
BufferDesc &BufferDesc::operator=(BufferDesc &&desc) noexcept {
  if (this == &desc) {
    return *this;
  }
  size_ = std::move(desc.size_);
  config_ = std::move(desc.config_);
  return *this;
}

BufferDesc::~BufferDesc(){};

bool BufferDesc::isSameConfig(const BufferDesc &desc) const {
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
bool BufferDesc::isSameDim(const BufferDesc &desc) const {
  if (size_.size() == desc.size_.size()) {
    return true;
  } else {
    return false;
  }
}
bool BufferDesc::is1D() const { return size_.size() == 1; }

bool BufferDesc::operator>=(const BufferDesc &other) const {
  bool flag = true;
  for (int i = 0; i < size_.size(); ++i) {
    if (size_[i] < other.size_[i]) {
      flag = false;
      break;
    }
  }
  return flag;
}
bool BufferDesc::operator==(const BufferDesc &other) const {
  bool flag = true;
  for (int i = 0; i < size_.size(); ++i) {
    if (size_[i] != other.size_[i]) {
      flag = false;
      break;
    }
  }
  return flag;
}
bool BufferDesc::operator!=(const BufferDesc &other) const {
  return !(*this == other);
}

// TensorDesc
TensorDesc::TensorDesc(){};

TensorDesc::TensorDesc(base::DataType data_type, base::DataFormat format,
                       const base::IntVector &shape)
    : data_type_(data_type), data_format_(format), shape_(shape){};

TensorDesc::TensorDesc(base::DataType data_type, base::DataFormat format,
                       const base::IntVector &shape,
                       const base::SizeVector &stride)
    : data_type_(data_type),
      data_format_(format),
      shape_(shape),
      stride_(stride){};

TensorDesc::TensorDesc(const TensorDesc &desc) {
  if (this == &desc) {
    return;
  }
  data_type_ = desc.data_type_;
  data_format_ = desc.data_format_;
  shape_ = desc.shape_;
  stride_ = desc.stride_;
}
TensorDesc &TensorDesc::operator=(const TensorDesc &desc) {
  if (this == &desc) {
    return *this;
  }
  data_type_ = desc.data_type_;
  data_format_ = desc.data_format_;
  shape_ = desc.shape_;
  stride_ = desc.stride_;
}

TensorDesc::TensorDesc(TensorDesc &&desc) noexcept {
  if (this == &desc) {
    return;
  }
  data_type_ = desc.data_type_;
  data_format_ = desc.data_format_;
  shape_ = std::move(desc.shape_);
  stride_ = std::move(desc.stride_);
}
TensorDesc &TensorDesc::operator=(TensorDesc &&desc) noexcept {
  if (this == &desc) {
    return *this;
  }
  data_type_ = desc.data_type_;
  data_format_ = desc.data_format_;
  shape_ = std::move(desc.shape_);
  stride_ = std::move(desc.stride_);
}

TensorDesc::~TensorDesc(){};

bool TensorDesc::operator==(const TensorDesc &other) const {
  bool flag0 = false;
  if (shape_.size() == other.shape_.size()) {
    flag0 = true;
    for (int i = 0; i < shape_.size(); ++i) {
      if (shape_[i] != other.shape_[i]) {
        flag0 = false;
        break;
      }
    }
  }
  bool flag1 = false;
  if (stride_.size() == other.stride_.size()) {
    flag1 = true;
    for (int i = 0; i < stride_.size(); ++i) {
      if (stride_[i] != other.stride_[i]) {
        flag1 = false;
        break;
      }
    }
  }
  bool flag2 = data_type_ == other.data_type_;
  bool flag3 = data_format_ == other.data_format_;
  return flag0 && flag1 && flag2 && flag3;
}
bool TensorDesc::operator!=(const TensorDesc &other) const {
  return !(*this == other);
}

}  // namespace device
}  // namespace nndeploy