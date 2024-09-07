
#include "nndeploy/device/type.h"

namespace nndeploy {
namespace device {

BufferDesc::BufferDesc() {}
BufferDesc::BufferDesc(size_t size) {
  size_.emplace_back(size);
  real_size_.emplace_back(size);
}
BufferDesc::BufferDesc(size_t *size, size_t len) {
  for (int i = 0; i < len; ++i) {
    size_.emplace_back(size[i]);
    real_size_.emplace_back(size[i]);
  }
}
BufferDesc::BufferDesc(size_t size, const base::IntVector &config)
    : config_(config) {
  size_.emplace_back(size);
  real_size_.emplace_back(size);
}
BufferDesc::BufferDesc(const base::SizeVector &size,
                       const base::IntVector &config)
    : size_(size), config_(config), real_size_(size) {}
BufferDesc::BufferDesc(size_t *size, size_t len, const base::IntVector &config)
    : config_(config) {
  for (int i = 0; i < len; ++i) {
    size_.emplace_back(size[i]);
    real_size_.emplace_back(size[i]);
  }
}

BufferDesc::BufferDesc(const BufferDesc &desc) {
  if (this == &desc) {
    return;
  }
  size_ = desc.size_;
  config_ = desc.config_;
  real_size_ = desc.real_size_;
}
BufferDesc &BufferDesc::operator=(const BufferDesc &desc) {
  if (this == &desc) {
    return *this;
  }
  size_ = desc.size_;
  config_ = desc.config_;
  real_size_ = desc.real_size_;
  return *this;
}
BufferDesc &BufferDesc::operator=(size_t size) {
  size_.emplace_back(size);
  real_size_.emplace_back(size);
  return *this;
}

BufferDesc::BufferDesc(BufferDesc &&desc) noexcept {
  if (this == &desc) {
    return;
  }
  size_ = std::move(desc.size_);
  config_ = std::move(desc.config_);
  real_size_ = std::move(desc.real_size_);
}
BufferDesc &BufferDesc::operator=(BufferDesc &&desc) noexcept {
  if (this == &desc) {
    return *this;
  }
  size_ = std::move(desc.size_);
  config_ = std::move(desc.config_);
  real_size_ = std::move(desc.real_size_);
  return *this;
}

BufferDesc::~BufferDesc(){};

size_t BufferDesc::getSize() const {
  if (size_.empty()) {
    return 0;
  }
  size_t size = 1;
  for (auto iter : size_) {
    size *= iter;
  }
  return size;
}
base::SizeVector BufferDesc::getSizeVector() const { return size_; }
size_t BufferDesc::getRealSize() const {
  if (real_size_.empty()) {
    return 0;
  }
  size_t size = 1;
  for (auto iter : real_size_) {
    size *= iter;
  }
  return size;
}
base::SizeVector BufferDesc::getRealSizeVector() const { return real_size_; }

base::IntVector BufferDesc::getConfig() const { return config_; }

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

void BufferDesc::print() {
  std::cout << "BufferDesc: \n";
  std::cout << "size: ";
  for (int i = 0; i < size_.size(); ++i) {
    std::cout << size_[i] << " ";
  }
  std::cout << "real_size: ";
  for (int i = 0; i < real_size_.size(); ++i) {
    std::cout << real_size_[i] << " ";
  }
  std::cout << std::endl;
  if (!config_.empty()) {
    std::cout << "config: ";
    for (int i = 0; i < config_.size(); ++i) {
      std::cout << config_[i] << " ";
    }
    std::cout << std::endl;
  }
}

bool BufferDesc::justModify(const size_t &size) {
  if (size_.size() == 1) {
    if (size > real_size_[0]) {
      NNDEPLOY_LOGE("size=%ld great than real_size_[0]=%ld.\n", size,
                    real_size_[0]);
      return false;
    }
    size_[0] = size;
    return true;
  } else {
    NNDEPLOY_LOGE("size_.size[%ld] not equal 1.\n", size_.size());
    return false;
  }
}

bool BufferDesc::justModify(const base::SizeVector &size) {
  if (size_.size() == size.size()) {
    for (size_t i = 0; i < size.size(); ++i) {
      if (size[i] > real_size_[i]) {
        NNDEPLOY_LOGE("size[%ld]=%ld great than real_size_[%ld]=%ld.\n", i,
                      size[i], i, real_size_[i]);
        return false;
      }
      size_[i] = size[i];
    }
    return true;
  } else {
    NNDEPLOY_LOGE("size_.size[%ld] not equal 1.\n", size_.size());
    return false;
  }
}
bool BufferDesc::justModify(const BufferDesc &desc) {
  if (config_.size() != desc.config_.size()) {
    return false;
  }
  for (size_t i = 0; i < config_.size(); ++i) {
    if (config_[i] != desc.config_[i]) {
      return false;
    }
  }
  const base::SizeVector size = desc.size_;
  if (size_.size() == size.size()) {
    for (size_t i = 0; i < size.size(); ++i) {
      if (size[i] > real_size_[i]) {
        NNDEPLOY_LOGE("size[%ld]=%ld great than real_size_[%ld]=%ld.\n", i,
                      size[i], i, real_size_[i]);
        return false;
      }
      size_[i] = size[i];
    }
    return true;
  } else {
    NNDEPLOY_LOGE("size_.size[%ld] not equal 1.\n", size_.size());
    return false;
  }
}

void BufferDesc::clear() {
  size_.clear();
  config_.clear();
  real_size_.clear();
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
  return *this;
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
  return *this;
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

void TensorDesc::print() {
  std::cout << "TensorDesc: \n";
  std::cout << "data_type: " << base::dataTypeToString(data_type_) << std::endl;
  std::cout << "data_format: " << base::dataFormatToString(data_format_)
            << std::endl;
  std::cout << "shape: ";
  for (int i = 0; i < shape_.size(); ++i) {
    std::cout << shape_[i] << " ";
  }
  std::cout << std::endl;
  if (!stride_.empty()) {
    std::cout << "stride: ";
    for (int i = 0; i < stride_.size(); ++i) {
      std::cout << stride_[i] << " ";
    }
    std::cout << std::endl;
  }
}

}  // namespace device
}  // namespace nndeploy