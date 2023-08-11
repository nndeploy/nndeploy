

#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace device {

static TypeTensorRegister<TypeTensorCreator<Tensor>> g_defalut_tensor_register(
    base::kTensorTypeDefault);

Tensor::Tensor() : buffer_(nullptr) {}
Tensor::~Tensor() { destory(); }

Tensor::Tensor(const std::string &name) : name_(name){};

Tensor::Tensor(const TensorDesc &desc, const std::string &name) {
  create(desc, name);
}

Tensor::Tensor(Device *device, const TensorDesc &desc, const std::string &name,
               const base::IntVector &config) {
  create(device, desc, name, config);
}

Tensor::Tensor(Device *device, const TensorDesc &desc, void *data_ptr,
               const std::string &name, const base::IntVector &config) {
  create(device, desc, data_ptr, name, config);
}
Tensor::Tensor(Device *device, const TensorDesc &desc, int data_id,
               const std::string &name, const base::IntVector &config) {
  create(device, desc, data_id, name, config);
}

Tensor::Tensor(const TensorDesc &desc, Buffer *buffer,
               const std::string &name) {
  create(desc, buffer, name);
}

// create
void Tensor::create(const TensorDesc &desc, const std::string &name) {
  create(nullptr, desc, nullptr, nullptr, -1, name, base::IntVector());
}

void Tensor::create(Device *device, const TensorDesc &desc,
                    const std::string &name, const base::IntVector &config) {
  create(device, desc, nullptr, nullptr, -1, name, config);
}

void Tensor::create(Device *device, const TensorDesc &desc, void *data_ptr,
                    const std::string &name, const base::IntVector &config) {
  create(device, desc, nullptr, data_ptr, -1, name, config);
}
void Tensor::create(Device *device, const TensorDesc &desc, int data_id,
                    const std::string &name, const base::IntVector &config) {
  create(device, desc, nullptr, nullptr, data_id, name, config);
}

void Tensor::create(const TensorDesc &desc, Buffer *buffer,
                    const std::string &name) {
  create(nullptr, desc, buffer, nullptr, -1, name, base::IntVector());
}

void Tensor::create(Device *device, const TensorDesc &desc, Buffer *buffer,
                    void *data_ptr, int data_id, const std::string &name,
                    const base::IntVector &config) {
  desc_ = desc;
  name_ = name;
  if (buffer != nullptr) {
    is_external_buffer_ = true;
    buffer_ = buffer;
    return;
  }
  if (device != nullptr) {
    is_external_buffer_ = false;
    if (data_ptr != nullptr) {
      BufferDesc buffer_desc = device->toBufferDesc(desc, config);
      buffer_ =
          device->create(buffer_desc, data_ptr, kBufferSourceTypeExternal);
      return;
    } else if (data_id != -1) {
      BufferDesc buffer_desc = device->toBufferDesc(desc, config);
      buffer_ = device->create(buffer_desc, data_id, kBufferSourceTypeExternal);
      return;
    } else {
      BufferDesc buffer_desc = device->toBufferDesc(desc, config);
      buffer_ = device->allocate(buffer_desc);
      return;
    }
  }
  return;
}

void Tensor::destory() {
  name_.clear();

  desc_.data_type_ = base::dataTypeOf<float>();
  desc_.data_format_ = base::kDataFormatNotSupport;
  desc_.shape_.clear();
  desc_.stride_.clear();

  deallocateBuffer();

  is_external_buffer_ = false;
}

void Tensor::allocBuffer(Device *device, const base::IntVector &config) {
  BufferDesc dst_buffer_desc = device->toBufferDesc(desc_, config);
  if (buffer_ != nullptr && device == buffer_->getDevice()) {
    BufferDesc src_buffer_desc = buffer_->getDesc();
    if (device->compareBufferDesc(dst_buffer_desc, src_buffer_desc) <= 0) {
      return;
    }
  }
  deallocateBuffer();
  is_external_buffer_ = false;
  buffer_ = device->allocate(dst_buffer_desc);
}
void Tensor::deallocateBuffer() {
  if (buffer_ != nullptr && is_external_buffer_ == false) {
    if (buffer_->subRef() == 1) {
      Device *device = buffer_->getDevice();
      device->deallocate(buffer_);
    }
  }
  buffer_ = nullptr;
}

bool Tensor::justModify(const TensorDesc &desc) {
  if (buffer_ == nullptr) {
    desc_ = desc;
    return true;
  } else {
    // TODO, 做到可以安全修改
    desc_ = desc;
    return true;
  }
}

bool Tensor::justModify(Buffer *buffer) {
  // TODO, 做到可以安全修改
  deallocateBuffer();
  is_external_buffer_ = true;
  buffer_ = buffer;
  return true;
}

// get
bool Tensor::empty() {
  bool flag = desc_.shape_.empty();
  if (buffer_) {
    flag = flag || buffer_->empty();
  }
  return flag;
}
bool Tensor::isExternalBuffer() { return is_external_buffer_; }
std::string Tensor::getName() { return name_; }

TensorDesc Tensor::getDesc() { return desc_; }
base::DataType Tensor::getDataType() { return desc_.data_type_; }
base::DataFormat Tensor::getDataFormat() { return desc_.data_format_; }
base::IntVector Tensor::getShape() { return desc_.shape_; }
int Tensor::getShapeIndex(int index) {
  if (index < desc_.shape_.size()) {
    return desc_.shape_[index];
  } else {
    return -1;
  }
}
int Tensor::getBatch() {
  if (!desc_.shape_.empty()) {
    return desc_.shape_[0];
  } else {
    return -1;
  }
}
int Tensor::getChannel() {
  int ret = -1;
  switch (desc_.data_format_) {
    case base::kDataFormatN:
      break;
    case base::kDataFormatNC:
      ret = desc_.shape_[1];
      break;
    case base::kDataFormatNHW:
      break;
    case base::kDataFormatNWC:
      ret = desc_.shape_[2];
      break;
    case base::kDataFormatNCW:
      ret = desc_.shape_[1];
      break;
    case base::kDataFormatNCHW:
      ret = desc_.shape_[1];
      break;
    case base::kDataFormatNHWC:
      ret = desc_.shape_[3];
      break;
    case base::kDataFormatOIHW:
      break;
    case base::kDataFormatNC4HW:
      ret = desc_.shape_[1];
      break;
    case base::kDataFormatNC8HW:
      ret = desc_.shape_[1];
      break;
    case base::kDataFormatNCDHW:
      ret = desc_.shape_[1];
      break;
    case base::kDataFormatNDHWC:
      ret = desc_.shape_[4];
      break;
    default:
      break;
  }
  return ret;
}
int Tensor::getDepth() {
  int ret = -1;
  switch (desc_.data_format_) {
    case base::kDataFormatNCDHW:
      ret = desc_.shape_[2];
      break;
    case base::kDataFormatNDHWC:
      ret = desc_.shape_[1];
      break;
    default:
      break;
  }
  return ret;
}
int Tensor::getHeight() {
  int ret = -1;
  switch (desc_.data_format_) {
    case base::kDataFormatN:
      break;
    case base::kDataFormatNC:
      break;
    case base::kDataFormatNHW:
      ret = desc_.shape_[1];
      break;
    case base::kDataFormatNWC:
      break;
    case base::kDataFormatNCW:
      break;
    case base::kDataFormatNCHW:
      ret = desc_.shape_[2];
      break;
    case base::kDataFormatNHWC:
      ret = desc_.shape_[1];
      break;
    case base::kDataFormatOIHW:
      ret = desc_.shape_[2];
      break;
    case base::kDataFormatNC4HW:
      ret = desc_.shape_[2];
      break;
    case base::kDataFormatNC8HW:
      ret = desc_.shape_[2];
      break;
    case base::kDataFormatNCDHW:
      ret = desc_.shape_[3];
      break;
    case base::kDataFormatNDHWC:
      ret = desc_.shape_[2];
      break;
    default:
      break;
  }
  return ret;
}
int Tensor::getWidth() {
  int ret = -1;
  switch (desc_.data_format_) {
    case base::kDataFormatN:
      break;
    case base::kDataFormatNC:
      break;
    case base::kDataFormatNHW:
      ret = desc_.shape_[2];
      break;
    case base::kDataFormatNWC:
      ret = desc_.shape_[1];
      break;
    case base::kDataFormatNCW:
      ret = desc_.shape_[1];
      break;
    case base::kDataFormatNCHW:
      ret = desc_.shape_[3];
      break;
    case base::kDataFormatNHWC:
      ret = desc_.shape_[2];
      break;
    case base::kDataFormatOIHW:
      ret = desc_.shape_[3];
      break;
    case base::kDataFormatNC4HW:
      ret = desc_.shape_[3];
      break;
    case base::kDataFormatNC8HW:
      ret = desc_.shape_[3];
      break;
    case base::kDataFormatNCDHW:
      ret = desc_.shape_[4];
      break;
    case base::kDataFormatNDHWC:
      ret = desc_.shape_[3];
      break;
    default:
      break;
  }
  return ret;
}
base::SizeVector Tensor::getStride() { return desc_.stride_; }
size_t Tensor::getStrideIndex(int index) {
  if (index < desc_.stride_.size()) {
    return desc_.stride_[index];
  } else {
    return 0;
  }
}
Buffer *Tensor::getBuffer() { return buffer_; }
base::DeviceType Tensor::getDeviceType() {
  if (buffer_) {
    return buffer_->getDeviceType();
  } else {
    return base::DeviceType(base::kDeviceTypeCodeNotSupport);
  }
}
Device *Tensor::getDevice() {
  if (buffer_) {
    return buffer_->getDevice();
  } else {
    return nullptr;
  }
}
BufferPool *Tensor::getBufferPool() {
  if (buffer_) {
    return buffer_->getBufferPool();
  } else {
    return nullptr;
  }
}
bool Tensor::isBufferPool() {
  if (buffer_) {
    return buffer_->isBufferPool();
  } else {
    return false;
  }
}
BufferDesc Tensor::getBufferDesc() {
  if (buffer_) {
    return buffer_->getDesc();
  } else {
    return BufferDesc();
  }
}
size_t Tensor::getSize() {
  if (buffer_) {
    return buffer_->getSize();
  } else {
    return 0;
  }
}
base::SizeVector Tensor::getSizeVector() {
  if (buffer_) {
    return buffer_->getSizeVector();
  } else {
    return base::SizeVector();
  }
}
base::IntVector Tensor::getConfig() {
  if (buffer_) {
    return buffer_->getConfig();
  } else {
    return base::IntVector();
  }
}
void *Tensor::getPtr() {
  if (buffer_) {
    return buffer_->getPtr();
  } else {
    return nullptr;
  }
}
int Tensor::getId() {
  if (buffer_) {
    return buffer_->getId();
  } else {
    return -1;
  }
}
BufferSourceType Tensor::getBufferSourceType() {
  if (buffer_) {
    return buffer_->getBufferSourceType();
  } else {
    return device::kBufferSourceTypeNone;
  }
}

std::map<base::TensorType, std::shared_ptr<TensorCreator>>
    &getGlobalTensorCreatorMap() {
  static std::once_flag once;
  static std::shared_ptr<
      std::map<base::TensorType, std::shared_ptr<TensorCreator>>>
      creators;
  std::call_once(once, []() {
    creators.reset(
        new std::map<base::TensorType, std::shared_ptr<TensorCreator>>);
  });
  return *creators;
}

Tensor *createTensor(base::TensorType type) {
  Tensor *temp = nullptr;
  auto &creater_map = getGlobalTensorCreatorMap();
  if (creater_map.count(type) > 0) {
    temp = creater_map[type]->createTensor();
  }
  return temp;
}

}  // namespace device
}  // namespace nndeploy
