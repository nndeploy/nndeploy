

#include "nndeploy/include/device/tensor.h"

namespace nndeploy {
namespace device {

static TypeTensorRegister<TypeTensorCreator<Tensor>> g_defalut_tensor_register(
    base::kTensorTypeDefault);

Tensor::Tensor() : buffer_(nullptr) {}
Tensor::~Tensor() { destory(); }

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
Tensor::Tensor(Device *device, const TensorDesc &desc, int32_t data_id,
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
void Tensor::create(Device *device, const TensorDesc &desc, int32_t data_id,
                    const std::string &name, const base::IntVector &config) {
  create(device, desc, nullptr, nullptr, data_id, name, config);
}

void Tensor::create(const TensorDesc &desc, Buffer *buffer,
                    const std::string &name) {
  create(nullptr, desc, buffer, nullptr, -1, name, base::IntVector());
}

void Tensor::create(Device *device, const TensorDesc &desc, Buffer *buffer,
                    void *data_ptr, int32_t data_id, const std::string &name,
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
  desc_.format_ = base::kDataFormatNotSupport;
  desc_.shape_.clear();
  desc_.stride_.clear();

  is_external_buffer_ = false;

  deallocateBuffer();
}

void Tensor::allocBuffer(Device *device, const base::IntVector &config) {
  if (empty()) {
    return;
  }
  deallocateBuffer();
  is_external_buffer_ = false;
  BufferDesc buffer_desc = device->toBufferDesc(desc_, config);
  buffer_ = device->allocate(buffer_desc);
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
  if (1) {
    deallocateBuffer();
    is_external_buffer_ = true;
    buffer_ = buffer;
    return true;
  } else {
    buffer_ = buffer;
    return true;
  }
}

// get
bool Tensor::empty() { return buffer_->empty(); }

std::string Tensor::getName() { return name_; }

TensorDesc Tensor::getDesc() { return desc_; }
base::DataType Tensor::getDataType() { return desc_.data_type_; }
base::DataFormat Tensor::getDataFormat() { return desc_.format_; }
base::IntVector Tensor::getShape() { return desc_.shape_; }
int32_t Tensor::getShapeIndex(int index) { return desc_.shape_[index]; }
base::SizeVector Tensor::getStride() { return desc_.stride_; }
size_t Tensor::getStrideIndex(int index) { return desc_.stride_[index]; }

Buffer *Tensor::getBuffer() { return buffer_; }
base::DeviceType Tensor::getDeviceType() { return buffer_->getDeviceType(); }
Device *Tensor::getDevice() { return buffer_->getDevice(); }
BufferPool *Tensor::getBufferPool() { return buffer_->getBufferPool(); }
bool Tensor::isBufferPool() { return buffer_->isBufferPool(); }
BufferDesc Tensor::getBufferDesc() { return buffer_->getDesc(); }
size_t Tensor::getSize() { return buffer_->getSize(); }
base::SizeVector Tensor::getSizeVector() { return buffer_->getSizeVector(); }
base::IntVector Tensor::getConfig() { return buffer_->getConfig(); }
void *Tensor::getPtr() { return buffer_->getPtr(); }
int32_t Tensor::getId() { return buffer_->getId(); }
BufferSourceType Tensor::getBufferSourceType() {
  return buffer_->getBufferSourceType();
}

std::map<base::TensorType, std::shared_ptr<TensorCreator>> &
getGlobalTensorCreatorMap() {
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
