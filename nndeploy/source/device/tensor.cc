

#include "nndeploy/source/device/tensor.h"

namespace nndeploy {
namespace device {

static TypeTensorRegister<TypeTensorCreator<DefaultTensorImpl>>
    g_defalut_tensor_register(base::kTensorImplTypeDefault);

DefaultTensorImpl::DefaultTensorImpl() : buffer_(nullptr) {}
DefaultTensorImpl::~DefaultTensorImpl() { destory(); }

DefaultTensorImpl::DefaultTensorImpl(const TensorImplDesc &desc,
                                     const std::string &name) {
  create(desc, name);
}

DefaultTensorImpl::DefaultTensorImpl(Device *device, const TensorImplDesc &desc,
                                     const std::string &name,
                                     const base::IntVector &config) {
  create(device, desc, name, config);
}

DefaultTensorImpl::DefaultTensorImpl(Device *device, const TensorImplDesc &desc,
                                     void *data_ptr, const std::string &name,
                                     const base::IntVector &config) {
  create(device, desc, data_ptr, name, config);
}
DefaultTensorImpl::DefaultTensorImpl(Device *device, const TensorImplDesc &desc,
                                     int32_t data_id, const std::string &name,
                                     const base::IntVector &config) {
  create(device, desc, data_id, name, config);
}

DefaultTensorImpl::DefaultTensorImpl(const TensorImplDesc &desc, Buffer *buffer,
                                     const std::string &name) {
  create(desc, buffer, name);
}

// create
void DefaultTensorImpl::create(const TensorImplDesc &desc,
                               const std::string &name) {
  create(nullptr, desc, nullptr, nullptr, -1, name, base::IntVector());
}

void DefaultTensorImpl::create(Device *device, const TensorImplDesc &desc,
                               const std::string &name,
                               const base::IntVector &config) {
  create(device, desc, nullptr, nullptr, -1, name, config);
}

void DefaultTensorImpl::create(Device *device, const TensorImplDesc &desc,
                               void *data_ptr, const std::string &name,
                               const base::IntVector &config) {
  create(device, desc, nullptr, data_ptr, -1, name, config);
}
void DefaultTensorImpl::create(Device *device, const TensorImplDesc &desc,
                               int32_t data_id, const std::string &name,
                               const base::IntVector &config) {
  create(device, desc, nullptr, nullptr, data_id, name, config);
}

void DefaultTensorImpl::create(const TensorImplDesc &desc, Buffer *buffer,
                               const std::string &name) {
  create(nullptr, desc, buffer, nullptr, -1, name, base::IntVector());
}

void DefaultTensorImpl::create(Device *device, const TensorImplDesc &desc,
                               Buffer *buffer, void *data_ptr, int32_t data_id,
                               const std::string &name,
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

void DefaultTensorImpl::destory() {
  name_.clear();

  desc_.data_type_ = base::dataTypeOf<float>();
  desc_.format_ = base::kDataFormatNotSupport;
  desc_.shape_.clear();
  desc_.stride_.clear();

  is_external_buffer_ = false;

  deallocateBuffer();
}

void DefaultTensorImpl::allocBuffer(Device *device,
                                    const base::IntVector &config) {
  if (empty()) {
    return;
  }
  deallocateBuffer();
  is_external_buffer_ = false;
  BufferDesc buffer_desc = device->toBufferDesc(desc_, config);
  buffer_ = device->allocate(buffer_desc);
}
void DefaultTensorImpl::deallocateBuffer() {
  if (buffer_ != nullptr && is_external_buffer_ == false) {
    buffer_->subRef();
    if (buffer_->getRef() == 0) {
      Device *device = buffer_->getDevice();
      device->deallocate(buffer_);
    }
  }
  buffer_ = nullptr;
}

bool DefaultTensorImpl::justModify(const TensorImplDesc &desc) {
  if (buffer_ == nullptr) {
    desc_ = desc;
    return true;
  } else {
    // TODO, 做到可以安全修改
    desc_ = desc;
    return true;
  }
}

bool DefaultTensorImpl::justModify(Buffer *buffer) {
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
bool DefaultTensorImpl::empty() { return buffer_->empty(); }

std::string DefaultTensorImpl::getName() { return name_; }

TensorImplDesc DefaultTensorImpl::getDesc() { return desc_; }
base::DataType DefaultTensorImpl::getDataType() { return desc_.data_type_; }
base::IntVector DefaultTensorImpl::getShape() { return desc_.shape_; }
int32_t DefaultTensorImpl::getShapeIndex(int index) {
  return desc_.shape_[index];
}
base::SizeVector DefaultTensorImpl::getStride() { return desc_.stride_; }
size_t DefaultTensorImpl::getStrideIndex(int index) {
  return desc_.stride_[index];
}

Buffer *DefaultTensorImpl::getBuffer() { return buffer_; }
base::DeviceType DefaultTensorImpl::getDeviceType() {
  return buffer_->getDeviceType();
}
Device *DefaultTensorImpl::getDevice() { return buffer_->getDevice(); }
BufferPool *DefaultTensorImpl::getBufferPool() {
  return buffer_->getBufferPool();
}
bool DefaultTensorImpl::isBufferPool() { return buffer_->isBufferPool(); }
BufferDesc DefaultTensorImpl::getBufferDesc() { return buffer_->getDesc(); }
size_t DefaultTensorImpl::getSize() { return buffer_->getSize(); }
base::SizeVector DefaultTensorImpl::getSizeVector() {
  return buffer_->getSizeVector();
}
base::IntVector DefaultTensorImpl::getConfig() { return buffer_->getConfig(); }
void *DefaultTensorImpl::getPtr() { return buffer_->getPtr(); }
int32_t DefaultTensorImpl::getId() { return buffer_->getId(); }
BufferSourceType DefaultTensorImpl::getBufferSourceType() {
  return buffer_->getBufferSourceType();
}

std::map<base::TensorImplType, std::shared_ptr<TensorCreator>>
    &getGlobalTensorCreatorMap() {
  static std::once_flag once;
  static std::shared_ptr<
      std::map<base::TensorImplType, std::shared_ptr<TensorCreator>>>
      creators;
  std::call_once(once, []() {
    creators.reset(
        new std::map<base::TensorImplType, std::shared_ptr<TensorCreator>>);
  });
  return *creators;
}

DefaultTensorImpl *createTensor(base::TensorImplType type) {
  DefaultTensorImpl *temp = nullptr;
  auto &creater_map = getGlobalTensorCreatorMap();
  if (creater_map.count(type) > 0) {
    temp = creater_map[type]->createTensor();
  }
  return temp;
}

Tensor::Tensor(base::TensorImplType type) {
  type_ = type;
  tensor_impl_ = createTensor(type);
}
Tensor::~Tensor() { destory(); }

Tensor::Tensor(const TensorImplDesc &desc, const std::string &name,
               base::TensorImplType type) {
  type_ = type;
  tensor_impl_ = createTensor(type);
  tensor_impl_->create(desc, name);
}
Tensor::Tensor(Device *device, const TensorImplDesc &desc,
               const std::string &name, const base::IntVector &config,
               base::TensorImplType type) {
  type_ = type;
  tensor_impl_ = createTensor(type);
  tensor_impl_->create(device, desc, name, config);
}
Tensor::Tensor(Device *device, const TensorImplDesc &desc, void *data_ptr,
               const std::string &name, const base::IntVector &config,
               base::TensorImplType type) {
  type_ = type;
  tensor_impl_ = createTensor(type);
  tensor_impl_->create(device, desc, data_ptr, name, config);
}
Tensor::Tensor(Device *device, const TensorImplDesc &desc, int32_t data_id,
               const std::string &name, const base::IntVector &config,
               base::TensorImplType type) {
  type_ = type;
  tensor_impl_ = createTensor(type);
  tensor_impl_->create(device, desc, data_id, name, config);
}
Tensor::Tensor(const TensorImplDesc &desc, Buffer *buffer,
               const std::string &name, base::TensorImplType type) {
  type_ = type;
  tensor_impl_ = createTensor(type);
  tensor_impl_->create(desc, buffer, name);
}

// create
void Tensor::create(const TensorImplDesc &desc, const std::string &name) {
  tensor_impl_->create(desc, name);
}
void Tensor::create(Device *device, const TensorImplDesc &desc,
                    const std::string &name, const base::IntVector &config) {
  tensor_impl_->create(device, desc, name, config);
}
void Tensor::create(Device *device, const TensorImplDesc &desc, void *data_ptr,
                    const std::string &name, const base::IntVector &config) {
  tensor_impl_->create(device, desc, data_ptr, name, config);
}
void Tensor::create(Device *device, const TensorImplDesc &desc, int32_t data_id,
                    const std::string &name, const base::IntVector &config) {
  tensor_impl_->create(device, desc, data_id, name, config);
}
void Tensor::create(const TensorImplDesc &desc, Buffer *buffer,
                    const std::string &name) {
  tensor_impl_->create(desc, buffer, name);
}

void Tensor::destory() {
  type_ = base::kTensorImplTypeDefault;
  if (tensor_impl_ != nullptr) {
    delete tensor_impl_;
  }
}

void Tensor::allocBuffer(Device *device, const base::IntVector &config) {
  tensor_impl_->allocBuffer(device, config);
}
void Tensor::deallocateBuffer() { tensor_impl_->deallocateBuffer(); }

bool Tensor::justModify(const TensorImplDesc &desc) {
  return tensor_impl_->justModify(desc);
}

bool Tensor::justModify(Buffer *buffer) {
  return tensor_impl_->justModify(buffer);
}

// get
bool Tensor::empty() { return tensor_impl_->empty(); }

std::string Tensor::getName() { return tensor_impl_->getName(); }
base::TensorImplType Tensor::getTensorImplType() { return type_; }

TensorImplDesc Tensor::getDesc() { return tensor_impl_->getDesc(); }
base::DataType Tensor::getDataType() { return tensor_impl_->getDataType(); }
base::IntVector Tensor::getShape() { return tensor_impl_->getShape(); }
int32_t Tensor::getShapeIndex(int index) {
  return tensor_impl_->getShapeIndex(index);
}
base::SizeVector Tensor::getStride() { return tensor_impl_->getStride(); }
size_t Tensor::getStrideIndex(int index) {
  return tensor_impl_->getStrideIndex(index);
}

Buffer *Tensor::getBuffer() { return tensor_impl_->getBuffer(); }
base::DeviceType Tensor::getDeviceType() {
  return tensor_impl_->getDeviceType();
}
Device *Tensor::getDevice() { return tensor_impl_->getDevice(); }
BufferPool *Tensor::getBufferPool() { return tensor_impl_->getBufferPool(); }
bool Tensor::isBufferPool() { return tensor_impl_->isBufferPool(); }
BufferDesc Tensor::getBufferDesc() { return tensor_impl_->getBufferDesc(); }
size_t Tensor::getSize() { return tensor_impl_->getSize(); }
base::SizeVector Tensor::getSizeVector() {
  return tensor_impl_->getSizeVector();
}
base::IntVector Tensor::getConfig() { return tensor_impl_->getConfig(); }
void *Tensor::getPtr() { return tensor_impl_->getPtr(); }
int32_t Tensor::getId() { return tensor_impl_->getId(); }
BufferSourceType Tensor::getBufferSourceType() {
  return tensor_impl_->getBufferSourceType();
}

TensorPtrArray::TensorPtrArray() {}
TensorPtrArray::TensorPtrArray(const std::vector<Tensor *> &tensors)
    : tensors_(tensors) {}
TensorPtrArray::TensorPtrArray(Tensor *tensor) { tensors_.push_back(tensor); }
TensorPtrArray::TensorPtrArray(Tensor &tensor) { tensors_.push_back(&tensor); }

TensorPtrArray::~TensorPtrArray() {}

void TensorPtrArray::add(Tensor *tensor) { tensors_.push_back(tensor); }
void TensorPtrArray::add(const std::vector<Tensor *> &tensors) {
  for (auto tensor : tensors) {
    tensors_.push_back(tensor);
  }
}
void TensorPtrArray::add(Tensor &tensor) { tensors_.push_back(&tensor); }

bool TensorPtrArray::empty() { return tensors_.size() == 0; }
int TensorPtrArray::getSize() { return tensors_.size(); }
Tensor *TensorPtrArray::get() { return tensors_[0]; }
Tensor *TensorPtrArray::get(int index) { return tensors_[index]; }

}  // namespace device
}  // namespace nndeploy
