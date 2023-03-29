

#include "nndeploy/source/device/tensor.h"

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/device/buffer.h"
#include "nndeploy/source/device/buffer_pool.h"
#include "nndeploy/source/device/default_tensor_impl.h"
#include "nndeploy/source/device/device.h"

namespace nndeploy {
namespace device {

Tensor::Tensor(base::TensorImplType type) {
  type_ = type;
  tensor_impl_ = createTensor(type);
}
Tensor::~Tensor() {
  if (tensor_impl_ != nullptr) {
    delete tensor_impl_;
  }
}

Tensor::Tensor(const TensorDesc &desc, const std::string &name,
               base::TensorImplType type) {
  type_ = type;
  tensor_impl_ = createTensor(type);
  tensor_impl_->create(nullptr, nullptr, desc, nullptr, name,
                       base::IntVector());
}
Tensor::Tensor(const TensorDesc &desc, Buffer *buffer, const std::string &name,
               base::TensorImplType type) {
  type_ = type;
  tensor_impl_ = createTensor(type);
  tensor_impl_->create(nullptr, nullptr, desc, buffer, name, base::IntVector());
}
Tensor::Tensor(Device *device, const TensorDesc &desc, const std::string &name,
               base::TensorImplType type, const base::IntVector &config) {
  type_ = type;
  tensor_impl_ = createTensor(type);
  tensor_impl_->create(device, nullptr, desc, nullptr, name, config);
}
Tensor::Tensor(BufferPool *buffer_pool, const TensorDesc &desc,
               const std::string &name, base::TensorImplType type,
               const base::IntVector &config) {
  type_ = type;
  tensor_impl_ = createTensor(type);
  tensor_impl_->create(nullptr, buffer_pool, desc, nullptr, name, config);
}

// create
void Tensor::create(const TensorDesc &desc, const std::string &name) {
  tensor_impl_->create(nullptr, nullptr, desc, nullptr, name,
                       base::IntVector());
}
void Tensor::create(const TensorDesc &desc, Buffer *buffer,
                    const std::string &name) {
  tensor_impl_->create(nullptr, nullptr, desc, buffer, name, base::IntVector());
}
void Tensor::create(Device *device, const TensorDesc &desc,
                    const std::string &name, const base::IntVector &config) {
  tensor_impl_->create(device, nullptr, desc, nullptr, name, config);
}
void Tensor::create(BufferPool *buffer_pool, const TensorDesc &desc,
                    const std::string &name, const base::IntVector &config) {
  tensor_impl_->create(nullptr, buffer_pool, desc, nullptr, name, config);
}

void Tensor::destory() { tensor_impl_->destory(); }

void Tensor::allocBuffer(Device *device, const base::IntVector &config) {
  tensor_impl_->allocBuffer(device, config);
}
void Tensor::allocBuffer(BufferPool *buffer_pool,
                         const base::IntVector &config) {
  tensor_impl_->allocBuffer(buffer_pool, config);
}
void Tensor::deallocateBuffer() { tensor_impl_->deallocateBuffer(); }

bool Tensor::justModify(const TensorDesc &desc) {
  return tensor_impl_->justModify(desc);
}

bool Tensor::justModify(Buffer *buffer) {
  return tensor_impl_->justModify(buffer);
}

// get
bool Tensor::empty() { return tensor_impl_->empty(); }

std::string Tensor::getName() { return tensor_impl_->getName(); }
base::TensorImplType Tensor::getTensorImplType() { return type_; }

TensorDesc Tensor::getDesc() { return tensor_impl_->getDesc(); }
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
