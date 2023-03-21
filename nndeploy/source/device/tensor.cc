

#include "nndeploy/include/device/tensor.h"

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/device/buffer.h"
#include "nndeploy/include/device/buffer_pool.h"
#include "nndeploy/include/device/device.h"

namespace nndeploy {
namespace device {

TensorImpl::TensorImpl() {}
TensorImpl::~TensorImpl() {
  if (buffer_ != nullptr) {
    buffer_->subRef();
    if (buffer_->getRef() == 0) {
      Device *device = buffer_->getDevice();
      device->free(buffer_);
    }
  }
}
TensorImpl::TensorImpl(Device *device, const TensorDesc &desc,
                       const std::string &name, base::TensorType type,
                       const base::IntVector &config) {
  create(device, desc, name, type, config);
}
TensorImpl::TensorImpl::TensorImpl(BufferPool *buffer_pool,
                                   const TensorDesc &desc,
                                   const std::string &name,
                                   base::TensorType type,
                                   const base::IntVector &config) {
  create(buffer_pool, desc, name, type, config);
}
TensorImpl::TensorImpl(TensorDesc desc, Buffer *buffer, const std::string &name,
                       base::TensorType type)
    : desc_(desc), buffer_(buffer), name_(name), type_(type) {}

//
// TensorImpl::TensorImpl(const TensorImpl &tensorimpl) {
//   if (this == &tensorimpl) {
//     return;
//   }

//   desc_ = tensorimpl.desc_;
//   buffer_ = tensorimpl.buffer_;
//   buffer_->addRef();
// }
// TensorImpl::TensorImpl(TensorImpl &&tensorimpl) {
//   if (this == &tensorimpl) {
//     return;
//   }
//   desc_ = tensorimpl.desc_;
//   buffer_ = tensorimpl.buffer_;
//   tensorimpl.buffer_ = nullptr;
// }

// //
// TensorImpl &TensorImpl::operator=(const TensorImpl &tensorimpl) {
//   if (this == &tensorimpl) {
//     return *this;
//   }

//   desc_ = tensorimpl.desc_;
//   buffer_ = tensorimpl.buffer_;
//   buffer_->addRef();
//   return *this;
// }
// TensorImpl &TensorImpl::operator==(TensorImpl &&tensorimpl) {
//   if (this == &tensorimpl) {
//     return *this;
//   }
//   desc_ = tensorimpl.desc_;
//   buffer_ = tensorimpl.buffer_;
//   tensorimpl.buffer_ = nullptr;
//   return *this;
// }

// create
void TensorImpl::create(Device *device, const TensorDesc &desc,
                        const std::string &name, base::TensorType type,
                        const base::IntVector &config) {
  desc_ = desc;
  name_ = name;
  type_ = type;
  BufferDesc buffer_desc = device->toBufferDesc(desc, config);
  buffer_ = device->malloc(buffer_desc);
}
void TensorImpl::create(BufferPool *buffer_pool, const TensorDesc &desc,
                        const std::string &name, base::TensorType type,
                        const base::IntVector &config) {
  desc_ = desc;
  name_ = name;
  type_ = type;
  BufferDesc buffer_desc =
      buffer_pool->getDevice()->toBufferDesc(desc_, config);
  buffer_ = buffer_pool->malloc(buffer_desc);
}

// get
bool TensorImpl::empty() { return buffer_->empty(); }

std::string TensorImpl::getName() { return name_; }
base::TensorType TensorImpl::getTensorType() { return type_; }

TensorDesc TensorImpl::getDesc() { return desc_; }
base::DataType TensorImpl::getDataType() { return desc_.data_type_; }
base::IntVector TensorImpl::getShape() { return desc_.shape_; }
int32_t TensorImpl::getShapeIndex(int index) { return desc_.shape_[index]; }
base::SizeVector TensorImpl::getStride() { return desc_.stride_; }
size_t TensorImpl::getStrideIndex(int index) { return desc_.stride_[index]; }

Buffer *TensorImpl::getBuffer() { return buffer_; }
base::DeviceType TensorImpl::getDeviceType() {
  return buffer_->getDeviceType();
}
Device *TensorImpl::getDevice() { return buffer_->getDevice(); }
BufferPool *TensorImpl::getBufferPool() { return buffer_->getBufferPool(); }
bool TensorImpl::isBufferPool() { return buffer_->isBufferPool(); }
BufferDesc TensorImpl::getBufferDesc() { return buffer_->getDesc(); }
size_t TensorImpl::getSize() { return buffer_->getSize(); }
base::SizeVector TensorImpl::getSizeVector() {
  return buffer_->getSizeVector();
}
base::IntVector TensorImpl::getConfig() { return buffer_->getConfig(); }
void *TensorImpl::getPtr() { return buffer_->getPtr(); }
int32_t TensorImpl::getId() { return buffer_->getId(); }
base::BufferSourceType TensorImpl::getBufferSourceType() {
  return buffer_->getBufferSourceType();
}

Tensor::Tensor() {}
Tensor::~Tensor() {
  if (tensor_impl_ != nullptr) {
    delete tensor_impl_;
  }
}
Tensor::Tensor(Device *device, const TensorDesc &desc, const std::string &name,
               base::TensorType type, const base::IntVector &config) {
  tensor_impl_ = new TensorImpl(device, desc, name, type, config);
}
Tensor::Tensor::Tensor(BufferPool *buffer_pool, const TensorDesc &desc,
                       const std::string &name, base::TensorType type,
                       const base::IntVector &config) {
  tensor_impl_ = new TensorImpl(buffer_pool, desc, name, type, config);
}
Tensor::Tensor(const TensorDesc &desc, Buffer *buffer, const std::string &name,
               base::TensorType type) {
  tensor_impl_ = new TensorImpl(desc, buffer, name, type);
}

//
// Tensor::Tensor(const Tensor &tensor) {
//   if (this == &tensor) {
//     return;
//   }

//   desc_ = tensor.desc_;
//   buffer_ = tensor.buffer_;
//   buffer_->addRef();
// }
// Tensor::Tensor(Tensor &&tensor) {
//   if (this == &tensor) {
//     return;
//   }
//   desc_ = tensor.desc_;
//   buffer_ = tensor.buffer_;
//   tensor.buffer_ = nullptr;
// }

// //
// Tensor &Tensor::operator=(const Tensor &tensor) {
//   if (this == &tensor) {
//     return *this;
//   }

//   desc_ = tensor.desc_;
//   buffer_ = tensor.buffer_;
//   buffer_->addRef();
//   return *this;
// }
// Tensor &Tensor::operator==(Tensor &&tensor) {
//   if (this == &tensor) {
//     return *this;
//   }
//   desc_ = tensor.desc_;
//   buffer_ = tensor.buffer_;
//   tensor.buffer_ = nullptr;
//   return *this;
// }

// create
void Tensor::create(Device *device, const TensorDesc &desc,
                    const std::string &name, base::TensorType type,
                    const base::IntVector &config) {
  if (tensor_impl_ != nullptr) {
    tensor_impl_ = new TensorImpl(device, desc, name, type, config);
  } else {
    tensor_impl_->create(device, desc, name, type, config);
  }
}
void Tensor::create(BufferPool *buffer_pool, const TensorDesc &desc,
                    const std::string &name, base::TensorType type,
                    const base::IntVector &config) {
  if (tensor_impl_ != nullptr) {
    tensor_impl_ = new TensorImpl(buffer_pool, desc, name, type, config);
  } else {
    tensor_impl_->create(buffer_pool, desc, name, type, config);
  }
}

// get
bool Tensor::empty() { return tensor_impl_->empty(); }

std::string Tensor::getName() { return tensor_impl_->getName(); }
base::TensorType Tensor::getTensorType() {
  return tensor_impl_->getTensorType();
}

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
base::BufferSourceType Tensor::getBufferSourceType() {
  return tensor_impl_->getBufferSourceType();
}

TensorPtrArray::TensorPtrArray() {}
TensorPtrArray::TensorPtrArray(std::vector<Tensor *> tensors)
    : tensors_(tensors) {}
TensorPtrArray::TensorPtrArray(Tensor *tensor) { tensors_.push_back(tensor); }

TensorPtrArray::~TensorPtrArray() {}

void TensorPtrArray::addTensor(Tensor *tensor) { tensors_.push_back(tensor); }
void TensorPtrArray::addTensor(std::vector<Tensor *> tensors_) {
  for (auto tensor : tensors_) {
    tensors_.push_back(tensor);
  }
}

int TensorPtrArray::getTensorSize() { return tensors_.size(); }
Tensor *TensorPtrArray::getTensor() { return tensors_[0]; }
Tensor *TensorPtrArray::getTensor(int index) { return tensors_[index]; }

}  // namespace device
}  // namespace nndeploy
