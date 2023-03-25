

#include "nndeploy/include/device/default_tensor_impl.h"

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/device/buffer.h"
#include "nndeploy/include/device/buffer_pool.h"
#include "nndeploy/include/device/device.h"
#include "nndeploy/include/device/tensor.h"

namespace nndeploy {
namespace device {

static TypeTensorRegister<TypeTensorCreator<DefaultTensorImpl>>
    g_defalut_tensor_register(base::kTensorImplTypeDefault);

DefaultTensorImpl::DefaultTensorImpl() {}
DefaultTensorImpl::~DefaultTensorImpl() { destory(); }

DefaultTensorImpl::DefaultTensorImpl(const TensorDesc &desc,
                                     const std::string &name) {
  create(desc, name);
}
DefaultTensorImpl::DefaultTensorImpl(const TensorDesc &desc, Buffer *buffer,
                                     const std::string &name) {
  create(desc, buffer, name);
}
DefaultTensorImpl::DefaultTensorImpl(Device *device, const TensorDesc &desc,
                                     const std::string &name,
                                     const base::IntVector &config) {
  create(device, desc, name, config);
}
DefaultTensorImpl::DefaultTensorImpl::DefaultTensorImpl(
    BufferPool *buffer_pool, const TensorDesc &desc, const std::string &name,
    const base::IntVector &config) {
  create(buffer_pool, desc, name, config);
}

// create
void DefaultTensorImpl::create(const TensorDesc &desc,
                               const std::string &name) {
  create(nullptr, nullptr, desc, nullptr, name, base::IntVector());
}

void DefaultTensorImpl::create(const TensorDesc &desc, Buffer *buffer,
                               const std::string &name) {
  create(nullptr, nullptr, desc, buffer, name, base::IntVector());
}

void DefaultTensorImpl::create(Device *device, const TensorDesc &desc,
                               const std::string &name,
                               const base::IntVector &config) {
  create(device, nullptr, desc, nullptr, name, config);
}
void DefaultTensorImpl::create(BufferPool *buffer_pool, const TensorDesc &desc,
                               const std::string &name,
                               const base::IntVector &config) {
  create(nullptr, buffer_pool, desc, nullptr, name, config);
}

void DefaultTensorImpl::create(Device *device, BufferPool *buffer_pool,
                               const TensorDesc &desc, Buffer *buffer,
                               const std::string &name,
                               const base::IntVector &config) {
  desc_ = desc;
  name_ = name;
  if (buffer != nullptr) {
    is_external_buffer_ = true;
    buffer_ = buffer;
    return;
  } else if (buffer_pool != nullptr) {
    is_external_buffer_ = false;
    BufferDesc buffer_desc =
        buffer_pool->getDevice()->toBufferDesc(desc_, config);
    buffer_ = buffer_pool->allocate(buffer_desc);
    return;
  } else if (device != nullptr) {
    is_external_buffer_ = false;
    BufferDesc buffer_desc = device->toBufferDesc(desc, config);
    buffer_ = device->allocate(buffer_desc);
    return;
  }
  return;
}

void DefaultTensorImpl::destory() {
  name_.clear();

  desc_.data_type_ = base::DataTypeOf<float>();
  desc_.format_ = base::kDataFormatNotSupport;
  desc_.shape_.clear();
  desc_.stride_.clear();

  deallocateBuffer();
}

void DefaultTensorImpl::allocBuffer(Device *device,
                                    const base::IntVector &config) {
  if (empty()) {
    return;
  }

  deallocateBuffer();

  BufferDesc buffer_desc = device->toBufferDesc(desc_, config);
  buffer_ = device->allocate(buffer_desc);
}
void DefaultTensorImpl::allocBuffer(BufferPool *buffer_pool,
                                    const base::IntVector &config) {
  if (empty()) {
    return;
  }

  deallocateBuffer();

  Device *device = buffer_pool->getDevice();
  BufferDesc buffer_desc = device->toBufferDesc(desc_, config);
  buffer_ = buffer_pool->allocate(buffer_desc);
}
void DefaultTensorImpl::deallocateBuffer() {
  if (buffer_ != nullptr && is_external_buffer_ == false) {
    buffer_->subRef();
    if (buffer_->getRef() == 1) {
      if (buffer_->isBufferPool()) {
        BufferPool *pool = buffer_->getBufferPool();
        pool->deallocate(buffer_);
      } else {
        Device *device = buffer_->getDevice();
        device->deallocate(buffer_);
      }
    }
  }
  buffer_ = nullptr;
}

bool DefaultTensorImpl::justModify(const TensorDesc &desc) {
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

TensorDesc DefaultTensorImpl::getDesc() { return desc_; }
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
    temp = creater_map[type]->CreateTensor();
  }
  return temp;
}

}  // namespace device
}  // namespace nndeploy
