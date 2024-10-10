
#include "nndeploy/device/tensor.h"

#include "nndeploy/base/string.h"

namespace nndeploy {
namespace device {

static TypeTensorRegister<TypeTensorCreator<Tensor>> g_defalut_tensor_register(
    base::kTensorTypeDefault);

Tensor::Tensor() {}
Tensor::Tensor(const std::string &name) : name_(name) {};
Tensor::Tensor(const TensorDesc &desc, const std::string &name)
    : name_(name), desc_(desc) {};
Tensor::Tensor(const TensorDesc &desc, Buffer *buffer, const std::string &name)
    : name_(name), desc_(desc), is_external_(true), buffer_(buffer) {
  ref_count_ = new int(1);
}
Tensor::Tensor(Device *device, const TensorDesc &desc, const std::string &name,
               const base::IntVector &config)
    : name_(name), desc_(desc), is_external_(false) {
  BufferDesc buffer_desc = device->toBufferDesc(desc, config);
  void *ptr = device->allocate(buffer_desc);
  buffer_ = new Buffer(device, buffer_desc, ptr, base::kMemoryTypeAllocate);
  ref_count_ = new int(1);
}
Tensor::Tensor(Device *device, const TensorDesc &desc, void *data_ptr,
               const std::string &name, const base::IntVector &config)
    : desc_(desc), name_(name), is_external_(false) {
  BufferDesc buffer_desc = device->toBufferDesc(desc, config);
  buffer_ =
      new Buffer(device, buffer_desc, data_ptr, base::kMemoryTypeExternal);
  ref_count_ = new int(1);
}
Tensor::Tensor(MemoryPool *memory_pool, const TensorDesc &desc,
               const std::string &name, const base::IntVector &config)
    : desc_(desc), name_(name), is_external_(false) {
  Device *device = memory_pool->getDevice();
  BufferDesc buffer_desc = device->toBufferDesc(desc, config);
  void *ptr = memory_pool->allocate(buffer_desc);
  buffer_ =
      new Buffer(memory_pool, buffer_desc, ptr, base::kMemoryTypeAllocate);
  ref_count_ = new int(1);
}
Tensor::Tensor(MemoryPool *memory_pool, const TensorDesc &desc, void *data_ptr,
               const std::string &name, const base::IntVector &config)
    : desc_(desc), name_(name), is_external_(false) {
  Device *device = memory_pool->getDevice();
  BufferDesc buffer_desc = device->toBufferDesc(desc, config);
  buffer_ =
      new Buffer(memory_pool, buffer_desc, data_ptr, base::kMemoryTypeExternal);
  ref_count_ = new int(1);
}

Tensor::Tensor(const Tensor &tensor) {
  if (this == &tensor) {
    return;
  }
  name_ = tensor.name_;
  desc_ = tensor.desc_;
  is_external_ = tensor.is_external_;
  ref_count_ = tensor.ref_count_;
  if (ref_count_ != nullptr) {
    tensor.addRef();
  }
  buffer_ = tensor.buffer_;
}
Tensor &Tensor::operator=(const Tensor &tensor) {
  if (this == &tensor) {
    return *this;
  }
  name_ = tensor.name_;
  desc_ = tensor.desc_;
  is_external_ = tensor.is_external_;
  ref_count_ = tensor.ref_count_;
  if (ref_count_ != nullptr) {
    tensor.addRef();
  }
  buffer_ = tensor.buffer_;
  return *this;
}

Tensor::Tensor(Tensor &&tensor) noexcept {
  if (this == &tensor) {
    return;
  }
  name_ = tensor.name_;
  desc_ = std::move(tensor.desc_);
  is_external_ = tensor.is_external_;
  ref_count_ = tensor.ref_count_;
  buffer_ = tensor.buffer_;
  tensor.clear();
}
Tensor &Tensor::operator=(Tensor &&tensor) noexcept {
  if (this == &tensor) {
    return *this;
  }
  name_ = tensor.name_;
  desc_ = std::move(tensor.desc_);
  is_external_ = tensor.is_external_;
  ref_count_ = tensor.ref_count_;
  buffer_ = tensor.buffer_;
  tensor.clear();
  return *this;
}

Tensor::~Tensor() { this->clear(); }

// create
void Tensor::create(const std::string &name) {
  if (!this->empty()) {
    NNDEPLOY_LOGI("Tensor is not empty, can not create");
    return;
  }
  name_ = name;
}

void Tensor::create(const TensorDesc &desc, const std::string &name) {
  if (!this->empty()) {
    NNDEPLOY_LOGI("Tensor is not empty, can not create");
    return;
  }
  name_ = name;
  desc_ = desc;
}
void Tensor::create(const TensorDesc &desc, Buffer *buffer,
                    const std::string &name) {
  if (!this->empty()) {
    NNDEPLOY_LOGI("Tensor is not empty, can not create");
    return;
  }
  name_ = name;
  desc_ = desc;
  is_external_ = true;
  ref_count_ = new int(1);
  buffer_ = buffer;
}
void Tensor::create(Device *device, const TensorDesc &desc,
                    const std::string &name, const base::IntVector &config) {
  if (!this->empty()) {
    NNDEPLOY_LOGI("Tensor is not empty, can not create");
    return;
  }
  name_ = name;
  desc_ = desc;
  is_external_ = false;
  BufferDesc buffer_desc = device->toBufferDesc(desc, config);
  void *ptr = device->allocate(buffer_desc);
  buffer_ = new Buffer(device, buffer_desc, ptr, base::kMemoryTypeAllocate);
  ref_count_ = new int(1);
}
void Tensor::create(Device *device, const TensorDesc &desc, void *data_ptr,
                    const std::string &name, const base::IntVector &config) {
  if (!this->empty()) {
    NNDEPLOY_LOGI("Tensor is not empty, can not create");
    return;
  }
  name_ = name;
  desc_ = desc;
  is_external_ = false;
  BufferDesc buffer_desc = device->toBufferDesc(desc, config);
  buffer_ =
      new Buffer(device, buffer_desc, data_ptr, base::kMemoryTypeExternal);
  ref_count_ = new int(1);
}
void Tensor::create(MemoryPool *memory_pool, const TensorDesc &desc,
                    const std::string &name, const base::IntVector &config) {
  if (!this->empty()) {
    NNDEPLOY_LOGI("Tensor is not empty, can not create");
    return;
  }
  name_ = name;
  desc_ = desc;
  is_external_ = false;
  Device *device = memory_pool->getDevice();
  BufferDesc buffer_desc = device->toBufferDesc(desc, config);
  void *ptr = memory_pool->allocate(buffer_desc);
  buffer_ =
      new Buffer(memory_pool, buffer_desc, ptr, base::kMemoryTypeAllocate);
  ref_count_ = new int(1);
}
void Tensor::create(MemoryPool *memory_pool, const TensorDesc &desc,
                    void *data_ptr, const std::string &name,
                    const base::IntVector &config) {
  if (!this->empty()) {
    NNDEPLOY_LOGI("Tensor is not empty, can not create");
    return;
  }
  name_ = name;
  desc_ = desc;
  is_external_ = false;
  Device *device = memory_pool->getDevice();
  BufferDesc buffer_desc = device->toBufferDesc(desc, config);
  buffer_ =
      new Buffer(memory_pool, buffer_desc, data_ptr, base::kMemoryTypeExternal);
  ref_count_ = new int(1);
}

void Tensor::clear() {
  deallocate();

  name_.clear();

  desc_.data_type_ = base::dataTypeOf<float>();
  desc_.data_format_ = base::kDataFormatNotSupport;
  desc_.shape_.clear();
  desc_.stride_.clear();

  is_external_ = false;
}

void Tensor::allocate(Device *device, const base::IntVector &config) {
  BufferDesc dst_buffer_desc = device->toBufferDesc(desc_, config);
  if (buffer_ != nullptr && device == buffer_->getDevice()) {
    BufferDesc src_buffer_desc = buffer_->getDesc();
    if (src_buffer_desc >= dst_buffer_desc) {
      return;
    }
  }
  deallocate();
  is_external_ = false;
  void *ptr = device->allocate(dst_buffer_desc);
  buffer_ = new Buffer(device, dst_buffer_desc, ptr, base::kMemoryTypeAllocate);
  ref_count_ = new int(1);
}
void Tensor::allocate(MemoryPool *memory_pool, const base::IntVector &config) {
  Device *device = memory_pool->getDevice();
  BufferDesc dst_buffer_desc = device->toBufferDesc(desc_, config);
  if (buffer_ != nullptr) {
    BufferDesc src_buffer_desc = buffer_->getDesc();
    if (src_buffer_desc >= dst_buffer_desc) {
      return;
    }
  }
  deallocate();
  is_external_ = false;
  void *ptr = memory_pool->allocate(dst_buffer_desc);
  buffer_ =
      new Buffer(memory_pool, dst_buffer_desc, ptr, base::kMemoryTypeAllocate);
  ref_count_ = new int(1);
}
void Tensor::deallocate() {
  if (buffer_ != nullptr && ref_count_ != nullptr && this->subRef() == 1) {
    if (!is_external_) {
      delete buffer_;
    }
    delete ref_count_;
  }
  buffer_ = nullptr;
  ref_count_ = nullptr;
}

base::Status Tensor::reshape(base::IntVector shape) {
  if (shape.empty()) {
    NNDEPLOY_LOGE("shape is empty");
    return base::kStatusCodeErrorInvalidParam;
  }
  if (desc_.shape_ == shape) {
    return base::kStatusCodeOk;
  }
  if (buffer_ == nullptr) {
    desc_.shape_ = shape;
    return base::kStatusCodeOk;
  }
  if (desc_.shape_.size() != shape.size()) {
    NNDEPLOY_LOGE("shape size is not equal");
    return base::kStatusCodeErrorInvalidParam;
  }
  desc_.shape_ = shape;
  return base::kStatusCodeOk;
}
bool Tensor::justModify(const TensorDesc &desc) {
  if (buffer_ == nullptr) {
    desc_ = desc;
    return true;
  } else {
    // TODO, 做到可以安全修改
    Device *device = buffer_->getDevice();
    base::IntVector config = buffer_->getConfig();
    BufferDesc buffer_desc = device->toBufferDesc(desc, config);
    bool flag = buffer_->justModify(buffer_desc.getSize());
    if (flag) {
      desc_ = desc;
      return true;
    } else {
      return false;
    }
  }
}

bool Tensor::justModify(Buffer *buffer) {
  // TODO, 做到可以安全修改
  deallocate();
  is_external_ = true;
  buffer_ = buffer;
  ref_count_ = new int(1);
  return true;
}

Tensor *Tensor::clone() {
  std::string name = this->getName();
  Device *device = this->getDevice();
  TensorDesc desc = this->getDesc();
  Tensor *dst = new Tensor(device, desc, name);
  this->copyTo(dst);
  return dst;
}

base::Status Tensor::copyTo(Tensor *dst) {
  Buffer *src_buffer = this->getBuffer();
  Buffer *dst_buffer = dst->getBuffer();
  if (src_buffer == nullptr || dst_buffer == nullptr) {
    return base::kStatusCodeErrorNotImplement;
  }
  base::Status status = src_buffer->copyTo(dst_buffer);
  return status;
}

// 类似pytorch的打印函数
void Tensor::print() {
  std::string name = this->getName();
  base::DataType data_type = desc_.data_type_;
  std::cout << "Tensor: " << name << std::endl;
  std::cout << "device type: "
            << base::deviceTypeToString(this->getDeviceType()) << std::endl;
  std::cout << "ref_count: " << ref_count_[0] << std::endl;
  desc_.print();
  Device *host_device = getDefaultHostDevice();
  Buffer *host_buffer = nullptr;
  if (!device::isHostDeviceType(this->getDeviceType())) {
    host_buffer = new Buffer(host_device, this->toBufferDesc());
    if (host_buffer == nullptr) {
      NNDEPLOY_LOGE("host_buffer is empty");
      return;
    }
    buffer_->copyTo(host_buffer);
  } else {
    host_buffer = buffer_;
  }
  size_t size = host_buffer->getSize();
  size_t ele_size = data_type.size();
  size_t ele_count = size / ele_size;
  void *data = host_buffer->getData();

  if (data_type.code_ == base::kDataTypeCodeInt && data_type.bits_ == 8 &&
      data_type.lanes_ == 1) {
    base::printData((int8_t *)data, ele_count);
  } else if (data_type.code_ == base::kDataTypeCodeInt &&
             data_type.bits_ == 16 && data_type.lanes_ == 1) {
    base::printData((int16_t *)data, ele_count);
  } else if (data_type.code_ == base::kDataTypeCodeInt &&
             data_type.bits_ == 32 && data_type.lanes_ == 1) {
    base::printData((int32_t *)data, ele_count);
  } else if (data_type.code_ == base::kDataTypeCodeInt &&
             data_type.bits_ == 64 && data_type.lanes_ == 1) {
    base::printData((int64_t *)data, ele_count);
  } else if (data_type.code_ == base::kDataTypeCodeUint &&
             data_type.bits_ == 8 && data_type.lanes_ == 1) {
    base::printData((uint8_t *)data, ele_count);
  } else if (data_type.code_ == base::kDataTypeCodeUint &&
             data_type.bits_ == 16 && data_type.lanes_ == 1) {
    base::printData((uint16_t *)data, ele_count);
  } else if (data_type.code_ == base::kDataTypeCodeUint &&
             data_type.bits_ == 32 && data_type.lanes_ == 1) {
    base::printData((uint32_t *)data, ele_count);
  } else if (data_type.code_ == base::kDataTypeCodeUint &&
             data_type.bits_ == 64 && data_type.lanes_ == 1) {
    base::printData((uint64_t *)data, ele_count);
  } else if (data_type.code_ == base::kDataTypeCodeFp &&
             data_type.bits_ == 32 && data_type.lanes_ == 1) {
    base::printData((float *)data, ele_count);
  } else if (data_type.code_ == base::kDataTypeCodeFp &&
             data_type.bits_ == 64 && data_type.lanes_ == 1) {
    base::printData((double *)data, ele_count);
  } else if (data_type.code_ == base::kDataTypeCodeBFp &&
             data_type.bits_ == 16 && data_type.lanes_ == 1) {
    float *fp32 = (float *)malloc(ele_count * sizeof(float));
    base::convertFromBfp16ToFloat((void *)data, fp32, ele_count);
    base::printData((float *)fp32, ele_count);
    free(fp32);
  } else if (data_type.code_ == base::kDataTypeCodeFp &&
             data_type.bits_ == 16 && data_type.lanes_ == 1) {
    float *fp32 = (float *)malloc(ele_count * sizeof(float));
    base::convertFromFp16ToFloat((void *)data, fp32, ele_count);
    base::printData((float *)data, ele_count);
    free(fp32);
  } else {
    NNDEPLOY_LOGE("data type is not support");
  }

  if (!device::isHostDeviceType(this->getDeviceType())) {
    delete host_buffer;
  }

  return;
}

bool Tensor::isSameDevice(Tensor *tensor) const {
  Device *src = this->getDevice();
  Device *dst = this->getDevice();
  return src == dst;
}
bool Tensor::isSameMemoryPool(Tensor *tensor) const {
  MemoryPool *src = this->getMemoryPool();
  MemoryPool *dst = this->getMemoryPool();
  return src == dst;
}
bool Tensor::isSameDesc(Tensor *tensor) const {
  TensorDesc src = this->getDesc();
  TensorDesc dst = this->getDesc();
  return src == dst;
}

// get
bool Tensor::empty() const {
  bool name_flag = name_.empty();
  bool desc_flag = desc_.shape_.empty();
  bool buffer_flag = (ref_count_ == nullptr) && (buffer_ == nullptr);
  bool flag = name_flag && desc_flag && buffer_flag;
  return flag;
}
bool Tensor::isContinue() const {
  if (desc_.stride_.size() == 0) {
    return true;
  } else {
    int size = desc_.stride_.size();
    size_t acc = 1;
    for (int i = size - 1; i >= 0; --i) {
      acc *= desc_.shape_[i];
      if (desc_.stride_[i] != acc) {
        return false;
      }
    }
  }
  return true;
}
bool Tensor::isExternalBuffer() const { return is_external_; }
std::string Tensor::getName() const { return name_; }

TensorDesc Tensor::getDesc() const { return desc_; }
base::DataType Tensor::getDataType() const { return desc_.data_type_; }
void Tensor::setDataType(base::DataType data_type) {
  desc_.data_type_ = data_type;
}
base::DataFormat Tensor::getDataFormat() const { return desc_.data_format_; }
void Tensor::setDataFormat(base::DataFormat data_format) {
  desc_.data_format_ = data_format;
}
base::IntVector Tensor::getShape() const { return desc_.shape_; }
int Tensor::getShapeIndex(int index) const {
  if (index < desc_.shape_.size()) {
    return desc_.shape_[index];
  } else {
    return -1;
  }
}
int Tensor::getBatch() const {
  if (!desc_.shape_.empty()) {
    return desc_.shape_[0];
  } else {
    return -1;
  }
}
int Tensor::getChannel() const {
  int ret = -1;
  switch (desc_.data_format_) {
    case base::kDataFormatN:
      break;
    case base::kDataFormatNC:
      ret = desc_.shape_[1];
      break;
    case base::kDataFormatNCL:
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
int Tensor::getDepth() const {
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
int Tensor::getHeight() const {
  int ret = -1;
  switch (desc_.data_format_) {
    case base::kDataFormatN:
      break;
    case base::kDataFormatNC:
      break;
    case base::kDataFormatNCL:
      ret = desc_.shape_[1];
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
int Tensor::getWidth() const {
  int ret = -1;
  switch (desc_.data_format_) {
    case base::kDataFormatN:
      break;
    case base::kDataFormatNC:
      break;
    case base::kDataFormatNCL:
      ret = desc_.shape_[2];
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
base::SizeVector Tensor::getStride() const { return desc_.stride_; }
size_t Tensor::getStrideIndex(int index) const {
  if (index < desc_.stride_.size()) {
    return desc_.stride_[index];
  } else {
    return 0;
  }
}
Buffer *Tensor::getBuffer() const { return buffer_; }
base::DeviceType Tensor::getDeviceType() const {
  if (buffer_) {
    return buffer_->getDeviceType();
  } else {
    return base::DeviceType(base::kDeviceTypeCodeNotSupport);
  }
}
Device *Tensor::getDevice() const {
  if (buffer_) {
    return buffer_->getDevice();
  } else {
    return nullptr;
  }
}
MemoryPool *Tensor::getMemoryPool() const {
  if (buffer_) {
    return buffer_->getMemoryPool();
  } else {
    return nullptr;
  }
}
bool Tensor::isMemoryPool() const {
  if (buffer_) {
    return buffer_->isMemoryPool();
  } else {
    return false;
  }
}
BufferDesc Tensor::toBufferDesc() const {
  if (buffer_) {
    return buffer_->getDesc();
  } else {
    return BufferDesc();
  }
}
size_t Tensor::getSize() const {
  if (buffer_) {
    return buffer_->getSize();
  } else {
    return 0;
  }
}
base::SizeVector Tensor::getSizeVector() const {
  if (buffer_) {
    return buffer_->getSizeVector();
  } else {
    return base::SizeVector();
  }
}
size_t Tensor::getRealSize() const {
  if (buffer_) {
    return buffer_->getRealSize();
  } else {
    return 0;
  }
}
base::SizeVector Tensor::getRealSizeVector() const {
  if (buffer_) {
    return buffer_->getRealSizeVector();
  } else {
    return base::SizeVector();
  }
}
base::IntVector Tensor::getConfig() const {
  if (buffer_) {
    return buffer_->getConfig();
  } else {
    return base::IntVector();
  }
}
void *Tensor::getData() const {
  if (buffer_) {
    return buffer_->getData();
  } else {
    return nullptr;
  }
}
base::MemoryType Tensor::getMemoryType() const {
  if (buffer_) {
    return buffer_->getMemoryType();
  } else {
    return base::kMemoryTypeNone;
  }
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
