
#include "nndeploy/device/tensor.h"

#include "nndeploy/base/shape.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/time_profiler.h"

namespace nndeploy {
namespace device {

static TypeTensorRegister<TypeTensorCreator<Tensor>> g_defalut_tensor_register(
    base::kTensorTypeDefault);

Tensor::Tensor() {}
Tensor::Tensor(const std::string &name) : name_(name){};
Tensor::Tensor(const TensorDesc &desc, const std::string &name)
    : name_(name), desc_(desc){};
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
    NNDEPLOY_LOGI("Tensor is not empty, can not create.\n");
    return;
  }
  name_ = name;
}

void Tensor::create(const TensorDesc &desc, const std::string &name) {
  if (!this->empty()) {
    NNDEPLOY_LOGI("Tensor is not empty, can not create.\n");
    return;
  }
  name_ = name;
  desc_ = desc;
}
void Tensor::create(const TensorDesc &desc, Buffer *buffer,
                    const std::string &name) {
  if (!this->empty()) {
    NNDEPLOY_LOGI("Tensor is not empty, can not create.\n");
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
    NNDEPLOY_LOGI("Tensor is not empty, can not create.\n");
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
    NNDEPLOY_LOGI("Tensor is not empty, can not create.\n");
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
    NNDEPLOY_LOGI("Tensor is not empty, can not create.\n");
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
    NNDEPLOY_LOGI("Tensor is not empty, can not create.\n");
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
    NNDEPLOY_LOGE("shape is empty.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  if (base::shapeEqual(desc_.shape_, shape, 0, -1)) {
    return base::kStatusCodeOk;
  }
  if (buffer_ == nullptr) {
    desc_.shape_ = shape;
    return base::kStatusCodeOk;
  }
  if (desc_.shape_.size() != shape.size()) {
    NNDEPLOY_LOGE("shape size is not equal.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  // bool max_flag = true;
  // for (int i = 0; i < desc_.shape_.size(); ++i) {
  //   if (desc_.shape_[i] < shape[i]) {
  //     max_flag = false;
  //     break;
  //   }
  // }
  // if (max_flag) {
  //   auto device = getDevice();
  //   auto buffer_desc = device->toBufferDesc(desc_, base::IntVector());
  //   if (!buffer_->justModify(buffer_desc)) {
  //     NNDEPLOY_LOGE("buffer_->justModify(buffer_desc) failed.\n");
  //     return base::kStatusCodeErrorInvalidParam;
  //   }
  //   desc_.shape_ = shape;
  //   return base::kStatusCodeOk;
  // } else {
  //   NNDEPLOY_LOGE("new shape is greater than old shape.\n");
  //   return base::kStatusCodeErrorInvalidParam;
  // }
  auto device = getDevice();
  auto buffer_desc = device->toBufferDesc(desc_, base::IntVector());
  if (!buffer_->justModify(buffer_desc)) {
    NNDEPLOY_LOGE("buffer_->justModify(buffer_desc) failed.\n");
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

bool Tensor::justModify(Buffer *buffer, bool is_external) {
  // TODO, 做到可以安全修改
  deallocate();
  is_external_ = is_external;
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
  if (src_buffer == nullptr) {
    NNDEPLOY_LOGE("tensor[%s] copyTo failed, src_buffer is nullptr.\n",
                  this->getName().c_str());
    return base::kStatusCodeErrorNotImplement;
  }
  if (dst_buffer == nullptr) {
    NNDEPLOY_LOGE("tensor[%s] copyTo failed, dst_buffer is nullptr.\n",
                  dst->getName().c_str());
    return base::kStatusCodeErrorNotImplement;
  }
  // base::Status status = src_buffer->copyTo(dst_buffer);
  // return status;
  Device *src_device = src_buffer->getDevice();
  base::DeviceType src_device_type = src_device->getDeviceType();
  Device *dst_device = dst_buffer->getDevice();
  base::DeviceType dst_device_type = dst_device->getDeviceType();
  void *src_data_ptr = src_buffer->getData();
  void *dst_data_ptr = dst_buffer->getData();
  BufferDesc src_buffer_desc = src_buffer->getDesc();
  BufferDesc dst_buffer_desc = dst_buffer->getDesc();
  size_t src_size = src_buffer_desc.getSize();
  size_t dst_size = dst_buffer_desc.getSize();
  size_t copy_size = std::min(src_size, dst_size);
  if (src_device_type == dst_device_type) {
    base::Status status =
        src_device->copy(src_data_ptr, dst_data_ptr, copy_size);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE(
          "src_device->copy(src_data_ptr, dst_data_ptr, copy_size) failed\n");
      return status;
    }
    return base::kStatusCodeOk;
  } else if (isHostDeviceType(src_device_type) &&
             isHostDeviceType(dst_device_type)) {
    return src_device->copy(src_data_ptr, dst_data_ptr, copy_size);
  } else if (isHostDeviceType(src_device_type) &&
             !isHostDeviceType(dst_device_type)) {
    return dst_device->upload(src_data_ptr, dst_data_ptr, copy_size);
  } else if (!isHostDeviceType(src_device_type) &&
             isHostDeviceType(dst_device_type)) {
    return src_device->download(src_data_ptr, dst_data_ptr, copy_size);
  } else {
    // NNDEPLOY_LOGE("Unsupported device type{%s->%s} for copy operation.",
    //               base::deviceTypeToString(src_device_type).c_str(),
    //               base::deviceTypeToString(dst_device_type).c_str());
    // return base::kStatusCodeErrorNotImplement;
    // For copying between different device types, use Host as an intermediate
    // std::string name = this->getName() + "_copyTo";
    // NNDEPLOY_TIME_POINT_START(name.c_str());
    Device *host_device = getDefaultHostDevice();
    if (!host_device) {
      NNDEPLOY_LOGE("Failed to get Host device for intermediate copy");
      return base::kStatusCodeErrorNotImplement;
    }

    // Create temporary Host buffer as intermediate
    Tensor *host_tensor =
        new Tensor(host_device, this->getDesc(), "temp_host_tensor");
    if (!host_tensor) {
      NNDEPLOY_LOGE("Failed to create temporary Host tensor");
      return base::kStatusCodeErrorOutOfMemory;
    }

    // First copy from source device to Host
    base::Status status = this->copyTo(host_tensor);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Failed to copy from source device to Host");
      delete host_tensor;
      return status;
    }

    // Then copy from Host to destination device
    status = host_tensor->copyTo(dst);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Failed to copy from Host to destination device");
      delete host_tensor;
      return status;
    }

    // Clean up temporary resources
    delete host_tensor;
    // NNDEPLOY_TIME_POINT_END(name.c_str());
    return status;
  }
}

// 序列化模型权重为二进制文件
base::Status Tensor::serialize(std::ostream &stream) {
  uint64_t name_size = name_.size();
  if (!stream.write(reinterpret_cast<const char *>(&name_size),
                    sizeof(name_size))) {
    return base::kStatusCodeErrorIO;
  }
  if (!stream.write(name_.c_str(), name_size)) {
    return base::kStatusCodeErrorIO;
  }

  desc_.serialize(stream);

  // 存在buffer不为空，但是大小为0的情况
  if (buffer_ != nullptr) {
    uint64_t buffer_size = buffer_->getRealSize();
    // NNDEPLOY_LOGE("%s,%ld.\n", name_.c_str(), buffer_size);
    if (!stream.write(reinterpret_cast<const char *>(&buffer_size),
                      sizeof(buffer_size))) {
      return base::kStatusCodeErrorIO;
    }
    if (buffer_size > 0) {
      base::Status status = buffer_->serialize(stream);
      NNDEPLOY_RETURN_VALUE_ON_NEQ(status, base::kStatusCodeOk, status,
                                   "buffer_->serialize(stream) failed!\n");
    }
  }
  return base::kStatusCodeOk;
}
#if ENABLE_NNDEPLOY_SAFETENSORS_CPP
base::Status Tensor::safetensorsDtype2Dtype(
    const safetensors::dtype &safetensors_data_type,
    base::DataType &data_type) {
  auto status = base::kStatusCodeOk;
  switch (safetensors_data_type) {
    case safetensors::kUINT8:
      data_type = base::dataTypeOf<uint8_t>();
      break;
    case safetensors::kINT8:
      data_type = base::dataTypeOf<int8_t>();
      break;
    case safetensors::kINT16:
      data_type = base::dataTypeOf<int16_t>();
      break;
    case safetensors::kUINT16:
      data_type = base::dataTypeOf<uint16_t>();
      break;
    case safetensors::kFLOAT16:
      data_type = base::DataType(base::kDataTypeCodeFp, 16);
      break;
    case safetensors::kBFLOAT16:
      data_type = base::DataType(base::kDataTypeCodeBFp, 16);
      break;
    case safetensors::kINT32:
      data_type = base::dataTypeOf<int32_t>();
      break;
    case safetensors::kUINT32:
      data_type = base::dataTypeOf<uint32_t>();
      break;
    case safetensors::kFLOAT32:
      data_type = base::dataTypeOf<float>();
      break;
    case safetensors::kFLOAT64:
      data_type = base::dataTypeOf<double>();
      break;
    case safetensors::kINT64:
      data_type = base::dataTypeOf<int64_t>();
      break;
    case safetensors::kUINT64:
      data_type = base::dataTypeOf<uint64_t>();
      break;
    case safetensors::kBOOL:
    default:
      NNDEPLOY_RETURN_VALUE_ON_NEQ(
          base::kStatusCodeErrorInvalidParam, base::kStatusCodeOk,
          base::kStatusCodeErrorInvalidParam,
          "Unsupported data_type to deserialize in !!");
  }
  return status;
}
#endif
#if ENABLE_NNDEPLOY_SAFETENSORS_CPP
base::Status Tensor::safetensorsShape2Shape(
    const std::vector<size_t> &safetensors_data_shape, base::IntVector &shape) {
  shape.clear();
  // for (const auto &it : safetensors_data_shape) {
  //   shape.emplace_back(it);
  // }
  for (int i = 0; i < safetensors_data_shape.size(); ++i) {
    shape.push_back(safetensors_data_shape[i]);
  }
  return base::kStatusCodeOk;
}
#endif
#if ENABLE_NNDEPLOY_SAFETENSORS_CPP
base::Status Tensor::dtype2SafetensorsDtype(
    const base::DataType &data_type,
    safetensors::dtype &safetensors_data_type) {
  auto status = base::kStatusCodeOk;
  switch (data_type.code_) {
    case base::DataTypeCode::kDataTypeCodeUint: {
      switch (data_type.bits_) {
        case 8:
          safetensors_data_type = safetensors::kUINT8;
          break;
        case 16:
          safetensors_data_type = safetensors::kUINT16;
          break;
        case 32:
          safetensors_data_type = safetensors::kUINT32;
          break;
        case 64:
          safetensors_data_type = safetensors::kUINT64;
          break;
        default:
          status = base::kStatusCodeErrorIO;
      }
      break;
    }

    case base::DataTypeCode::kDataTypeCodeInt: {
      switch (data_type.bits_) {
        case 8:
          safetensors_data_type = safetensors::kINT8;
          break;
        case 16:
          safetensors_data_type = safetensors::kINT16;
          break;
        case 32:
          safetensors_data_type = safetensors::kINT32;
          break;
        case 64:
          safetensors_data_type = safetensors::kINT64;
          break;
        default:
          status = base::kStatusCodeErrorIO;
      }
      break;
    }
    case base::DataTypeCode::kDataTypeCodeFp: {
      switch (data_type.bits_) {
        case 16:
          safetensors_data_type = safetensors::kFLOAT16;
          break;
        case 32:
          safetensors_data_type = safetensors::kFLOAT32;
          break;
        case 64:
          safetensors_data_type = safetensors::kFLOAT64;
          break;
        default:
          status = base::kStatusCodeErrorIO;
      }
      break;
    }
    case base::DataTypeCode::kDataTypeCodeBFp: {
      switch (data_type.bits_) {
        case 16:
          safetensors_data_type = safetensors::kBFLOAT16;
          break;
        default:
          status = base::kStatusCodeErrorIO;
      }
      break;
    }
    case base::DataTypeCode::kDataTypeCodeOpaqueHandle:
    case base::DataTypeCode::kDataTypeCodeNotSupport:
    default:
      status = base::kStatusCodeErrorIO;
  }
  return status;
}
#endif
#if ENABLE_NNDEPLOY_SAFETENSORS_CPP
base::Status Tensor::shape2SafetensorsShape(
    const base::IntVector &shape, std::vector<size_t> &safetensors_data_shape) {
  safetensors_data_shape.clear();
  for (const auto &it : shape) {
    safetensors_data_shape.emplace_back(static_cast<size_t>(it));
  }
  return base::kStatusCodeOk;
}
#endif
#if ENABLE_NNDEPLOY_SAFETENSORS_CPP
base::Status Tensor::serializeToSafetensors(safetensors::safetensors_t &st,
                                            bool serialize_buffer) {
  // NOTE: we should call not serialize_buffer at first time, then we serialize
  // buffer, so the second time we could say, the storage has already has
  // allocate the space for data.

  if (!serialize_buffer) {
    // serialize tensor and it's desc
    // safetensors::dtype dtype;
    // std::vector<size_t> shape;
    // std::array<size_t, 2> data_offsets;

    auto counted_size = [&st]() -> size_t {
      if (NNDEPLOY_LIKELY(st.tensors.size() > 0)) {
        safetensors::tensor_t pre_t;
        st.tensors.at(st.tensors.size() - 1, &pre_t);
        return pre_t.data_offsets[1];
      } else {
        return 0;
      }
    };

    safetensors::tensor_t t_t;
    dtype2SafetensorsDtype(desc_.data_type_, t_t.dtype);
    auto status = shape2SafetensorsShape(desc_.shape_, t_t.shape);
    std::string err_msg = std::string(
                              "data type %s, do not supported to be transfered "
                              "to safetensors !") +
                          dataTypeToString(desc_.data_type_).c_str();
    NNDEPLOY_RETURN_VALUE_ON_NEQ(status, base::kStatusCodeOk, status,
                                 err_msg.c_str());
    auto tensor_size = safetensors::get_dtype_bytes(t_t.dtype) *
                       safetensors::get_shape_size(t_t);
    auto pre_offsets = counted_size();
    t_t.data_offsets = {pre_offsets, pre_offsets + tensor_size};
    NNDEPLOY_LOGD("name : %s, data_offsets : %d, %d\n", name_, pre_offsets,
                  pre_offsets + tensor_size);
    st.tensors.insert(name_, std::move(t_t));
  } else {
    safetensors::tensor_t t_t;
    st.tensors.at(name_, &t_t);
    auto status = buffer_->serializeToSafetensors(st, t_t);
    NNDEPLOY_RETURN_VALUE_ON_NEQ(
        status, base::kStatusCodeOk, status,
        "buffer_->serializeToSafetensors(st, t_t) failed");
  }

  return base::kStatusCodeOk;
}
#endif
// 从二进制文件反序列化模型权重
base::Status Tensor::deserialize(std::istream &stream) {
  uint64_t name_size = 0;
  if (!stream.read(reinterpret_cast<char *>(&name_size), sizeof(name_size))) {
    return base::kStatusCodeErrorIO;
  }
  char *name_data = new char[name_size + 1];
  if (!stream.read(name_data, name_size)) {
    delete[] name_data;
    return base::kStatusCodeErrorIO;
  }
  name_data[name_size] = '\0';
  name_ = name_data;
  delete[] name_data;

  desc_.deserialize(stream);

  uint64_t buffer_size = 0;
  if (!stream.read(reinterpret_cast<char *>(&buffer_size),
                   sizeof(buffer_size))) {
    return base::kStatusCodeErrorIO;
  }
  // NNDEPLOY_LOGE("%s,%ld.\n", name_.c_str(), buffer_size);
  if (buffer_size > 0) {
    Device *device = getDefaultHostDevice();
    buffer_ = new Buffer(device, buffer_size);
    base::Status status = buffer_->deserialize(stream);
    if (status != base::kStatusCodeOk) {
      delete buffer_;
      NNDEPLOY_LOGE("buffer_->deserialize(stream) failed!\n");
      return base::kStatusCodeErrorIO;
    }
  }

  is_external_ = false;
  ref_count_ = new int(1);

  return base::kStatusCodeOk;
}

#if ENABLE_NNDEPLOY_SAFETENSORS_CPP
base::Status Tensor::serializeFromSafetensors(
    const safetensors::safetensors_t &st) {
  auto status = base::kStatusCodeOk;
  safetensors::tensor_t t_t;
  st.tensors.at(name_, &t_t);
  safetensorsShape2Shape(t_t.shape, this->desc_.shape_);
  safetensorsDtype2Dtype(t_t.dtype, this->desc_.data_type_);
  if (this->ref_count_) {
    delete this->ref_count_;
  }
  if (this->buffer_) {
    delete this->buffer_;
  }
  this->ref_count_ = new int(1);
  const char *data_ptr = reinterpret_cast<const char *>(
      st.databuffer_addr);  // NOTE: cause it is loaded by mmap
  // const char *data_ptr = reinterpret_cast<const char *>(st.storage.data());
  // @Realtyxxx
  this->buffer_ = new Buffer(
      getDefaultHostDevice(), t_t.data_offsets[1] - t_t.data_offsets[0],
      (void *)(data_ptr + t_t.data_offsets[0]), base::kMemoryTypeExternal);
  // this->buffer_->serializeFromSafetensors(
  //     data_ptr + t_t.data_offsets[0],
  //     t_t.data_offsets[1] - t_t.data_offsets[0]);
  return status;
}
#endif

// 类似pytorch的打印函数
void Tensor::print(std::ostream &stream) const {
  std::string name = this->getName();
  base::DataType data_type = desc_.data_type_;
  stream << "Tensor: " << name << std::endl;
  stream << "device type: " << base::deviceTypeToString(this->getDeviceType())
         << std::endl;
  desc_.print(stream);
  stream << std::endl;

  if (ref_count_ != nullptr && buffer_ != nullptr) {
    stream << "ref_count: " << ref_count_[0] << std::endl;
    Device *host_device = getDefaultHostDevice();
    Buffer *host_buffer = nullptr;
    if (!device::isHostDeviceType(this->getDeviceType())) {
      host_buffer = new Buffer(host_device, this->getBufferDesc());
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
    base::IntVector shape = desc_.shape_;
    void *data = host_buffer->getData();

    if (data_type.code_ == base::kDataTypeCodeInt && data_type.bits_ == 8 &&
        data_type.lanes_ == 1) {
      base::printData((int8_t *)data, shape, stream);
    } else if (data_type.code_ == base::kDataTypeCodeInt &&
               data_type.bits_ == 16 && data_type.lanes_ == 1) {
      base::printData((int16_t *)data, shape, stream);
    } else if (data_type.code_ == base::kDataTypeCodeInt &&
               data_type.bits_ == 32 && data_type.lanes_ == 1) {
      base::printData((int32_t *)data, shape, stream);
    } else if (data_type.code_ == base::kDataTypeCodeInt &&
               data_type.bits_ == 64 && data_type.lanes_ == 1) {
      base::printData((int64_t *)data, shape, stream);
    } else if (data_type.code_ == base::kDataTypeCodeUint &&
               data_type.bits_ == 8 && data_type.lanes_ == 1) {
      base::printData((uint8_t *)data, shape, stream);
    } else if (data_type.code_ == base::kDataTypeCodeUint &&
               data_type.bits_ == 16 && data_type.lanes_ == 1) {
      base::printData((uint16_t *)data, shape, stream);
    } else if (data_type.code_ == base::kDataTypeCodeUint &&
               data_type.bits_ == 32 && data_type.lanes_ == 1) {
      base::printData((uint32_t *)data, shape, stream);
    } else if (data_type.code_ == base::kDataTypeCodeUint &&
               data_type.bits_ == 64 && data_type.lanes_ == 1) {
      base::printData((uint64_t *)data, shape, stream);
    } else if (data_type.code_ == base::kDataTypeCodeFp &&
               data_type.bits_ == 32 && data_type.lanes_ == 1) {
      base::printData((float *)data, shape, stream);
    } else if (data_type.code_ == base::kDataTypeCodeFp &&
               data_type.bits_ == 64 && data_type.lanes_ == 1) {
      base::printData((double *)data, shape, stream);
    } else if (data_type.code_ == base::kDataTypeCodeBFp &&
               data_type.bits_ == 16 && data_type.lanes_ == 1) {
      float *fp32 = (float *)malloc(ele_count * sizeof(float));
      base::convertFromBfp16ToFloat((void *)data, fp32, ele_count);
      base::printData((float *)fp32, shape, stream);
      free(fp32);
    } else if (data_type.code_ == base::kDataTypeCodeFp &&
               data_type.bits_ == 16 && data_type.lanes_ == 1) {
      float *fp32 = (float *)malloc(ele_count * sizeof(float));
      base::convertFromFp16ToFloat((void *)data, fp32, ele_count);
      base::printData((float *)fp32, shape, stream);
      free(fp32);
    } else {
      NNDEPLOY_LOGE("data type is not support");
    }

    if (!device::isHostDeviceType(this->getDeviceType())) {
      delete host_buffer;
    }
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
base::Status Tensor::setName(const std::string &name) {
  name_ = name;
  return base::kStatusCodeOk;
}

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
BufferDesc Tensor::getBufferDesc() const {
  if (buffer_) {
    return buffer_->getDesc();
  } else {
    return BufferDesc();
  }
}
size_t Tensor::getSize() const {
  BufferDesc buffer_desc = getBufferDesc();
  if (buffer_) {
    return buffer_desc.getSize();
  } else {
    return 0;
  }
}
base::SizeVector Tensor::getSizeVector() const {
  BufferDesc buffer_desc = getBufferDesc();
  if (buffer_) {
    return buffer_desc.getSizeVector();
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
