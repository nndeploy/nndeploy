
#ifndef _NNDEPLOY_DEVICE_TENSOR_H_
#define _NNDEPLOY_DEVICE_TENSOR_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/half.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#if ENABLE_NNDEPLOY_SAFETENSORS_CPP
#include "safetensors.hh"
#endif

namespace nndeploy {
namespace device {

/**
 * @brief Tensor类
 *
 */
class NNDEPLOY_CC_API Tensor {
 public:
  Tensor();
  Tensor(const std::string &name);
  Tensor(const TensorDesc &desc, const std::string &name = "");
  Tensor(const TensorDesc &desc, Buffer *buffer, const std::string &name = "");

  Tensor(Device *device, const TensorDesc &desc, const std::string &name = "",
         const base::IntVector &config = base::IntVector());
  Tensor(Device *device, const TensorDesc &desc, void *data_ptr,
         const std::string &name = "",
         const base::IntVector &config = base::IntVector());

  Tensor(MemoryPool *memory_pool, const TensorDesc &desc,
         const std::string &name = "",
         const base::IntVector &config = base::IntVector());
  Tensor(MemoryPool *memory_pool, const TensorDesc &desc, void *data_ptr,
         const std::string &name = "",
         const base::IntVector &config = base::IntVector());

  Tensor(const Tensor &tensor);
  Tensor &operator=(const Tensor &tensor);

  Tensor(Tensor &&tensor) noexcept;
  Tensor &operator=(Tensor &&tensor) noexcept;

  virtual ~Tensor();

  void create(const std::string &name);

  void create(const TensorDesc &desc, const std::string &name = "");
  void create(const TensorDesc &desc, Buffer *buffer,
              const std::string &name = "");

  void create(Device *device, const TensorDesc &desc,
              const std::string &name = "",
              const base::IntVector &config = base::IntVector());
  void create(Device *device, const TensorDesc &desc, void *data_ptr,
              const std::string &name = "",
              const base::IntVector &config = base::IntVector());

  void create(MemoryPool *memory_pool, const TensorDesc &desc,
              const std::string &name = "",
              const base::IntVector &config = base::IntVector());
  void create(MemoryPool *memory_pool, const TensorDesc &desc, void *data_ptr,
              const std::string &name = "",
              const base::IntVector &config = base::IntVector());

  // clear tensor
  void clear();

  // alloc
  void allocate(Device *device,
                const base::IntVector &config = base::IntVector());
  void allocate(MemoryPool *memory_pool,
                const base::IntVector &config = base::IntVector());
  void deallocate();

  template <typename T>
  base::Status set(T value) {
    if (buffer_ == nullptr) {
      NNDEPLOY_LOGE("buffer_ is empty");
      return base::kStatusCodeErrorNullParam;
    }

    return buffer_->set(value);
  }

  // modify
  /**
   * @brief
   *
   * @param shape
   * @return base::Status
   * @note 三种情况
   * # buffer为空，直接reshape
   * #
   * buffer不为空，reshape后的buffer空间小于或当前buffer的空间，reshape并且更新buffer
   * # buffer不为空，
   */
  base::Status reshape(base::IntVector shape);
  bool justModify(const TensorDesc &desc);
  bool justModify(Buffer *buffer, bool is_external = true);

  // clone and copy
  Tensor *clone();
  // dst必须预先分配内存
  base::Status copyTo(Tensor *dst);

  // 序列化模型权重为二进制文件
  base::Status serialize(std::ostream &stream);

#if ENABLE_NNDEPLOY_SAFETENSORS_CPP
  base::Status serializeToSafetensors(safetensors::safetensors_t &st,
                                      bool serialize_buffer = false);
#endif

  // 从二进制文件反序列化模型权重
  base::Status deserialize(std::istream &stream);

#if ENABLE_NNDEPLOY_SAFETENSORS_CPP
  base::Status serializeFromSafetensors(const safetensors::safetensors_t &st);
#endif

  // print
  void print(std::ostream &stream = std::cout) const;

  // bool
  bool isSameDevice(Tensor *tensor) const;
  bool isSameMemoryPool(Tensor *tensor) const;
  bool isSameDesc(Tensor *tensor) const;

  // get
  bool empty() const;
  bool isContinue() const;
  bool isExternalBuffer() const;

  std::string getName() const;
  base::Status setName(const std::string &);
  TensorDesc getDesc() const;
  base::DataType getDataType() const;
  void setDataType(base::DataType data_type);
  base::DataFormat getDataFormat() const;
  void setDataFormat(base::DataFormat data_format);
  base::IntVector getShape() const;
  int getShapeIndex(int index) const;
  int getBatch() const;
  int getChannel() const;
  int getDepth() const;
  int getHeight() const;
  int getWidth() const;
  base::SizeVector getStride() const;
  size_t getStrideIndex(int index) const;

  Buffer *getBuffer() const;
  base::DeviceType getDeviceType() const;
  Device *getDevice() const;
  MemoryPool *getMemoryPool() const;
  bool isMemoryPool() const;
  BufferDesc getBufferDesc() const;
  size_t getSize() const;
  base::SizeVector getSizeVector() const;
  size_t getRealSize() const;
  base::SizeVector getRealSizeVector() const;
  base::IntVector getConfig() const;
  void *getData() const;
  base::MemoryType getMemoryType() const;

#if ENABLE_NNDEPLOY_SAFETENSORS_CPP
  static base::Status dtype2SafetensorsDtype(
      const base::DataType &data_type,
      safetensors::dtype &safetensors_data_type);
  static base::Status shape2SafetensorsShape(
      const base::IntVector &shape,
      std::vector<size_t> &safetensors_data_shape);

  static base::Status safetensorsDtype2Dtype(
      const safetensors::dtype &safetensors_data_type,
      base::DataType &data_type);

  static base::Status safetensorsShape2Shape(
      const std::vector<size_t> &safetensors_data_shape,
      base::IntVector &shape);
#endif

  inline int addRef() const { return NNDEPLOY_XADD(ref_count_, 1); }
  inline int subRef() const { return NNDEPLOY_XADD(ref_count_, -1); }

 private:
  std::string name_ = "";     // tensor name
  TensorDesc desc_;           // tensor desc
  bool is_external_ = false;  // is external
  int *ref_count_ = nullptr;  // 引用计数
  Buffer *buffer_ = nullptr;  // buffer
  // bool is_quant_ = false;
  // Buffer *scale_ = nullptr;
  // Buffer *zero_point_ = nullptr;
};

class TensorCreator {
 public:
  virtual ~TensorCreator() {};
  virtual Tensor *createTensor() = 0;
};

template <typename T>
class TypeTensorCreator : public TensorCreator {
  virtual Tensor *createTensor() { return new T(); }
};

std::map<base::TensorType, std::shared_ptr<TensorCreator>> &
getGlobalTensorCreatorMap();

template <typename T>
class TypeTensorRegister {
 public:
  explicit TypeTensorRegister(base::TensorType type) {
    getGlobalTensorCreatorMap()[type] = std::shared_ptr<T>(new T());
  }
};

extern NNDEPLOY_CC_API Tensor *createTensor(base::TensorType type);

template <typename T>
base::Status randnTensor(T &generator, float mean, float std, Tensor *tensor,
                         int64_t seed = -1) {
  base::Status status = base::kStatusCodeOk;
  if (tensor == nullptr) {
    NNDEPLOY_LOGE("tensor is empty");
    return base::kStatusCodeErrorNullParam;
  }
  Device *host_device = getDefaultHostDevice();
  Buffer *host_buffer = nullptr;
  if (!device::isHostDeviceType(tensor->getDeviceType())) {
    host_buffer = new Buffer(host_device, tensor->getBufferDesc());
    if (host_buffer == nullptr) {
      NNDEPLOY_LOGE("host_buffer is empty");
      return base::kStatusCodeErrorNullParam;
    }
  } else {
    host_buffer = tensor->getBuffer();
  }
  size_t size = host_buffer->getSize();
  base::DataType data_type = tensor->getDataType();
  size_t ele_size = data_type.size();
  size_t ele_count = size / ele_size;
  void *data = host_buffer->getData();
  if (seed == -1) {
    generator.seed(std::random_device()());
  } else {
    generator.seed(seed);
  }
  std::normal_distribution<float> normal(mean, std);
  if (data_type.code_ == base::kDataTypeCodeInt && data_type.bits_ == 8 &&
      data_type.lanes_ == 1) {
    for (size_t i = 0; i < ele_count; ++i) {
      ((int8_t *)data)[i] = (int8_t)(normal(generator));
    }
  } else if (data_type.code_ == base::kDataTypeCodeInt &&
             data_type.bits_ == 16 && data_type.lanes_ == 1) {
    for (size_t i = 0; i < ele_count; ++i) {
      ((int16_t *)data)[i] = (int16_t)(normal(generator));
    }
  } else if (data_type.code_ == base::kDataTypeCodeInt &&
             data_type.bits_ == 32 && data_type.lanes_ == 1) {
    for (size_t i = 0; i < ele_count; ++i) {
      ((int32_t *)data)[i] = (int32_t)(normal(generator));
    }
  } else if (data_type.code_ == base::kDataTypeCodeInt &&
             data_type.bits_ == 64 && data_type.lanes_ == 1) {
    for (size_t i = 0; i < ele_count; ++i) {
      ((int64_t *)data)[i] = (int64_t)(normal(generator));
    }
  } else if (data_type.code_ == base::kDataTypeCodeUint &&
             data_type.bits_ == 8 && data_type.lanes_ == 1) {
    for (size_t i = 0; i < ele_count; ++i) {
      ((uint8_t *)data)[i] = (uint8_t)(normal(generator));
    }
  } else if (data_type.code_ == base::kDataTypeCodeUint &&
             data_type.bits_ == 16 && data_type.lanes_ == 1) {
    for (size_t i = 0; i < ele_count; ++i) {
      ((uint16_t *)data)[i] = (uint16_t)(normal(generator));
    }
  } else if (data_type.code_ == base::kDataTypeCodeUint &&
             data_type.bits_ == 32 && data_type.lanes_ == 1) {
    for (size_t i = 0; i < ele_count; ++i) {
      ((uint32_t *)data)[i] = (uint32_t)(normal(generator));
    }
  } else if (data_type.code_ == base::kDataTypeCodeUint &&
             data_type.bits_ == 64 && data_type.lanes_ == 1) {
    for (size_t i = 0; i < ele_count; ++i) {
      ((uint64_t *)data)[i] = (uint64_t)(normal(generator));
    }
  } else if (data_type.code_ == base::kDataTypeCodeFp &&
             data_type.bits_ == 32 && data_type.lanes_ == 1) {
    for (size_t i = 0; i < ele_count; ++i) {
      ((float *)data)[i] = (float)(normal(generator));
    }
  } else if (data_type.code_ == base::kDataTypeCodeFp &&
             data_type.bits_ == 64 && data_type.lanes_ == 1) {
    for (size_t i = 0; i < ele_count; ++i) {
      ((double *)data)[i] = (double)(normal(generator));
    }
  } else if (data_type.code_ == base::kDataTypeCodeBFp &&
             data_type.bits_ == 16 && data_type.lanes_ == 1) {
    float *fp32 = (float *)malloc(ele_count * sizeof(float));
    for (size_t i = 0; i < ele_count; ++i) {
      ((float *)fp32)[i] = (float)(normal(generator));
    }
    base::convertFromFloatToBfp16(fp32, (void *)data, ele_count);
    free(fp32);
  } else if (data_type.code_ == base::kDataTypeCodeFp &&
             data_type.bits_ == 16 && data_type.lanes_ == 1) {
    float *fp32 = (float *)malloc(ele_count * sizeof(float));
    for (size_t i = 0; i < ele_count; ++i) {
      ((float *)fp32)[i] = (float)(normal(generator));
    }
    base::convertFromFloatToFp16(fp32, (void *)data, ele_count);
    free(fp32);
  } else {
    NNDEPLOY_LOGE("data type is not support");
  }

  if (!device::isHostDeviceType(tensor->getDeviceType())) {
    status = host_buffer->copyTo(tensor->getBuffer());
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "copyTo failed!");
    delete host_buffer;
  }
  return status;
}

}  // namespace device
}  // namespace nndeploy

#endif
