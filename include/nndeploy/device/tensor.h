
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
  base::Status reshape(base::IntVector shape);
  bool justModify(const TensorDesc &desc);
  bool justModify(Buffer *buffer);

  // clone and copy
  Tensor *clone();
  // dst必须预先分配内存
  base::Status copyTo(Tensor *dst);

  // print
  void print();

  // bool
  bool isSameDevice(Tensor *tensor) const;
  bool isSameMemoryPool(Tensor *tensor) const;
  bool isSameDesc(Tensor *tensor) const;

  // get
  bool empty() const;
  bool isContinue() const;
  bool isExternalBuffer() const;

  std::string getName() const;
  TensorDesc getDesc() const;
  base::DataType getDataType() const;
  base::DataFormat getDataFormat() const;
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
  BufferDesc toBufferDesc() const;
  size_t getSize() const;
  base::SizeVector getSizeVector() const;
  base::IntVector getConfig() const;
  void *getData() const;
  base::MemoryType getMemoryType() const;

  inline int addRef() const { return NNDEPLOY_XADD(ref_count_, 1); }
  inline int subRef() const { return NNDEPLOY_XADD(ref_count_, -1); }

  friend Tensor &operator+(const Tensor &a, const Tensor &b);
  template <typename T>
  friend Tensor &operator+(const T &a, const Tensor &b);
  template <typename T>
  friend Tensor &operator+(const Tensor &a, const T &b);

  friend Tensor &operator-(const Tensor &a, const Tensor &b);
  template <typename T>
  friend Tensor &operator-(const T &a, const Tensor &b);
  template <typename T>
  friend Tensor &operator-(const Tensor &a, const T &b);

  friend Tensor &operator*(const Tensor &a, const Tensor &b);
  template <typename T>
  friend Tensor &operator*(const T &a, const Tensor &b);
  template <typename T>
  friend Tensor &operator*(const Tensor &a, const T &b);

  friend Tensor &operator/(const Tensor &a, const Tensor &b);
  template <typename T>
  friend Tensor &operator/(const T &a, const Tensor &b);
  template <typename T>
  friend Tensor &operator/(const Tensor &a, const T &b);

 private:
  std::string name_ = "";     // tensor name
  TensorDesc desc_;           // tensor desc
  bool is_external_ = false;  // is external
  int *ref_count_ = nullptr;  // 引用计数
  Buffer *buffer_ = nullptr;  // buffer
};

class TensorCreator {
 public:
  virtual ~TensorCreator(){};
  virtual Tensor *createTensor() = 0;
};

template <typename T>
class TypeTensorCreator : public TensorCreator {
  virtual Tensor *createTensor() { return new T(); }
};

std::map<base::TensorType, std::shared_ptr<TensorCreator>>
    &getGlobalTensorCreatorMap();

template <typename T>
class TypeTensorRegister {
 public:
  explicit TypeTensorRegister(base::TensorType type) {
    getGlobalTensorCreatorMap()[type] = std::shared_ptr<T>(new T());
  }
};

extern NNDEPLOY_CC_API Tensor *createTensor(base::TensorType type);

template <typename T, typename T1>
base::Status randnTensor(T &generator, T1 mean, T1 std,
                         device ::Tensor *tensor) {
  base::Status status = base::kStatusCodeOk;
  if (tensor == nullptr) {
    NNDEPLOY_LOGE("tensor is empty");
    return base::kStatusCodeErrorNullParam;
  }
  Device *host_device = getDefaultHostDevice();
  Buffer *host_buffer = nullptr;
  if (!device::isHostDeviceType(tensor->getDeviceType())) {
    host_buffer = new Buffer(host_device, tensor->toBufferDesc());
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
  generator.seed(std::random_device()());
  std::normal_distribution<T1> normal(mean, std);
  for (size_t i = 0; i < ele_count; ++i) {
    if (data_type.code_ == base::kDataTypeCodeBFp && data_type.bits_ == 16 &&
        data_type.lanes_ == 1) {
      float normal_g = normal(generator);
      base::convertFromFloatToBfp16(&normal_g, (void *)((uint16_t *)data + i),
                                    1);
    } else if (data_type.code_ == base::kDataTypeCodeFp &&
               data_type.bits_ == 16 && data_type.lanes_ == 1) {
      float normal_g = normal(generator);
      base::convertFromFloatToFp16(&normal_g, (void *)((uint16_t *)data + i),
                                   1);
    } else {
      ((T1 *)data)[i] = normal(generator);
    }
  }
  if (!device::isHostDeviceType(tensor->getDeviceType())) {
    status = host_buffer->copyTo(tensor->getBuffer());
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("copyTo failed!");
      return status;
    }
  }
  return status;
}

}  // namespace device
}  // namespace nndeploy

#endif
