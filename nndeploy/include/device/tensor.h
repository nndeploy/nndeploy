
#ifndef _NNDEPLOY_INCLUDE_DEVICE_TENSOR_H_
#define _NNDEPLOY_INCLUDE_DEVICE_TENSOR_H_

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/log.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/base/value.h"
#include "nndeploy/include/device/buffer.h"

namespace nndeploy {
namespace device {

class Device;
struct TensorDesc {
  base::DataType data_type_;
  base::DataFormat format;
  base::IntVector shape_;
  base::SizeVector stride_;
};

class TensorImpl : public base::NonCopyable{
 public:
  TensorImpl();
  virtual ~TensorImpl();

  TensorImpl(Device *device, const TensorDesc &desc,
             const std::string &name = "",
             base::TensorImplType type = base::kTensorImplTypeDefault,
             const base::IntVector &config = base::IntVector());
  TensorImpl(BufferPool *buffer_pool, const TensorDesc &desc,
             const std::string &name = "",
             base::TensorImplType type = base::kTensorImplTypeDefault,
             const base::IntVector &config = base::IntVector());
  TensorImpl(TensorDesc desc, Buffer *buffer, const std::string &name = "",
             base::TensorImplType type = base::kTensorImplTypeDefault);

  // create
  void create(Device *device, const TensorDesc &desc,
              const std::string &name = "",
              base::TensorImplType type = base::kTensorImplTypeDefault,
              const base::IntVector &config = base::IntVector());
  void create(BufferPool *buffer_pool, const TensorDesc &desc,
              const std::string &name = "",
              base::TensorImplType type = base::kTensorImplTypeDefault,
              const base::IntVector &config = base::IntVector());

  // get
  bool empty();

  std::string getName();
  base::TensorImplType getTensorImplType();

  TensorDesc getDesc();
  base::DataType getDataType();
  base::DataFormat getDataFormat();
  base::IntVector getShape();
  int32_t getShapeIndex(int index);
  base::SizeVector getStride();
  size_t getStrideIndex(int index);

  Buffer *getBuffer();
  base::DeviceType getDeviceType();
  Device *getDevice();
  BufferPool *getBufferPool();
  bool isBufferPool();
  BufferDesc getBufferDesc();
  size_t getSize();
  base::SizeVector getSizeVector();
  base::IntVector getConfig();
  void *getPtr();
  int32_t getId();
  BufferSourceType getBufferSourceType();

 private:
  base::TensorImplType type_;
  std::string name_;
  TensorDesc desc_;
  device::Buffer *buffer_;
};

/**
 * @brief 需要扩张对量化的tensor支持
 *
 */
class Tensor : public base::NonCopyable {
 public:
  Tensor();
  virtual ~Tensor();

  Tensor(Device *device, const TensorDesc &desc, const std::string &name = "",
         base::TensorImplType type = base::kTensorImplTypeDefault,
         const base::IntVector &config = base::IntVector());
  Tensor(BufferPool *buffer_pool, const TensorDesc &desc,
         const std::string &name = "",
         base::TensorImplType type = base::kTensorImplTypeDefault,
         const base::IntVector &config = base::IntVector());
  Tensor(const TensorDesc &desc, Buffer *buffer, const std::string &name = "",
         base::TensorImplType type = base::kTensorImplTypeDefault);

  // create
  void create(Device *device, const TensorDesc &desc,
              const std::string &name = "",
              base::TensorImplType type = base::kTensorImplTypeDefault,
              const base::IntVector &config = base::IntVector());
  void create(BufferPool *buffer_pool, const TensorDesc &desc,
              const std::string &name = "",
              base::TensorImplType type = base::kTensorImplTypeDefault,
              const base::IntVector &config = base::IntVector());

  // get
  bool empty();

  std::string getName();
  base::TensorImplType getTensorImplType();

  TensorDesc getDesc();
  base::DataType getDataType();
  base::DataFormat getDataFormat();
  base::IntVector getShape();
  int32_t getShapeIndex(int index);
  base::SizeVector getStride();
  size_t getStrideIndex(int index);

  Buffer *getBuffer();
  base::DeviceType getDeviceType();
  Device *getDevice();
  BufferPool *getBufferPool();
  bool isBufferPool();
  BufferDesc getBufferDesc();
  size_t getSize();
  base::SizeVector getSizeVector();
  base::IntVector getConfig();
  void *getPtr();
  int32_t getId();
  BufferSourceType getBufferSourceType();

 private:
  TensorImpl *tensor_impl_ = nullptr;
};

class TensorPtrArray {
 public:
  TensorPtrArray();
  TensorPtrArray(std::vector<Tensor *> tensors_);
  TensorPtrArray(Tensor *tensor);

  virtual ~TensorPtrArray();

  void addTensor(Tensor *tensor);
  void addTensor(std::vector<Tensor *> tensors_);

  int getTensorSize();
  Tensor *getTensor();
  Tensor *getTensor(int index);

 private:
  std::vector<Tensor *> tensors_;
};

using TensorMap = std::map<std::string, std::shared_ptr<Tensor>>;

}  // namespace device
}  // namespace nndeploy

#endif
