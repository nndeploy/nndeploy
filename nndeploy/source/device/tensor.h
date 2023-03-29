
#ifndef _NNDEPLOY_SOURCE_DEVICE_TENSOR_H_
#define _NNDEPLOY_SOURCE_DEVICE_TENSOR_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/base/value.h"
#include "nndeploy/source/device/buffer.h"
#include "nndeploy/source/device/default_tensor_impl.h"

namespace nndeploy {
namespace device {

class Device;

/**
 * @brief 需要扩张对量化的tensor支持
 *
 */
class NNDEPLOY_CC_API Tensor : public base::NonCopyable {
 public:
  Tensor(base::TensorImplType type = base::kTensorImplTypeDefault);
  virtual ~Tensor();

  Tensor(const TensorDesc &desc, const std::string &name = "",
         base::TensorImplType type = base::kTensorImplTypeDefault);

  Tensor(const TensorDesc &desc, Buffer *buffer, const std::string &name = "",
         base::TensorImplType type = base::kTensorImplTypeDefault);

  Tensor(Device *device, const TensorDesc &desc, const std::string &name = "",
         base::TensorImplType type = base::kTensorImplTypeDefault,
         const base::IntVector &config = base::IntVector());
  Tensor(BufferPool *buffer_pool, const TensorDesc &desc,
         const std::string &name = "",
         base::TensorImplType type = base::kTensorImplTypeDefault,
         const base::IntVector &config = base::IntVector());

  // create
  void create(const TensorDesc &desc, const std::string &name = "");
  void create(const TensorDesc &desc, Buffer *buffer,
              const std::string &name = "");
  void create(Device *device, const TensorDesc &desc,
              const std::string &name = "",
              const base::IntVector &config = base::IntVector());
  void create(BufferPool *buffer_pool, const TensorDesc &desc,
              const std::string &name = "",
              const base::IntVector &config = base::IntVector());

  void destory();

  void allocBuffer(Device *device,
                   const base::IntVector &config = base::IntVector());
  void allocBuffer(BufferPool *buffer_pool,
                   const base::IntVector &config = base::IntVector());
  void deallocateBuffer();

  bool justModify(const TensorDesc &desc);

  bool justModify(Buffer *buffer);

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
  DefaultTensorImpl *tensor_impl_ = nullptr;
};

class TensorPtrArray {
 public:
  TensorPtrArray();
  TensorPtrArray(const std::vector<Tensor *> &tensors);
  TensorPtrArray(Tensor *tensor);
  TensorPtrArray(Tensor &tensor);

  virtual ~TensorPtrArray();

  void add(Tensor *tensor);
  void add(const std::vector<Tensor *> &tensors);
  void add(Tensor &tensor);

  bool empty();
  int getSize();
  Tensor *get();
  Tensor *get(int index);

 private:
  std::vector<Tensor *> tensors_;
};

using TensorMap = std::map<std::string, std::shared_ptr<Tensor>>;

}  // namespace device
}  // namespace nndeploy

#endif
