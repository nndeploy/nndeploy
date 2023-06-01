
#ifndef _NNDEPLOY_SOURCE_DEVICE_TENSOR_H_
#define _NNDEPLOY_SOURCE_DEVICE_TENSOR_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/device/buffer.h"
#include "nndeploy/source/device/buffer_pool.h"
#include "nndeploy/source/device/device.h"

namespace nndeploy {
namespace device {

struct NNDEPLOY_CC_API TensorDesc {
  TensorDesc(){};
  explicit TensorDesc(base::DataType data_type, base::DataFormat format,
                      const base::IntVector &shape,
                      const base::SizeVector &stride)
      : data_type_(data_type),
        format_(format),
        shape_(shape),
        stride_(stride){};

  TensorDesc(const TensorDesc &desc) {
    this->data_type_ = desc.data_type_;
    this->format_ = desc.format_;
    this->shape_ = desc.shape_;
    this->stride_ = desc.stride_;
  };
  TensorDesc &operator=(const TensorDesc &desc) = default;

  virtual ~TensorDesc(){};

  base::DataType data_type_ = base::dataTypeOf<float>();
  base::DataFormat format_ = base::kDataFormatNotSupport;
  base::IntVector shape_;
  base::SizeVector stride_;
};

class NNDEPLOY_CC_API Tensor : public base::NonCopyable {
 public:
  Tensor();
  virtual ~Tensor();

  Tensor(const TensorDesc &desc, const std::string &name = "");

  Tensor(Device *device, const TensorDesc &desc, const std::string &name = "",
         const base::IntVector &config = base::IntVector());

  Tensor(Device *device, const TensorDesc &desc, void *data_ptr,
         const std::string &name = "",
         const base::IntVector &config = base::IntVector());
  Tensor(Device *device, const TensorDesc &desc, int32_t data_id,
         const std::string &name = "",
         const base::IntVector &config = base::IntVector());

  Tensor(const TensorDesc &desc, Buffer *buffer, const std::string &name = "");

  // create
  // 必须确保为空
  void create(const TensorDesc &desc, const std::string &name = "");

  void create(Device *device, const TensorDesc &desc,
              const std::string &name = "",
              const base::IntVector &config = base::IntVector());

  void create(Device *device, const TensorDesc &desc, void *data_ptr,
              const std::string &name = "",
              const base::IntVector &config = base::IntVector());
  void create(Device *device, const TensorDesc &desc, int32_t data_id,
              const std::string &name = "",
              const base::IntVector &config = base::IntVector());

  void create(const TensorDesc &desc, Buffer *buffer,
              const std::string &name = "");

  // destroy
  void destory();

  // alloc
  void allocBuffer(Device *device,
                   const base::IntVector &config = base::IntVector());
  void deallocateBuffer();

  // modify
  bool justModify(const TensorDesc &desc);
  bool justModify(Buffer *buffer);

  // get
  bool empty();

  std::string getName();

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
  //! internal function
  void create(Device *device, const TensorDesc &desc, Buffer *buffer,
              void *data_ptr, int32_t data_id, const std::string &name,
              const base::IntVector &config);

 private:
  std::string name_;
  TensorDesc desc_;
  bool is_external_buffer_ = false;
  Buffer *buffer_;
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

}  // namespace device
}  // namespace nndeploy

#endif
