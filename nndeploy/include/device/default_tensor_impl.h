
#ifndef _NNDEPLOY_INCLUDE_DEVICE_DEFALUT_TENSOR_IMPL_H_
#define _NNDEPLOY_INCLUDE_DEVICE_DEFALUT_TENSOR_IMPL_H_

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/log.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/base/value.h"
#include "nndeploy/include/device/buffer.h"
#include "nndeploy/include/device/buffer_pool.h"

namespace nndeploy {
namespace device {

class Device;

struct TensorDesc {
  TensorDesc(){};
  explicit TensorDesc(base::DataType data_type, base::DataFormat format,
                      const base::IntVector &shape,
                      const base::SizeVector &stride)
      : data_type_(data_type),
        format_(format),
        shape_(shape),
        stride_(stride){};

  TensorDesc(const TensorDesc &desc) = default;
  TensorDesc &operator=(const TensorDesc &desc) = default;

  virtual ~TensorDesc(){};

  base::DataType data_type_ = base::DataTypeOf<float>();
  base::DataFormat format_ = base::kDataFormatNotSupport;
  base::IntVector shape_;
  base::SizeVector stride_;
};

class DefaultTensorImpl : public base::NonCopyable {
 public:
  DefaultTensorImpl();
  virtual ~DefaultTensorImpl();

  DefaultTensorImpl(const TensorDesc &desc, const std::string &name = "");

  DefaultTensorImpl(const TensorDesc &desc, Buffer *buffer,
                    const std::string &name = "");

  DefaultTensorImpl(Device *device, const TensorDesc &desc,
                    const std::string &name = "",
                    const base::IntVector &config = base::IntVector());
  DefaultTensorImpl(BufferPool *buffer_pool, const TensorDesc &desc,
                    const std::string &name = "",
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

  void create(Device *device, BufferPool *buffer_pool, const TensorDesc &desc,
              Buffer *buffer, const std::string &name,
              const base::IntVector &config);

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
  std::string name_;
  TensorDesc desc_;
  bool is_external_buffer_ = false;
  Buffer *buffer_;
};

class TensorCreator {
 public:
  virtual ~TensorCreator(){};
  virtual DefaultTensorImpl *createTensor() = 0;
};

template <typename T>
class TypeTensorCreator : public TensorCreator {
  virtual DefaultTensorImpl *createTensor() { return new T(); }
};

std::map<base::TensorImplType, std::shared_ptr<TensorCreator>>
    &getGlobalTensorCreatorMap();

template <typename T>
class TypeTensorRegister {
 public:
  explicit TypeTensorRegister(base::TensorImplType type) {
    getGlobalTensorCreatorMap()[type] = std::shared_ptr<T>(new T());
  }
};

DefaultTensorImpl *createTensor(base::TensorImplType type);

}  // namespace device
}  // namespace nndeploy

#endif
