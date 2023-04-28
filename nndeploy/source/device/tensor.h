
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

struct NNDEPLOY_CC_API TensorImplDesc {
  TensorImplDesc(){};
  explicit TensorImplDesc(base::DataType data_type, base::DataFormat format,
                          const base::IntVector &shape,
                          const base::SizeVector &stride)
      : data_type_(data_type),
        format_(format),
        shape_(shape),
        stride_(stride){};

  TensorImplDesc(const TensorImplDesc &desc) = default;
  TensorImplDesc &operator=(const TensorImplDesc &desc) = default;

  virtual ~TensorImplDesc(){};

  base::DataType data_type_ = base::dataTypeOf<float>();
  base::DataFormat format_ = base::kDataFormatNotSupport;
  base::IntVector shape_;
  base::SizeVector stride_;
};

class NNDEPLOY_CC_API DefaultTensorImpl : public base::NonCopyable {
 public:
  DefaultTensorImpl();
  virtual ~DefaultTensorImpl();

  DefaultTensorImpl(const TensorImplDesc &desc, const std::string &name = "");

  DefaultTensorImpl(Device *device, const TensorImplDesc &desc,
                    const std::string &name = "",
                    const base::IntVector &config = base::IntVector());

  DefaultTensorImpl(Device *device, const TensorImplDesc &desc, void *data_ptr,
                    const std::string &name = "",
                    const base::IntVector &config = base::IntVector());
  DefaultTensorImpl(Device *device, const TensorImplDesc &desc, int32_t data_id,
                    const std::string &name = "",
                    const base::IntVector &config = base::IntVector());

  DefaultTensorImpl(const TensorImplDesc &desc, Buffer *buffer,
                    const std::string &name = "");

  // create
  // 必须确保为空
  void create(const TensorImplDesc &desc, const std::string &name = "");

  void create(Device *device, const TensorImplDesc &desc,
              const std::string &name = "",
              const base::IntVector &config = base::IntVector());

  void create(Device *device, const TensorImplDesc &desc, void *data_ptr,
              const std::string &name = "",
              const base::IntVector &config = base::IntVector());
  void create(Device *device, const TensorImplDesc &desc, int32_t data_id,
              const std::string &name = "",
              const base::IntVector &config = base::IntVector());

  void create(const TensorImplDesc &desc, Buffer *buffer,
              const std::string &name = "");

  // destroy
  void destory();

  // alloc
  void allocBuffer(Device *device,
                   const base::IntVector &config = base::IntVector());
  void deallocateBuffer();

  // modify
  bool justModify(const TensorImplDesc &desc);
  bool justModify(Buffer *buffer);

  // get
  bool empty();

  std::string getName();

  TensorImplDesc getDesc();
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
  void create(Device *device, const TensorImplDesc &desc, Buffer *buffer,
              void *data_ptr, int32_t data_id, const std::string &name,
              const base::IntVector &config);

 private:
  std::string name_;
  TensorImplDesc desc_;
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

extern NNDEPLOY_CC_API DefaultTensorImpl *createTensor(
    base::TensorImplType type);

/**
 * @brief
 *
 */
class NNDEPLOY_CC_API Tensor : public base::NonCopyable {
 public:
  Tensor(base::TensorImplType type = base::kTensorImplTypeDefault);
  virtual ~Tensor();

  Tensor(const TensorImplDesc &desc, const std::string &name = "",
         base::TensorImplType type = base::kTensorImplTypeDefault);

  Tensor(Device *device, const TensorImplDesc &desc,
         const std::string &name = "",
         const base::IntVector &config = base::IntVector(),
         base::TensorImplType type = base::kTensorImplTypeDefault);

  Tensor(Device *device, const TensorImplDesc &desc, void *data_ptr,
         const std::string &name = "",
         const base::IntVector &config = base::IntVector(),
         base::TensorImplType type = base::kTensorImplTypeDefault);
  Tensor(Device *device, const TensorImplDesc &desc, int32_t data_id,
         const std::string &name = "",
         const base::IntVector &config = base::IntVector(),
         base::TensorImplType type = base::kTensorImplTypeDefault);

  Tensor(const TensorImplDesc &desc, Buffer *buffer,
         const std::string &name = "",
         base::TensorImplType type = base::kTensorImplTypeDefault);

  // create
  // 必须确保为空
  void create(const TensorImplDesc &desc, const std::string &name = "");

  void create(Device *device, const TensorImplDesc &desc,
              const std::string &name = "",
              const base::IntVector &config = base::IntVector());

  void create(Device *device, const TensorImplDesc &desc, void *data_ptr,
              const std::string &name = "",
              const base::IntVector &config = base::IntVector());
  void create(Device *device, const TensorImplDesc &desc, int32_t data_id,
              const std::string &name = "",
              const base::IntVector &config = base::IntVector());

  void create(const TensorImplDesc &desc, Buffer *buffer,
              const std::string &name = "");

  void destory();

  void allocBuffer(Device *device,
                   const base::IntVector &config = base::IntVector());
  void deallocateBuffer();

  bool justModify(const TensorImplDesc &desc);

  bool justModify(Buffer *buffer);

  // get
  bool empty();

  std::string getName();
  base::TensorImplType getTensorImplType();

  TensorImplDesc getDesc();
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

using TensorMap = std::map<std::string, std::shared_ptr<Tensor>>;

}  // namespace device
}  // namespace nndeploy

#endif
