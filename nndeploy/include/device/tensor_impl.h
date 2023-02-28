
#ifndef _NNDEPLOY_INCLUDE_DEVICE_TENSOR_IMPL_H_
#define _NNDEPLOY_INCLUDE_DEVICE_TENSOR_IMPL_H_

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/log.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/base/value.h"
#include "nndeploy/include/device/device.h"


namespace nndeploy {
namespace device {

struct TensorDesc {
  base::DataType data_type_;
  base::DataFormat format;
  base::IntVector shape_;
  base::SizeVector stride_;
};

class TensorImpl {
 public:
  TensorImpl();
  virtual ~TensorImpl();

  TensorImpl(Device *device, TensorDesc desc, IntVector config,
             const std::strinng &name = "");
  TensorImpl(TensorDesc desc, Buffer *buffer, const std::strinng &name = "");

  //
  TensorImpl(const TensorImpl &tensor);
  TensorImpl(const TensorImpl &&tensor);

  //
  TensorImpl &operator=(const TensorImpl &tensor);
  rTensorImpl &operator==(const TensorImpl &&mat);

  // create
  void create(Device *device, TensorDesc desc, IntVector config);
  void create(TensorDesc desc, Buffer *buffer);

  // get
  bool empty();
  bool isContinue();

  std::string getName();
  base::DEVICEType getDEVICEType();

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
  MemoryPool *getMemoryPool();
  bool isMemoryPool();
  bool isExternal();
  base::MemoryBufferType getMemoryBufferType();
  size_t getSize();
  base::SizeVector getSizeVector();
  base::IntVector getConfig();
  void *getPtr();
  int32_t getId();

 private:
  std::string name_;
  TensorDesc desc_;
  device::Buffer *buffer_;
};

}  // namespace device
}  // namespace nndeploy

#endif
