
#ifndef _NNDEPLOY_INCLUDE_DEVICE_TENSOR_H_
#define _NNDEPLOY_INCLUDE_DEVICE_TENSOR_H_

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/log.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/base/value.h"
#include "nndeploy/include/device/device.h"
#include "nndeploy/include/device/tensor_impl.h"


namespace nndeploy {
namespace device {

/**
 * @brief 需要扩张对量化的tensor支持
 * 
 */
class Tensor {
 public:
  Tensor(TensorDesc desc);

  virtual ~Tensor();

 private:
  std::shared_ptr<TensorImpl> tensor_impl_;
};

class TensorPtrArray {
 public:
  TensorArray(std::vector<Tensor *> tensors_);
  TensorArray(Tensor* tensor);

  virtual ~TensorArray();

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
