/**
 * @file config_impl.h
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-12-20
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#ifndef _NNDEPLOY_INCLUDE_INFERENCE_TENSOR_H_
#define _NNDEPLOY_INCLUDE_INFERENCE_TENSOR_H_


#include "nndeploy/include/base/log.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/base/type.h"
#include "nndeploy/include/base/value.h"
#include "nndeploy/include/device/device.h"
#include "nndeploy/include/inference/tensor_impl.h"

namespace nndeploy {
namespace inference {

class Tensor {
 public:
  Tensor(TensorDescImpl tensor_impl);

  virtual ~Tensor();

 private:
  std::shared_ptr<TensorImpl> tensor_impl_;
};

using TensorMap = std::map<std::string, std::shared_ptr<Tensor>>;

}  // namespace inference
}  // namespace nndeploy

#endif
