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
#ifndef _NNCORE_INCLUDE_INFERENCE_TENSOR_H_
#define _NNCORE_INCLUDE_INFERENCE_TENSOR_H_


#include "nncore/include/base/log.h"
#include "nncore/include/base/macro.h"
#include "nncore/include/base/object.h"
#include "nncore/include/base/status.h"
#include "nncore/include/base/type.h"
#include "nncore/include/base/value.h"
#include "nncore/include/device/device.h"
#include "nncore/include/inference/tensor_impl.h"

namespace nncore {
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
}  // namespace nncore

#endif
