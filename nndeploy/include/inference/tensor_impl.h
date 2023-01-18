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
#ifndef _NNDEPLOY_INCLUDE_INFERENCE_TENSOR_IMPL_H_
#define _NNDEPLOY_INCLUDE_INFERENCE_TENSOR_IMPL_H_

#include "nndeploy/include/base/log.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/base/type.h"
#include "nndeploy/include/base/value.h"
#include "nndeploy/include/device/device.h"

namespace nndeploy {
namespace inference {

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

 private:
  std::string name_;
  TensorDesc desc;
  device::Buffer *buffer_;
};

}  // namespace inference
}  // namespace nndeploy

#endif
