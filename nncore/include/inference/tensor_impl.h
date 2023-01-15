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
#ifndef _NNCORE_INCLUDE_INFERENCE_TENSOR_IMPL_H_
#define _NNCORE_INCLUDE_INFERENCE_TENSOR_IMPL_H_

#include "nncore/include/base/log.h"
#include "nncore/include/base/macro.h"
#include "nncore/include/base/object.h"
#include "nncore/include/base/status.h"
#include "nncore/include/base/type.h"
#include "nncore/include/base/value.h"
#include "nncore/include/device/device.h"

namespace nncore {
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
}  // namespace nncore

#endif
