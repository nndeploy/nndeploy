
#include "nndeploy/op/op_concat.h"

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

base::Status OpConcat::inferShape() {
  base::Status status = base::kStatusCodeOk;
  // 参数
  auto param = dynamic_cast<ir::ConcatParam *>(op_desc_.op_param_.get());
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param, "op_desc_.op_param_ is nullptr");
  int axis = param->axis_;
  int rank = inputs_[0]->getShape().size();
  if (axis < -rank || axis >= rank) {
    NNDEPLOY_LOGE("axis is invalid.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  if (axis < 0) {
    axis += (int)inputs_[0]->getShape().size();
    param->axis_ = axis;
  }

  // check input shape
  for (size_t i = 1; i < inputs_.size(); i++) {
    if (inputs_[i]->getShape().size() != inputs_[0]->getShape().size()) {
      NNDEPLOY_LOGE("input shape is not equal.\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    for (size_t j = 0; j < inputs_[0]->getShape().size(); j++) {
      if (j == (size_t)axis) {
        continue;
      }
      if (inputs_[i]->getShape()[j] != inputs_[0]->getShape()[j]) {
        NNDEPLOY_LOGE("input shape is not equal.\n");
        return base::kStatusCodeErrorInvalidParam;
      }
    }
  }

  // infer output shape
  auto output_shape = inputs_[0]->getShape();
  for (size_t i = 1; i < inputs_.size(); i++) {
    output_shape[axis] += inputs_[i]->getShape()[axis];
  }
  outputs_[0]->reshape(output_shape);

  return status;
}

base::Status concat(std::vector<device::Tensor *> input,
                    std::shared_ptr<ir::ConcatParam> param,
                    device::Tensor *output) {
  base::Status status = base::kStatusCodeOk;
  return status;
}

}  // namespace op
}  // namespace nndeploy
