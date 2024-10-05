
#include "nndeploy/op/op_softmax.h"

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

base::Status OpSoftmax::inferShape() {
  base::Status status = base::kStatusCodeOk;

  // 参数
  auto param = dynamic_cast<ir::SoftmaxParam*>(op_desc_.op_param_.get());
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

  // infer output shape
  auto output_shape = inputs_[0]->getShape();
  outputs_[0]->reshape(output_shape);

  return status;
}

}  // namespace op
}  // namespace nndeploy
