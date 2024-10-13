
#include "nndeploy/op/op_transpose.h"

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

base::Status OpTranspose::inferShape() {
  base::Status status = base::kStatusCodeOk;
  // 参数
  auto param = dynamic_cast<ir::TransposeParam *>(op_desc_.op_param_.get());
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param, "op_desc_.op_param_ is nullptr");
  std::vector<int> perm = param->perm_;

  // infer shape
  auto input_shape = inputs_[0]->getShape();
  auto output_shape = input_shape;
  if (perm.size() != 0) {
    if (perm.size() != input_shape.size()) {
      NNDEPLOY_LOGE("perm.size() != input_shape.size().\n");
      return base::kStatusCodeErrorInvalidParam;
    }
    for (size_t i = 0; i < perm.size(); i++) {
      output_shape[i] = input_shape[perm[i]];
    }
  } else {
    for (size_t i = 0; i < input_shape.size(); i++) {
      output_shape[i] = input_shape[input_shape.size() - 1 - i];
    }
  }
  outputs_[0]->reshape(output_shape);

  return status;
}

base::Status OpTranspose::run() {
  NNDEPLOY_LOGI("not implemented.\n");
  return base::kStatusCodeOk;
}

base::Status transpose(device::Tensor *input,
                       std::shared_ptr<ir::TransposeParam> param,
                       device::Tensor *output) {
  NNDEPLOY_LOGI("not implemented.\n");
  return base::kStatusCodeOk;
}

REGISTER_OP_IMPLEMENTION(base::DeviceTypeCode::kDeviceTypeCodeCpu,
                         ir::kOpTypeTranspose, OpTranspose)

}  // namespace op
}  // namespace nndeploy
