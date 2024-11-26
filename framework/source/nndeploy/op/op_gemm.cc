
#include "nndeploy/op/op_gemm.h"

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/time_profiler.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/util.h"


namespace nndeploy {
namespace op {

base::Status OpGemm::inferShape() {
  base::Status status = base::kStatusCodeOk;
  if (inputs_.size() < 2) {
    NNDEPLOY_LOGE("inputs_.size() < 2.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  auto param = dynamic_cast<ir::GemmParam *>(op_desc_.op_param_.get());
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param, "op_desc_.op_param_ is nullptr");

  bool trans_a = param->trans_a_ != 0;
  bool trans_b = param->trans_b_ != 0;
  auto first_input_shape = inputs_[0]->getShape();
  auto second_input_shape = inputs_[1]->getShape();
  if (first_input_shape.size() != 2) {
    NNDEPLOY_LOGE("First input does not have rank 2");
    return base::kStatusCodeErrorInvalidParam;
  }
  if (second_input_shape.size() != 2) {
    NNDEPLOY_LOGE("First input does not have rank 2");
    return base::kStatusCodeErrorInvalidParam;
  }

  base::IntVector output_shape;
  int32_t dim_0 = trans_a ? first_input_shape[1] : first_input_shape[0];
  int32_t dim_1 = trans_b ? second_input_shape[0] : second_input_shape[1];
  output_shape.emplace_back(dim_0);
  output_shape.emplace_back(dim_1);

  outputs_[0]->reshape(output_shape);

  return status;
}

base::Status OpGemm::run() {
  NNDEPLOY_LOGI("not implemented.\n");
  return base::kStatusCodeOk;
}

base::Status gemm(device::Tensor * inputs_a,
device::Tensor * inputs_b,
device::Tensor * inputs_c,
                                    std::shared_ptr<ir::GemmParam> param,
                                    device::Tensor *output) {
  base::Status status = base::kStatusCodeOk;

  Op *op = createOp(inputs_a->getDeviceType(), "", ir::kOpTypeMaxPool);
  if (op == nullptr) {
    NNDEPLOY_LOGE("createOp failed");
    return base::kStatusCodeErrorNotImplement;
  }
  status = op->setParam(param);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setParam failed");
  status = op->setInput(inputs_a, 0);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setInput(inputs_b, 0);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setInput(inputs_c, 0);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setOutput(output, 0);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setOutput failed");
  status = op->init();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "init failed");
  status = op->preRun();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "preRun failed");
  status = op->checkOrAllocOutput();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "checkOrAllocOutput failed");
  status = op->run();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "run failed");
  status = op->postRun();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "postRun failed");
  status = op->deinit();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "deinit failed");
  delete op;
  return status;
}

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeCpu,
                         ir::kOpTypeGemm, OpGemm)

}  // namespace op
}  // namespace nndeploy
