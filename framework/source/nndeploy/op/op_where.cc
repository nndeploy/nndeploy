
#include "nndeploy/op/op_where.h"

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

namespace nndeploy {
namespace op {

base::Status OpWhere::run() {
  // TODO: 实现where
  NNDEPLOY_LOGI("not implemented.\n");
  return base::kStatusCodeOk;
}

base::Status OpWhere::inferShape() {
  auto input_shape = inputs_[0]->getShape();
  outputs_[0]->reshape(input_shape);
  return base::kStatusCodeOk;
}

base::Status where(device::Tensor* input1, device::Tensor* input2, device::Tensor* condition, 
                       device::Tensor* output) {
  base::Status status = base::kStatusCodeOk;

  Op* op = createOp(input1->getDeviceType(), "", ir::kOpTypeWhere);
  if (op == nullptr) {
    NNDEPLOY_LOGE("createOp failed");
    return base::kStatusCodeErrorNotImplement;
  }

  status = op->setInput(input1, 0);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setInput(input2, 1);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setInput(condition, 2);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setOutput(output, 0);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setOutput failed");
  status = op->init();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "init failed");
  status = op->checkOrAllocOutput();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                         "checkOrAllocOutput failed");
  status = op->preRun();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "preRun failed");
  status = op->run();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "run failed");
  status = op->postRun();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "postRun failed");
  status = op->deinit();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "deinit failed");
  delete op;

  return status;
}

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeCpu, ir::kOpTypeWhere, OpWhere)

}  // namespace op
}  // namespace nndeploy
