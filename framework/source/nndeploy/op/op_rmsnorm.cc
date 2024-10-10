#include "nndeploy/op/op_rmsnorm.h"

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

base::Status OpRMSNorm::inferShape() {
  auto input_shape = inputs_[0]->getShape();
  outputs_[0]->reshape(input_shape);
  return base::kStatusCodeOk;
}

<<<<<<< HEAD
base::Status rmsNorm(device::Tensor *input1, device::Tensor *input2, device::Tensor *input3,
                 device::Tensor *output) {
  base::Status status = base::kStatusCodeOk;
  if (input1 == nullptr || input2 == nullptr || input3 == nullptr || output == nullptr) {
=======
base::Status rmsNorm(device::Tensor *input1, device::Tensor *input2,
                     device::Tensor *input3, device::Tensor *output) {
  base::Status status = base::kStatusCodeOk;
  if (input1 == nullptr || input2 == nullptr || input3 == nullptr ||
      output == nullptr) {
>>>>>>> main
    NNDEPLOY_LOGE("input1 or input2 or output is nullptr");
    return base::kStatusCodeErrorNullParam;
  }
  Op *op = createOp(input1->getDeviceType(), "", ir::kOpTypeRMSNorm);
  if (op == nullptr) {
    NNDEPLOY_LOGE("createOp failed");
    return base::kStatusCodeErrorNotImplement;
  }
  // if (output->getDataType().code_ == base::kDataTypeCodeFp ||
  //     output->getDataType().code_ == base::kDataTypeCodeBFp) {
  //   base::PrecisionType precision_type =
  //       getPrecisionType(output->getDataType());
  //   status = op->setPrecisionType(precision_type);
  //   NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
  //                          "setPrecisionType failed");
  // }

  status = op->setInput(input1, 0);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setInput(input2, 1);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setInput(input3, 2);
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

}  // namespace op
}  // namespace nndeploy
