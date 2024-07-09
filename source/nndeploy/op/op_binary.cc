
#include "nndeploy/op/op_binary.h"

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
#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

base::Status add(device::Tensor *input1, device::Tensor *input2,
                 device::Tensor *output) {
  base::Status status = base::kStatusCodeOk;
  if (input1 == nullptr || input2 == nullptr || output == nullptr) {
    NNDEPLOY_LOGE("input1 or input2 or output is nullptr");
    return base::kStatusCodeErrorNullParam;
  }
  if (input1->getDataType() != input2->getDataType() ||
      input1->getDataType() != output->getDataType()) {
    NNDEPLOY_LOGE("input1 or input2 or output data type is not equal");
    return base::kStatusCodeErrorInvalidParam;
  }
  if (input1->getDeviceType() != input2->getDeviceType() ||
      input1->getDeviceType() != output->getDeviceType()) {
    NNDEPLOY_LOGE("input1 or input2 or output device type is not equal");
    return base::kStatusCodeErrorInvalidParam;
  }
  if (input1->getShape() != output->getShape() &&
      input2->getShape() != output->getShape()) {
    NNDEPLOY_LOGE("input1 or input2 or output shape is not equal");
    return base::kStatusCodeErrorInvalidParam;
  }

  Op *op = createOp(output->getDeviceType(), "", kOpTypeAdd);
  if (op == nullptr) {
    NNDEPLOY_LOGE("createOp failed");
    return base::kStatusCodeErrorNotImplement;
  }
  if (output->getDataType().code_ == base::kDataTypeCodeFp ||
      output->getDataType().code_ == base::kDataTypeCodeBFp) {
    base::PrecisionType precision_type =
        getPrecisionType(output->getDataType());
    status = op->setPrecisionType(precision_type);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk,
                           "setPrecisionType failed");
  }
  status = op->setInput(input1, 0);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setInput(input2, 1);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setOutput(output, 0);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setOutput failed");
  status = op->init();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "init failed");
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

base::Status sub(device::Tensor *input1, device::Tensor *input2,
                 device::Tensor *output) {
  return base::Status();
}

base::Status mul(device::Tensor *input1, device::Tensor *input2,
                 device::Tensor *output) {
  return base::Status();
}

base::Status div(device::Tensor *input1, device::Tensor *input2,
                 device::Tensor *output) {
  return base::Status();
}

base::Status clamp(device::Tensor *input, float min, float max,
                   device::Tensor *output) {
  return base::Status();
}

}  // namespace op
}  // namespace nndeploy
