
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

base::Status OpBinary::inferShape() {
  base::Status status = base::kStatusCodeOk;
  auto input0_shape = inputs_[0]->getShape();
  auto input1_shape = inputs_[1]->getShape();

  // 广播的形状推理
  base::IntVector output_shape;
  int input0_size = input0_shape.size();
  int input1_size = input1_shape.size();

  if (input0_size == input1_size) {
    for (int i = 0; i < input0_size; i++) {
      if (input0_shape[i] != input1_shape[i] &&
          (input0_shape[i] == 1 || input1_shape[i] == 1)) {
        output_shape.push_back(std::max(input0_shape[i], input1_shape[i]));
      } else {
        output_shape.push_back(input0_shape[i]);
      }
    }
  } else {
    // 处理不同维度的情况
    int max_size = std::max(input0_size, input1_size);
    const base::IntVector &larger_shape =
        (input0_size > input1_size) ? input0_shape : input1_shape;
    const base::IntVector &smaller_shape =
        (input0_size > input1_size) ? input1_shape : input0_shape;

    output_shape.resize(max_size);

    // 从右向左填充较小的shape
    int diff = max_size - smaller_shape.size();
    for (int i = max_size - 1; i >= 0; i--) {
      if (i >= diff) {
        int smaller_idx = i - diff;
        if (larger_shape[i] != smaller_shape[smaller_idx] &&
            (larger_shape[i] == 1 || smaller_shape[smaller_idx] == 1)) {
          output_shape[i] =
              std::max(larger_shape[i], smaller_shape[smaller_idx]);
        } else if (larger_shape[i] == smaller_shape[smaller_idx]) {
          output_shape[i] = larger_shape[i];
        } else {
          NNDEPLOY_LOGE("无法进行广播,形状不兼容");
          return base::kStatusCodeErrorInvalidParam;
        }
      } else {
        output_shape[i] = larger_shape[i];
      }
    }
  }

  outputs_[0]->reshape(output_shape);
  return status;
}

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

  Op *op = createOp(output->getDeviceType(), "", kOpTypeSub);
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
  status = op->allocOutput();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "allocOutput failed");
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

base::Status mul(device::Tensor *input1, device::Tensor *input2,
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

  Op *op = createOp(output->getDeviceType(), "", kOpTypeMul);
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

base::Status div(device::Tensor *input1, device::Tensor *input2,
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

  Op *op = createOp(output->getDeviceType(), "", kOpTypeDiv);
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

}  // namespace op
}  // namespace nndeploy
