
#include "nndeploy/op/op_mat_mul.h"

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

base::Status OpMatMul::inferShape() {
  base::Status status = base::kStatusCodeOk;
  if (inputs_.size() < 2) {
    NNDEPLOY_LOGE("inputs_.size() < 2.\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  auto first_input_shape = inputs_[0]->getShape();
  auto second_input_shape = inputs_[1]->getShape();
  int shape_size = second_input_shape.size();

  base::IntVector output_shape = first_input_shape;
  output_shape[shape_size - 1] = second_input_shape[shape_size - 1];

  outputs_[0]->reshape(output_shape);

  return status;
}

base::Status OpMatMul::run() {
  // NNDEPLOY_LOGI("not implemented.\n");
  device::Tensor *input_a = inputs_[0];
  device::Tensor *input_b = inputs_[1];
  device::Tensor *output = outputs_[0];
  auto input_a_shape = input_a->getShape();
  auto input_b_shape = input_b->getShape();
  auto output_shape = output->getShape();
  int m = input_a_shape[0];
  int n = input_b_shape[1];
  int k = input_a_shape[1];
  float *input_a_data = static_cast<float *>(input_a->getData());
  float *input_b_data = static_cast<float *>(input_b->getData());
  float *output_data = static_cast<float *>(output->getData());
  memset(output_data, 0, m * n * sizeof(float));
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int l = 0; l < k; l++) {
        output_data[i * n + j] +=
            input_a_data[i * k + l] * input_b_data[l * n + j];
      }
    }
  }
  return base::kStatusCodeOk;
}

base::Status matmul(device::Tensor *inputs_a, device::Tensor *inputs_b,
                    device::Tensor *output) {
  base::Status status = base::kStatusCodeOk;

  Op *op = createOp(inputs_a->getDeviceType(), "", ir::kOpTypeMatMul);
  if (op == nullptr) {
    NNDEPLOY_LOGE("createOp failed");
    return base::kStatusCodeErrorNotImplement;
  }
  status = op->setInput(inputs_a, 0);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setInput(inputs_b, 0);
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

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeCpu, ir::kOpTypeMatMul, OpMatMul)

}  // namespace op
}  // namespace nndeploy
