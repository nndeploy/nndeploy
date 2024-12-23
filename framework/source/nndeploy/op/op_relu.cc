
#include "nndeploy/op/op_relu.h"

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

base::Status OpRelu::run() {
  base::Status status = base::kStatusCodeOk;

  // 获取输入和输出张量
  device::Tensor* input_tensor = inputs_[0];
  device::Tensor* output_tensor = outputs_[0];

  // 获取输入张量的维度信息
  auto input_shape = input_tensor->getShape();
  long input_elements = std::accumulate(input_shape.begin(), input_shape.end(),
                                        1, std::multiplies<int>());

  // 获取输入和输出张量的数据指针
  float* input_data = static_cast<float*>(input_tensor->getData());
  float* output_data = static_cast<float*>(output_tensor->getData());

  // 执行ReLU操作
  for (long i = 0; i < input_elements; ++i) {
    if (input_data[i] < 0.0f) {
      output_data[i] = 0.0f;
    } else {
      output_data[i] = input_data[i];
    }
  }

  return status;
}

base::Status relu(device::Tensor* input, device::Tensor* output) {
  base::Status status = base::kStatusCodeOk;

  Op* op = createOp(input->getDeviceType(), "", ir::kOpTypeRelu);
  if (op == nullptr) {
    NNDEPLOY_LOGE("createOp failed");
    return base::kStatusCodeErrorNotImplement;
  }
  status = op->setInput(input, 0);
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

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeCpu, ir::kOpTypeRelu, OpRelu)

}  // namespace op
}  // namespace nndeploy
