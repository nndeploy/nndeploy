
#include "nndeploy/op/op_muls.h"

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

base::Status OpMuls::run() {
  // 实现乘法运算
  device::Tensor* input1 = inputs_[0];
  device::Tensor* input2 = inputs_[1];
  device::Tensor* output = outputs_[0];

  // 获取输入和输出张量的形状
  base::IntVector input_shape = input2->getShape();

  // 获取输入和输出张量的数据指针
  float* input1_data = reinterpret_cast<float*>(input1->getData());
  float* input2_data = reinterpret_cast<float*>(input2->getData());
  float* output_data = reinterpret_cast<float*>(output->getData());

  // 计算总元素数量
  size_t total_elements = std::accumulate(
      input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());

  // 执行逐元素乘法运算
  for (size_t i = 0; i < total_elements; ++i) {
    output_data[i] = input1_data[0] * input2_data[i];
  }
  return base::kStatusCodeOk;
}

base::Status muls(device::Tensor* input1, device::Tensor* input2,
                  device::Tensor* output) {
  base::Status status = base::kStatusCodeOk;

  Op* op = createOp(input1->getDeviceType(), "", ir::kOpTypeMuls);
  if (op == nullptr) {
    NNDEPLOY_LOGE("createOp failed");
    return base::kStatusCodeErrorNotImplement;
  }
  status = op->setInput(input1, 0);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setInput(input2, 1);
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

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeCpu, ir::kOpTypeMuls, OpMuls)

}  // namespace op
}  // namespace nndeploy
