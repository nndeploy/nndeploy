
#include "nndeploy/op/op_add.h"

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

base::Status OpAdd::run() {
  device::Tensor* input1 = inputs_[0];
  device::Tensor* input2 = inputs_[1];
  device::Tensor* output = outputs_[0];

  // 获取输入张量的形状
  base::IntVector input1_shape = input1->getShape();
  base::IntVector input2_shape = input2->getShape();

  // TODO: 暂时只支持完全相等的shape
  if (input1_shape.size() != input2_shape.size()) {
    // 维度数不同，直接返回错误
    NNDEPLOY_LOGE("Input tensors do not have the same number of dimensions.\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  for (size_t i = 0; i < input1_shape.size(); ++i) {
    if (input1_shape[i] != input2_shape[i]) {
      // 对应维度大小不同，返回错误
      NNDEPLOY_LOGE("Input tensors do not have the same shape.\n");
      return base::kStatusCodeErrorInvalidParam;
    }
  }

  // 获取输入和输出张量的数据指针
  float* input1_data = reinterpret_cast<float*>(input1->getData());
  float* input2_data = reinterpret_cast<float*>(input2->getData());
  float* output_data = reinterpret_cast<float*>(output->getData());

  // 获取输入张量的总元素数量
  size_t total_elements = std::accumulate(
      input1_shape.begin(), input1_shape.end(), 1, std::multiplies<size_t>());

  // 执行逐元素加法运算
  for (size_t i = 0; i < total_elements; ++i) {
    output_data[i] = input1_data[i] + input2_data[i];
  }
  return base::kStatusCodeOk;
}

base::Status add(device::Tensor* input1, device::Tensor* input2,
                 device::Tensor* output) {
  base::Status status = base::kStatusCodeOk;

  Op* op = createOp(input1->getDeviceType(), "", ir::kOpTypeAdd);
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

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeCpu, ir::kOpTypeAdd, OpAdd)

}  // namespace op
}  // namespace nndeploy
