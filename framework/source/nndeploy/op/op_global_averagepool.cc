
#include "nndeploy/op/op_global_averagepool.h"

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

base::Status OpGlobalAveragepool::inferShape() {
  base::Status status = base::kStatusCodeOk;

  base::IntVector input_shape = inputs_[0]->getShape();
  if (input_shape.size() < 2) {
    NNDEPLOY_LOGE("input_shape.size() < 2.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.size() - 2);
  base::IntVector output_shape;
  output_shape.emplace_back(input_shape[0]);
  output_shape.emplace_back(input_shape[1]);
  for (size_t i = 0; i < n_input_dims; ++i) {
    output_shape.emplace_back(1);
  }

  outputs_[0]->reshape(output_shape);

  return status;
}

base::Status OpGlobalAveragepool::run() {
  // 获取输入和输出张量
  device::Tensor* input_tensor = inputs_[0];
  device::Tensor* output_tensor = outputs_[0];

  // 获取输入张量的形状
  base::IntVector input_shape = input_tensor->getShape();

  // 获取输入和输出张量的数据指针
  float* input_data = reinterpret_cast<float*>(input_tensor->getData());
  float* output_data = reinterpret_cast<float*>(output_tensor->getData());

  // 计算输入张量的总元素数量（不包括 batch 和 channels）
  size_t total_elements = std::accumulate(
      input_shape.begin() + 2, input_shape.end(), 1, std::multiplies<size_t>());

  // 计算全局平均池化
  for (size_t b = 0; b < input_shape[0]; ++b) {    // batch dimension
    for (size_t c = 0; c < input_shape[1]; ++c) {  // channel dimension
      float sum = 0.0f;
      // size_t total_elements = 1;
      // for (size_t i = 2; i < input_shape.size(); ++i) {
      //   total_elements *= input_shape[i];
      // }
      size_t input_index =
          b * input_shape[1] * total_elements + c * total_elements;
      for (size_t e = 0; e < total_elements; ++e) {
        sum += input_data[input_index++];
      }
      output_data[b * input_shape[1] + c] =
          sum / static_cast<float>(total_elements);
    }
  }

  return base::kStatusCodeOk;
}

base::Status globalAveragepool(device::Tensor* input, device::Tensor* output) {
  base::Status status = base::kStatusCodeOk;

  Op* op = createOp(input->getDeviceType(), "", ir::kOpTypeGlobalAveragePool);
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

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeCpu, ir::kOpTypeGlobalAveragePool,
                         OpGlobalAveragepool)

}  // namespace op
}  // namespace nndeploy
