
#include "nndeploy/op/op_mul.h"

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

base::Status OpMul::run() {
  // 实现支持广播的乘法运算
  device::Tensor* input1 = inputs_[0];
  device::Tensor* input2 = inputs_[1];
  device::Tensor* output = outputs_[0];

  // 获取输入张量的形状
  base::IntVector input1_shape = input1->getShape();
  base::IntVector input2_shape = input2->getShape();
  base::IntVector output_shape = output->getShape();

  // 获取输入和输出张量的数据指针
  float* input1_data = reinterpret_cast<float*>(input1->getData());
  float* input2_data = reinterpret_cast<float*>(input2->getData());
  float* output_data = reinterpret_cast<float*>(output->getData());

  // 计算输出张量的总元素数量
  size_t total_elements = std::accumulate(
      output_shape.begin(), output_shape.end(), 1, std::multiplies<size_t>());

  // 计算各维度的步长
  std::vector<size_t> input1_strides(input1_shape.size());
  std::vector<size_t> input2_strides(input2_shape.size());
  std::vector<size_t> output_strides(output_shape.size());

  // 计算输入1的步长
  if (!input1_shape.empty()) {
    input1_strides[input1_shape.size() - 1] = 1;
    for (int i = input1_shape.size() - 2; i >= 0; --i) {
      input1_strides[i] = input1_strides[i + 1] * input1_shape[i + 1];
    }
  }

  // 计算输入2的步长
  if (!input2_shape.empty()) {
    input2_strides[input2_shape.size() - 1] = 1;
    for (int i = input2_shape.size() - 2; i >= 0; --i) {
      input2_strides[i] = input2_strides[i + 1] * input2_shape[i + 1];
    }
  }

  // 计算输出的步长
  if (!output_shape.empty()) {
    output_strides[output_shape.size() - 1] = 1;
    for (int i = output_shape.size() - 2; i >= 0; --i) {
      output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
    }
  }

  // 执行支持广播的逐元素乘法运算
  for (size_t linear_idx = 0; linear_idx < total_elements; ++linear_idx) {
    // 将线性索引转换为多维索引
    std::vector<size_t> multi_idx(output_shape.size());
    size_t temp_idx = linear_idx;
    for (int i = output_shape.size() - 1; i >= 0; --i) {
      multi_idx[i] = temp_idx % output_shape[i];
      temp_idx /= output_shape[i];
    }

    // 计算输入1的索引（支持广播）
    size_t input1_idx = 0;
    int input1_dim_offset = output_shape.size() - input1_shape.size();
    for (size_t i = 0; i < input1_shape.size(); ++i) {
      size_t output_dim_idx = i + input1_dim_offset;
      size_t coord = (input1_shape[i] == 1) ? 0 : multi_idx[output_dim_idx];
      input1_idx += coord * input1_strides[i];
    }

    // 计算输入2的索引（支持广播）
    size_t input2_idx = 0;
    int input2_dim_offset = output_shape.size() - input2_shape.size();
    for (size_t i = 0; i < input2_shape.size(); ++i) {
      size_t output_dim_idx = i + input2_dim_offset;
      size_t coord = (input2_shape[i] == 1) ? 0 : multi_idx[output_dim_idx];
      input2_idx += coord * input2_strides[i];
    }

    // 执行乘法运算
    output_data[linear_idx] = input1_data[input1_idx] * input2_data[input2_idx];
  }

  return base::kStatusCodeOk;
}

base::Status mul(device::Tensor* input1, device::Tensor* input2,
                 device::Tensor* output) {
  base::Status status = base::kStatusCodeOk;

  Op* op = createOp(input1->getDeviceType(), "", ir::kOpTypeMul);
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

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeCpu, ir::kOpTypeMul, OpMul)
REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeArm, ir::kOpTypeMul, OpMul)
REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeX86, ir::kOpTypeMul, OpMul)

}  // namespace op
}  // namespace nndeploy
