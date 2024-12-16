
#include "nndeploy/op/op_gemm.h"

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

base::Status OpGemm::inferShape() {
  base::Status status = base::kStatusCodeOk;
  if (inputs_.size() < 2) {
    NNDEPLOY_LOGE("inputs_.size() < 2.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  auto param = dynamic_cast<ir::GemmParam*>(op_desc_.op_param_.get());
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param, "op_desc_.op_param_ is nullptr");

  bool trans_a = param->trans_a_ != 0;
  bool trans_b = param->trans_b_ != 0;
  auto first_input_shape = inputs_[0]->getShape();
  auto second_input_shape = inputs_[1]->getShape();
  if (first_input_shape.size() != 2) {
    NNDEPLOY_LOGE("First input does not have rank 2");
    return base::kStatusCodeErrorInvalidParam;
  }
  if (second_input_shape.size() != 2) {
    NNDEPLOY_LOGE("First input does not have rank 2");
    return base::kStatusCodeErrorInvalidParam;
  }

  base::IntVector output_shape;
  int32_t dim_0 = trans_a ? first_input_shape[1] : first_input_shape[0];
  int32_t dim_1 = trans_b ? second_input_shape[0] : second_input_shape[1];
  output_shape.emplace_back(dim_0);
  output_shape.emplace_back(dim_1);

  outputs_[0]->reshape(output_shape);

  // bias shape兼容检查
  // output_shape的维度为[batch, channel]
  // 1. 完全与output shape大小相等
  // 2. 为[1, channel]
  // 3. 为 [channel]
  if (inputs_.size() > 2) {
    auto bias_shape = inputs_[2]->getShape();
    auto output_shape = outputs_[0]->getShape();

    // 检查bias是否与输出形状兼容
    bool is_compatible = false;
    if (bias_shape.size() == 1) {
      // 规则3: bias形状为[channel]
      is_compatible = bias_shape[0] == output_shape[1];
    } else if (bias_shape.size() == 2) {
      // 规则1: bias形状完全与输出形状相等
      // 规则2: bias形状为[1, channel]
      is_compatible = (bias_shape[0] == output_shape[0] &&
                       bias_shape[1] == output_shape[1]) ||
                      (bias_shape[0] == 1 && bias_shape[1] == output_shape[1]);
    }

    if (!is_compatible) {
      NNDEPLOY_LOGE(
          "Bias shape is not compatible with output shape for broadcasting");
      return base::kStatusCodeErrorInvalidParam;
    }
  }
  return status;
}

base::Status OpGemm::run() {
  // 获取输入和输出张量
  device::Tensor* input_a = inputs_[0];
  device::Tensor* input_b = inputs_[1];
  device::Tensor* input_c =
      inputs_.size() > 2 ? inputs_[2] : nullptr;  // 可选的偏置矩阵
  device::Tensor* output = outputs_[0];

  // 获取输入张量的形状
  base::IntVector shape_a = input_a->getShape();
  base::IntVector shape_b = input_b->getShape();
  base::IntVector shape_c = input_c ? input_c->getShape() : base::IntVector();

  // 获取参数
  auto param = dynamic_cast<ir::GemmParam*>(op_desc_.op_param_.get());
  if (!param) {
    NNDEPLOY_LOGE("Failed to cast op param to GemmParam\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  // 确定矩阵乘法的维度
  size_t M = param->trans_a_ ? shape_a[1] : shape_a[0];
  size_t N = param->trans_b_ ? shape_b[0] : shape_b[1];
  size_t K = param->trans_a_ ? shape_a[0] : shape_a[1];

  // 确保输入张量的形状与参数一致
  if ((param->trans_b_ ? shape_b[1] : shape_b[0]) != K) {
    NNDEPLOY_LOGE(
        "Input shapes are not compatible for matrix multiplication.\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  // 获取输入和输出张量的数据指针
  float* data_a = reinterpret_cast<float*>(input_a->getData());
  float* data_b = reinterpret_cast<float*>(input_b->getData());
  float* data_c =
      input_c ? reinterpret_cast<float*>(input_c->getData()) : nullptr;
  float* data_output = reinterpret_cast<float*>(output->getData());

  // 执行矩阵乘法
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (size_t k = 0; k < K; ++k) {
        size_t a_index = param->trans_a_ ? (k * M + m) : (m * K + k);
        size_t b_index = param->trans_b_ ? (n * K + k) : (k * N + n);
        sum += data_a[a_index] * data_b[b_index];
      }
      // 应用 bias，如果存在，并且支持广播
      if (data_c) {
        // 根据 bias 的形状进行索引计算
        if (shape_c.size() == 1) {
          // 规则3: bias形状为[channel]
          sum += param->beta_ * data_c[n];
        } else if (shape_c.size() == 2) {
          // 规则2: bias形状为[1, channel]
          if (shape_c[0] == 1) {
            sum += param->beta_ * data_c[n];
          } else {
            // 规则3： bias形状与output一致
            sum += param->beta_ * data_c[m * N + n];
          }
        }
      }
      data_output[m * N + n] = param->alpha_ * sum;
    }
  }

  return base::kStatusCodeOk;
}

base::Status gemm(device::Tensor* inputs_a, device::Tensor* inputs_b,
                  device::Tensor* inputs_c,
                  std::shared_ptr<ir::GemmParam> param,
                  device::Tensor* output) {
  base::Status status = base::kStatusCodeOk;

  Op* op = createOp(inputs_a->getDeviceType(), "", ir::kOpTypeGemm);
  if (op == nullptr) {
    NNDEPLOY_LOGE("createOp failed");
    return base::kStatusCodeErrorNotImplement;
  }
  status = op->setParam(param);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setParam failed");
  status = op->setInput(inputs_a, 0);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setInput(inputs_b, 1);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  if (inputs_c != nullptr) {
    status = op->setInput(inputs_c, 2);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  }
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

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeCpu, ir::kOpTypeGemm, OpGemm)

}  // namespace op
}  // namespace nndeploy
