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
#include "nndeploy/op/op_rmsnorm.h"

namespace nndeploy {
namespace op {

base::Status OpRMSNorm::inferShape() {
  auto input_shape = inputs_[0]->getShape();
  outputs_[0]->reshape(input_shape);
  return base::kStatusCodeOk;
}

namespace rmsnorm {
void rmsnorm_forward_0(float* input, float* output, const float& eps_,
                       size_t iter_shape, size_t normalized_shape) {
  double buffer;
  for (int i = 0; i < iter_shape; ++i) {
    buffer = 0.;
    float* in = input + i * normalized_shape;
    float* out = output + i * normalized_shape;
    for (int j = 0; j < normalized_shape; ++j) {
      buffer += in[j] * in[j];
    }
    buffer = std::sqrt(buffer / normalized_shape);
    buffer += eps_;
    buffer = 1. / buffer;
    for (int j = 0; j < normalized_shape; ++j) {
      out[j] = in[j] * buffer;
    }
  }
}
void rmsnorm_forward_1(float* input, float* weight, float* output,
                       const float& eps_, size_t iter_shape,
                       size_t normalized_shape) {
  double buffer;
  for (int i = 0; i < iter_shape; ++i) {
    buffer = 0.;
    float* in = input + i * normalized_shape;
    float* out = output + i * normalized_shape;
    for (int j = 0; j < normalized_shape; ++j) {
      buffer += in[j] * in[j];
    }
    buffer = std::sqrt(buffer / normalized_shape);
    buffer += eps_;
    buffer = 1. / buffer;
    for (int j = 0; j < normalized_shape; ++j) {
      out[j] = in[j] * buffer * weight[j];
    }
  }
}
void rmsnorm_forward_2(float* input, float* weight, float* residual,
                       float* output, const float& eps_, size_t iter_shape,
                       size_t normalized_shape) {
  double buffer;
  for (int i = 0; i < iter_shape; ++i) {
    buffer = 0.;
    float* in = input + i * normalized_shape;
    float* out = output + i * normalized_shape;
    for (int j = 0; j < normalized_shape; ++j) {
      buffer += in[j] * in[j];
    }
    buffer = std::sqrt(buffer / normalized_shape);
    buffer += eps_;
    buffer = 1. / buffer;
    for (int j = 0; j < normalized_shape; ++j) {
      out[j] = in[j] * buffer * weight[j] + residual[j];
    }
  }
}
}  // namespace rmsnorm

template <>
base::Status OpRMSNorm::run<3>() {
  device::Tensor* input = inputs_[0];
  device::Tensor* weight = inputs_[1];
  device::Tensor* residual = inputs_[2];
  device::Tensor* output = outputs_[0];

  // 获取输入张量的形状
  base::IntVector input_shape = input->getShape();
  base::IntVector weight_shape = weight->getShape();
  base::IntVector residual_shape = residual->getShape();

  // 获取RMSNorm参数
  const auto& param = dynamic_cast<ir::RMSNormParam*>(op_desc_.op_param_.get());
  const float& eps_ = param->eps_;
  const base::IntVector& normalized_shape_ = param->normalized_shape_;

  // 检查输入张量的形状是否相同
  int iter_shape_begin = input_shape.size() - normalized_shape_.size();
  for (int i = 0; i < normalized_shape_.size(); ++i) {
    // norm shape 检查
    if (input_shape[iter_shape_begin + i] != normalized_shape_[i] ||
        input_shape[iter_shape_begin + i] != weight_shape[i] ||
        (residual != nullptr &&
         input_shape[iter_shape_begin + i] != residual_shape[i])) {
      NNDEPLOY_LOGE(
          "Input tensors do not have the same number of dimensions.\n");
      return base::kStatusCodeErrorInvalidParam;
    }
  }

  size_t iter_shape = std::accumulate(input_shape.begin(),
                                      input_shape.begin() + iter_shape_begin, 1,
                                      std::multiplies<int>());
  size_t normalized_shape =
      std::accumulate(normalized_shape_.begin(), normalized_shape_.end(), 1,
                      std::multiplies<int>());

  // 获取输入张量的数据
  float* input_data = (float*)input->getData();
  float* weight_data = (float*)weight->getData();
  float* residual_data = (float*)residual->getData();
  float* output_data = (float*)output->getData();
  // 计算RMSNorm
  rmsnorm::rmsnorm_forward_2(input_data, weight_data, residual_data,
                             output_data, eps_, iter_shape, normalized_shape);
}

template <>
base::Status OpRMSNorm::run<2>() {
  device::Tensor* input = inputs_[0];
  device::Tensor* weight = inputs_[1];
  device::Tensor* output = outputs_[0];

  // 获取输入张量的形状
  base::IntVector input_shape = input->getShape();
  base::IntVector weight_shape = weight->getShape();

  // 获取RMSNorm参数
  const auto& param = dynamic_cast<ir::RMSNormParam*>(op_desc_.op_param_.get());
  const float& eps_ = param->eps_;
  const base::IntVector& normalized_shape_ = param->normalized_shape_;

  // 检查输入张量的形状是否相同
  int iter_shape_begin = input_shape.size() - normalized_shape_.size();
  for (int i = 0; i < normalized_shape_.size(); ++i) {
    // norm shape 检查
    if (input_shape[iter_shape_begin + i] != normalized_shape_[i] ||
        input_shape[iter_shape_begin + i] != weight_shape[i]) {
      NNDEPLOY_LOGE(
          "Input tensors do not have the same number of dimensions.\n");
      return base::kStatusCodeErrorInvalidParam;
    }
  }

  size_t iter_shape = std::accumulate(input_shape.begin(),
                                      input_shape.begin() + iter_shape_begin, 1,
                                      std::multiplies<int>());
  size_t normalized_shape =
      std::accumulate(normalized_shape_.begin(), normalized_shape_.end(), 1,
                      std::multiplies<int>());

  // 获取输入张量的数据
  float* input_data = (float*)input->getData();
  float* weight_data = (float*)weight->getData();
  float* output_data = (float*)output->getData();
  // 计算RMSNorm
  rmsnorm::rmsnorm_forward_1(input_data, weight_data, output_data, eps_,
                             iter_shape, normalized_shape);
  return base::kStatusCodeOk;
}

template <>
base::Status OpRMSNorm::run<1>() {
  device::Tensor* input = inputs_[0];
  device::Tensor* output = outputs_[0];

  // 获取输入张量的形状
  base::IntVector input_shape = input->getShape();

  // 获取RMSNorm参数
  const auto& param = dynamic_cast<ir::RMSNormParam*>(op_desc_.op_param_.get());
  const float& eps_ = param->eps_;
  const base::IntVector& normalized_shape_ = param->normalized_shape_;

  // 检查输入张量的形状是否相同
  int iter_shape_begin = input_shape.size() - normalized_shape_.size();
  for (int i = 0; i < normalized_shape_.size(); ++i) {
    // norm shape 检查
    if (input_shape[iter_shape_begin + i] != normalized_shape_[i]) {
      NNDEPLOY_LOGE(
          "Input tensors do not have the same number of dimensions.\n");
      return base::kStatusCodeErrorInvalidParam;
    }
  }

  size_t iter_shape = std::accumulate(input_shape.begin(),
                                      input_shape.begin() + iter_shape_begin, 1,
                                      std::multiplies<int>());
  size_t normalized_shape =
      std::accumulate(normalized_shape_.begin(), normalized_shape_.end(), 1,
                      std::multiplies<int>());

  // 获取输入张量的数据
  float* input_data = (float*)input->getData();
  float* output_data = (float*)output->getData();
  // 计算RMSNorm
  rmsnorm::rmsnorm_forward_0(input_data, output_data, eps_, iter_shape,
                             normalized_shape);
  return base::kStatusCodeOk;
}

base::Status OpRMSNorm::run() {
  int input_size = inputs_.size();
  if (inputs_.size() == 1) {
    return run<1>();
  } else if (inputs_.size() == 2) {
    return run<2>();
  } else {
    return run<3>();
  }
}

#if 1
base::Status rmsNorm(device::Tensor* input, std::shared_ptr<base::Param> param,
                     device::Tensor* output) {
  base::Status status = base::kStatusCodeOk;

  Op* op = createOp(input->getDeviceType(), "", ir::kOpTypeRMSNorm);
  if (op == nullptr) {
    NNDEPLOY_LOGE("createOp failed");
    return base::kStatusCodeErrorNotImplement;
  }
  status = op->setParam(param);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setParam failed");
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

base::Status rmsNorm(device::Tensor* input, device::Tensor* weight,
                     std::shared_ptr<base::Param> param,
                     device::Tensor* output) {
  base::Status status = base::kStatusCodeOk;

  Op* op = createOp(input->getDeviceType(), "", ir::kOpTypeRMSNorm);
  if (op == nullptr) {
    NNDEPLOY_LOGE("createOp failed");
    return base::kStatusCodeErrorNotImplement;
  }
  status = op->setParam(param);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setParam failed");
  status = op->setInput(input, 0);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setInput(weight, 1);
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

base::Status rmsNorm(device::Tensor* input, device::Tensor* weight,
                     device::Tensor* residual,
                     std::shared_ptr<base::Param> param,
                     device::Tensor* output) {
  base::Status status = base::kStatusCodeOk;

  Op* op = createOp(input->getDeviceType(), "", ir::kOpTypeRMSNorm);
  if (op == nullptr) {
    NNDEPLOY_LOGE("createOp failed");
    return base::kStatusCodeErrorNotImplement;
  }
  status = op->setParam(param);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setParam failed");
  status = op->setInput(input, 0);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setInput(weight, 1);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setInput(residual, 2);
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
#endif


REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeCpu, ir::kOpTypeRMSNorm, OpRMSNorm)

}  // namespace op
}  // namespace nndeploy
