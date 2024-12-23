
#include "nndeploy/op/op_maxpool.h"

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

base::Status OpMaxPool::inferShape() {
  base::Status status = base::kStatusCodeOk;

  base::IntVector input_shape = inputs_[0]->getShape();
  if (input_shape.size() < 2) {
    NNDEPLOY_LOGE("input_shape.size() < 2.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.size() - 2);
  // 参数
  auto param = dynamic_cast<ir::MaxPoolParam*>(op_desc_.op_param_.get());
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param, "op_desc_.op_param_ is nullptr");
  std::vector<int> dilations = param->dilations_;
  if (dilations.size() == 0) {
    dilations.resize(n_input_dims, 1);
    param->dilations_ = dilations;
  }
  std::vector<int> kernel_shape = param->kernel_shape_;
  std::vector<int> strides = param->strides_;
  if (strides.size() == 0) {
    strides.resize(n_input_dims, 1);
    param->strides_ = strides;
  }

  std::vector<int> effective_kernel_shape = kernel_shape;
  for (int i = 0; i < static_cast<int>(kernel_shape.size()); i++) {
    // accounting for dilation, how big is the kernel in this dimension
    effective_kernel_shape[i] =
        (effective_kernel_shape[i] - 1) * dilations[i] + 1;
  }

  std::vector<int> pads = param->pads_;
  if (pads.size() == 0) {
    pads.resize(kernel_shape.size() * 2, 0);
    auto auto_pad_attr = param->auto_pad_;
    if ((!auto_pad_attr.empty()) && (auto_pad_attr != "VALID")) {
      int input_dims_size = static_cast<int>(n_input_dims);
      for (int i = 0; i < input_dims_size; ++i) {
        int64_t residual = 0;
        int64_t stride = strides[i];
        if (stride > 1) {
          residual = input_shape[2 + i];
          while (residual >= stride) {
            residual -= stride;
          }
        }
        int64_t total_pad = residual == 0
                                ? effective_kernel_shape[i] - stride
                                : effective_kernel_shape[i] - residual;
        if (total_pad < 0) total_pad = 0;
        int64_t half_pad_small = total_pad >> 1;
        int64_t half_pad_big = total_pad - half_pad_small;
        if (auto_pad_attr == "SAME_UPPER") {
          pads[i] = half_pad_small;
          pads[i + input_dims_size] = half_pad_big;
        } else if (auto_pad_attr == "SAME_LOWER") {
          pads[i] = half_pad_big;
          pads[i + input_dims_size] = half_pad_small;
        }
      }
    }
    param->pads_ = pads;
  }

  // add the first two dimensions from the input.
  base::IntVector output_shape = input_shape;
  for (size_t i = 2; i < output_shape.size(); i++) {
    output_shape[i] = -1;
  }
  int new_i = 1;

  int kernel_shape_size = static_cast<int>(kernel_shape.size());
  for (int i = 0; i < kernel_shape_size; ++i) {
    new_i++;

    // how big is the input, including padding
    int effective_input_size = input_shape[2 + i];
    effective_input_size += pads[i];
    effective_input_size += pads[i + kernel_shape_size];

    // default is floor mode .i.e. ceil_mode is set to 0
    auto ceil_mode = param->ceil_mode_;

    // how many times we can move the kernel from it's initial position, based
    // on the stride
    int64_t strided_kernel_positions;

    if (ceil_mode == 1)
      strided_kernel_positions = (int64_t)(std::ceil(
          (effective_input_size - effective_kernel_shape[i]) /
          float(strides[i])));
    else
      strided_kernel_positions =
          (effective_input_size - effective_kernel_shape[i]) / strides[i];

    // add in the initial position
    output_shape[new_i] = 1 + strided_kernel_positions;
  }

  if (outputs_.size() == 1) {
    outputs_[0]->reshape(output_shape);
  } else {
    outputs_[0]->reshape(output_shape);
    outputs_[1]->reshape(output_shape);
  }

  return status;
}

base::Status OpMaxPool::run() {
  device::Tensor* input_tensor = inputs_[0];
  device::Tensor* output_tensor = outputs_[0];

  // 获取输入张量的形状
  base::IntVector input_shape = input_tensor->getShape();

  // 获取池化参数
  auto param = dynamic_cast<ir::MaxPoolParam*>(op_desc_.op_param_.get());
  if (!param) {
    NNDEPLOY_LOGE("Failed to cast op param to MaxPoolParam\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  // 获取池化核大小、步长和填充
  std::vector<int> kernel_shape = param->kernel_shape_;
  std::vector<int> strides = param->strides_;
  std::vector<int> pads = param->pads_;

  // 计算输出张量的形状
  base::IntVector output_shape = output_tensor->getShape();

  // 获取输入和输出张量的数据指针
  float* input_data = reinterpret_cast<float*>(input_tensor->getData());
  float* output_data = reinterpret_cast<float*>(output_tensor->getData());

  // 执行最大池化操作
  for (int b = 0; b < input_shape[0]; ++b) {            // batch dimension
    for (int c = 0; c < input_shape[1]; ++c) {          // channel dimension
      for (int oh = 0; oh < output_shape[2]; ++oh) {    // output height
        for (int ow = 0; ow < output_shape[3]; ++ow) {  // output width
          float max_value = -std::numeric_limits<float>::max();
          int h_start = oh * strides[0] - pads[0];
          int w_start = ow * strides[1] - pads[1];
          int h_end = h_start + kernel_shape[0];
          int w_end = w_start + kernel_shape[1];

          for (int i = h_start; i < h_end; ++i) {
            for (int j = w_start; j < w_end; ++j) {
              if (i < 0 || i >= input_shape[2] || j < 0 ||
                  j >= input_shape[3]) {
                continue;
              }
              int input_index =
                  b * input_shape[1] * input_shape[2] * input_shape[3] +
                  c * input_shape[2] * input_shape[3] + i * input_shape[3] + j;
              max_value = std::max(max_value, input_data[input_index]);
            }
          }

          int output_index =
              b * output_shape[1] * output_shape[2] * output_shape[3] +
              c * output_shape[2] * output_shape[3] + oh * output_shape[3] + ow;
          output_data[output_index] = max_value;
        }
      }
    }
  }

  return base::kStatusCodeOk;
}

base::Status maxPool(device::Tensor* input,
                     std::shared_ptr<ir::MaxPoolParam> param,
                     device::Tensor* output) {
  base::Status status = base::kStatusCodeOk;

  Op* op = createOp(input->getDeviceType(), "", ir::kOpTypeMaxPool);
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

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeCpu, ir::kOpTypeMaxPool, OpMaxPool)

}  // namespace op
}  // namespace nndeploy
