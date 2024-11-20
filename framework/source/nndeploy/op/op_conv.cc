
#include "nndeploy/op/op_conv.h"

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

base::Status OpConv::inferShape() {
  base::Status status = base::kStatusCodeOk;

  base::IntVector input_shape = inputs_[0]->getShape();
  if (input_shape.size() < 2) {
    NNDEPLOY_LOGE("input_shape.size() < 2.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.size() - 2);
  // 参数
  auto param = dynamic_cast<ir::ConvParam *>(op_desc_.op_param_.get());
  NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param, "op_desc_.op_param_ is nullptr");
  std::vector<int> dilations = param->dilations_;
  if (dilations.size() == 0) {
    dilations.resize(n_input_dims, 1);
    param->dilations_ = dilations;
  }
  std::vector<int> kernel_shape = param->kernel_shape_;
  if (kernel_shape.size() == 0) {
    auto second_input_shape = inputs_[1]->getShape();
    for (int i = 2; i < second_input_shape.size(); ++i) {
      kernel_shape.push_back(second_input_shape[i]);
    }
  }
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
  output_shape[0] = input_shape[0];
  output_shape[1] = inputs_[1]->getShape()[0];
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

    // how many times we can move the kernel from it's initial position, based
    // on the stride
    int64_t strided_kernel_positions =
        (effective_input_size - effective_kernel_shape[i]) / strides[i];

    // add in the initial position
    output_shape[new_i] = 1 + strided_kernel_positions;
  }

  outputs_[0]->reshape(output_shape);
  // outputs_[0]->print();

  return status;
}

base::Status OpConv::run() {
  base::Status status = base::kStatusCodeOk;
  // 获取输入和权重张量
  device::Tensor *input_tensor = inputs_[0];
  device::Tensor *weight_tensor = inputs_[1];
  device::Tensor *bias_tensor = inputs_.size() > 2 ? inputs_[2] : nullptr;
  device::Tensor *output_tensor = outputs_[0];

  // 获取输入和权重的维度信息
  auto input_shape = input_tensor->getShape();
  auto weight_shape = weight_tensor->getShape();
  auto output_shape = output_tensor->getShape();

  // 获取卷积参数
  auto param = dynamic_cast<ir::ConvParam *>(op_desc_.op_param_.get());
  std::vector<int> pads = param->pads_;
  std::vector<int> strides = param->strides_;
  std::vector<int> dilations = param->dilations_;
  std::vector<int> kernel_shape = param->kernel_shape_;

  // 执行卷积操作
  float *input_data = static_cast<float *>(input_tensor->getData());
  float *weight_data = static_cast<float *>(weight_tensor->getData());
  float *output_data = static_cast<float *>(output_tensor->getData());
  float *bias_data =
      bias_tensor ? static_cast<float *>(bias_tensor->getData()) : nullptr;

  int output_height = output_shape[2];
  int output_width = output_shape[3];

  for (int n = 0; n < input_shape[0]; ++n) {
    for (int f = 0; f < weight_shape[0]; ++f) {
      for (int y = 0; y < output_height; ++y) {
        for (int x = 0; x < output_width; ++x) {
          float value = 0.0f;
          for (int k = 0; k < input_shape[1]; ++k) {
            for (int i = 0; i < kernel_shape[0]; ++i) {
              for (int j = 0; j < kernel_shape[1]; ++j) {
                int input_y = y * strides[0] + i * dilations[0] - pads[0];
                int input_x = x * strides[1] + j * dilations[1] - pads[1];
                if (input_y >= 0 && input_y < input_shape[2] && input_x >= 0 &&
                    input_x < input_shape[3]) {
                  value += input_data[n * input_shape[1] * input_shape[2] *
                                          input_shape[3] +
                                      k * input_shape[2] * input_shape[3] +
                                      input_y * input_shape[3] + input_x] *
                           weight_data[f * input_shape[1] * kernel_shape[0] *
                                           kernel_shape[1] +
                                       k * kernel_shape[0] * kernel_shape[1] +
                                       i * kernel_shape[1] + j];
                }
              }
            }
          }

          // bias  add
          if (bias_data) {
            value += bias_data[f];
          }

          // 融合算子
          if (param->is_fusion_op_) {
            switch (param->activate_op_) {
              case ir::kOpTypeRelu:
                value = value > 0.0f ? value : 0.0f;
                break;

              default:
                NNDEPLOY_LOGI("not implemented.\n");
                return base::kStatusCodeOk;
            }
          }
          output_data[n * weight_shape[0] * output_height * output_width +
                      f * output_height * output_width + y * output_width + x] =
              value;
        }
      }
    }
  }

  return status;
}

base::Status conv(device::Tensor *input, device::Tensor *weight,
                  device::Tensor *bias, std::shared_ptr<ir::ConvParam> param,
                  device::Tensor *output) {
  base::Status status = base::kStatusCodeOk;

  Op *op = createOp(input->getDeviceType(), "", ir::kOpTypeConv);
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
  if (bias != nullptr) {
    status = op->setInput(bias, 2);
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  }
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

REGISTER_OP_IMPLEMENTION(base::DeviceTypeCode::kDeviceTypeCodeCpu,
                         ir::kOpTypeConv, OpConv)

}  // namespace op
}  // namespace nndeploy
