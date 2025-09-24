#include "nndeploy/op/op_qlinear_conv.h"

#include "nndeploy/base/common.h"
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
#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {
base::Status OpQLinearConv::inferShape() {
  base::Status status = base::kStatusCodeOk;

  if (inputs_.size() < 8) {
    NNDEPLOY_LOGE(
        "QLinearConv requires at least 8 inputs (x, x_scale, x_zero_point, w, "
        "w_scale, w_zero_point, y_scale, y_zero_point)");
    return base::kStatusCodeErrorInvalidParam;
  }

  device::Tensor* x = inputs_[0];
  device::Tensor* w = inputs_[3];
  const auto& x_shape = x->getShape();
  const auto& w_shape = w->getShape();

  base::IntVector input_shape = inputs_[0]->getShape();
  if (input_shape.size() < 2) {
    NNDEPLOY_LOGE("input_shape.size() < 2.\n");
    return base::kStatusCodeErrorInvalidParam;
  }
  // first dim is the batch axis and the next is the number of channels.
  size_t n_input_dims = static_cast<size_t>(input_shape.size() - 2);

  // 获取卷积参数
  auto param = dynamic_cast<ir::QLinearConvParam*>(op_desc_.op_param_.get());
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

template <typename T>
base::Status OpQLinearConv::qLinearConvImpl(
    device::Tensor* x, device::Tensor* x_scale, device::Tensor* x_zero_point,
    device::Tensor* w, device::Tensor* w_scale, device::Tensor* w_zero_point,
    device::Tensor* y_scale, device::Tensor* y_zero_point, device::Tensor* B,
    void* output_data) {
  const T* x_ptr = reinterpret_cast<T*>(x->getData());
  const float* x_scale_ptr = reinterpret_cast<float*>(x_scale->getData());
  const T* x_zp_ptr = reinterpret_cast<T*>(x_zero_point->getData());
  const T* w_ptr = reinterpret_cast<T*>(w->getData());
  const float* w_scale_ptr = reinterpret_cast<float*>(w_scale->getData());
  const T* w_zp_ptr = reinterpret_cast<T*>(w_zero_point->getData());
  const float* y_scale_ptr = reinterpret_cast<float*>(y_scale->getData());
  const T* y_zp_ptr = reinterpret_cast<T*>(y_zero_point->getData());
  const int32_t* B_ptr = B ? reinterpret_cast<int32_t*>(B->getData()) : nullptr;
  T* output_ptr = reinterpret_cast<T*>(output_data);

  auto param = dynamic_cast<ir::QLinearConvParam*>(op_desc_.op_param_.get());
  std::vector<int> kernel_shape = param->kernel_shape_;
  std::vector<int> strides = param->strides_;
  std::vector<int> dilations = param->dilations_;
  std::vector<int> pads = param->pads_;

  const auto& x_shape = x->getShape();
  const auto& w_shape = w->getShape();
  const auto& output_shape = outputs_[0]->getShape();

  long output_size = std::accumulate(output_shape.begin(), output_shape.end(),
                                     1, std::multiplies<int>());

  long x_size = std::accumulate(x_shape.begin(), x_shape.end(), 1,
                                std::multiplies<int>());

  long w_size = std::accumulate(w_shape.begin(), w_shape.end(), 1,
                                std::multiplies<int>());

  // 转换为int32计算 防止溢出
  int32_t* x_dequantized = new int32_t[x_size];
  int32_t* w_dequantized = new int32_t[w_size];

  for (size_t i = 0; i < x_size; ++i) {
    x_dequantized[i] =
        static_cast<int32_t>(x_ptr[i]) - static_cast<int32_t>(x_zp_ptr[0]);
  }

  for (size_t i = 0; i < w_size; ++i) {
    w_dequantized[i] =
        static_cast<int32_t>(w_ptr[i]) - static_cast<int32_t>(w_zp_ptr[0]);
  }

  // Convolution implementation
  int32_t* conv_result = new int32_t[output_size];
  int output_height = output_shape[2];
  int output_width = output_shape[3];
  for (int n = 0; n < x_shape[0]; ++n) {
    for (int f = 0; f < w_shape[0]; ++f) {
      for (int y = 0; y < output_height; ++y) {
        for (int x = 0; x < output_width; ++x) {
          int32_t value = 0;

          for (int k = 0; k < x_shape[1]; ++k) {
            for (int i = 0; i < kernel_shape[0]; ++i) {
              for (int j = 0; j < kernel_shape[1]; ++j) {
                int input_y = y * strides[0] + i * dilations[0] - pads[0];
                int input_x = x * strides[1] + j * dilations[1] - pads[1];

                if (input_y >= 0 && input_y < x_shape[2] && input_x >= 0 &&
                    input_x < x_shape[3]) {
                  int32_t x_val =
                      x_dequantized[n * x_shape[1] * x_shape[2] * x_shape[3] +
                                    k * x_shape[2] * x_shape[3] +
                                    input_y * x_shape[3] + input_x];
                  int32_t w_val =
                      w_dequantized[f * x_shape[1] * kernel_shape[0] *
                                        kernel_shape[1] +
                                    k * kernel_shape[0] * kernel_shape[1] +
                                    i * kernel_shape[1] + j];

                  value += x_val * w_val;
                }
              }
            }
          }

          // Add bias
          if (B_ptr) {
            value += B_ptr[f];
          }

          conv_result[n * w_shape[0] * output_height * output_width +
                      f * output_height * output_width + y * output_width + x] =
              value;
        }
      }
    }
  }

  // Requantize the result
  float scale = x_scale_ptr[0] * w_scale_ptr[0] / y_scale_ptr[0];
  for (size_t i = 0; i < output_size; ++i) {
    float quantized_value = conv_result[i] * scale + y_zp_ptr[0];
    if (std::is_integral<T>::value) {
      // int8范围
      if (std::is_signed<T>::value) {
        quantized_value = std::max(-128.0f, std::min(127.0f, quantized_value));
      } else {
        // uint8范围

        quantized_value = std::max(0.0f, std::min(255.0f, quantized_value));
      }
    }
    output_ptr[i] = static_cast<T>(std::round(quantized_value));
  }

  // Cleanup
  delete[] x_dequantized;
  delete[] w_dequantized;
  delete[] conv_result;

  return base::kStatusCodeOk;
}

// 显式模板实例化
template base::Status OpQLinearConv::qLinearConvImpl<int8_t>(
    device::Tensor* x, device::Tensor* x_scale, device::Tensor* x_zero_point,
    device::Tensor* w, device::Tensor* w_scale, device::Tensor* w_zero_point,
    device::Tensor* y_scale, device::Tensor* y_zero_point, device::Tensor* B,
    void* output_data);

template base::Status OpQLinearConv::qLinearConvImpl<uint8_t>(
    device::Tensor* x, device::Tensor* x_scale, device::Tensor* x_zero_point,
    device::Tensor* w, device::Tensor* w_scale, device::Tensor* w_zero_point,
    device::Tensor* y_scale, device::Tensor* y_zero_point, device::Tensor* B,
    void* output_data);

base::Status OpQLinearConv::run() {
  device::Tensor* x = inputs_[0];
  device::Tensor* x_scale = inputs_[1];
  device::Tensor* x_zero_point = inputs_[2];
  device::Tensor* w = inputs_[3];
  device::Tensor* w_scale = inputs_[4];
  device::Tensor* w_zero_point = inputs_[5];
  device::Tensor* y_scale = inputs_[6];
  device::Tensor* y_zero_point = inputs_[7];
  device::Tensor* B = inputs_.size() > 8 ? inputs_[8] : nullptr;
  device::Tensor* output = outputs_[0];

  base::DataType x_dtype = x->getDataType();

  if (x_dtype != base::dataTypeOf<int8_t>() &&
      x_dtype != base::dataTypeOf<uint8_t>()) {
    NNDEPLOY_LOGE("Unsupported input dtype");
    return base::kStatusCodeErrorInvalidParam;
  }

  void* output_data = output->getData();

  auto param = dynamic_cast<ir::QLinearConvParam*>(op_desc_.op_param_.get());

  switch (x_dtype.code_) {
    case base::kDataTypeCodeInt:
      if (x_dtype.bits_ == 8) {
        return this->qLinearConvImpl<int8_t>(x, x_scale, x_zero_point, w,
                                             w_scale, w_zero_point, y_scale,
                                             y_zero_point, B, output_data);
      }
      break;
    case base::kDataTypeCodeUint:
      if (x_dtype.bits_ == 8) {
        return this->qLinearConvImpl<uint8_t>(x, x_scale, x_zero_point, w,
                                              w_scale, w_zero_point, y_scale,
                                              y_zero_point, B, output_data);
      }
      break;
    default:
      break;
  }

  NNDEPLOY_LOGE("Unsupported qlinear conv input dtype");
  return base::kStatusCodeErrorInvalidParam;
}

base::Status qLinearConv(device::Tensor* x, device::Tensor* x_scale,
                         device::Tensor* x_zero_point, device::Tensor* w,
                         device::Tensor* w_scale, device::Tensor* w_zero_point,
                         device::Tensor* y_scale, device::Tensor* y_zero_point,
                         device::Tensor* B,
                         std::shared_ptr<ir::QLinearConvParam> param,
                         device::Tensor* output) {
  base::Status status = base::kStatusCodeOk;

  Op* op = createOp(x->getDeviceType(), "", ir::kOpTypeQLinearConv);
  if (op == nullptr) {
    NNDEPLOY_LOGE("createOp failed");
    return base::kStatusCodeErrorNotImplement;
  }
  status = op->setParam(param);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setParam failed");
  status = op->setInput(x, 0);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setInput(x_scale, 1);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setInput(x_zero_point, 2);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setInput(w, 3);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setInput(w_scale, 4);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setInput(w_zero_point, 5);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setInput(y_scale, 6);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  status = op->setInput(y_zero_point, 7);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setInput failed");
  if (B != nullptr) {
    status = op->setInput(B, 8);
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

REGISTER_OP_IMPLEMENTION(kDeviceTypeCodeCpu, ir::kOpTypeQLinearConv,
                         OpQLinearConv)

}  // namespace op
}  // namespace nndeploy
