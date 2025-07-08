#include "op/op_func.h"

namespace nndeploy {

device::Tensor* rmsNormFunc(device::Tensor* input, device::Tensor* weight,
                            std::shared_ptr<ir::RMSNormParam> param) {
  std::stringstream ss;

  device::Tensor* output = new device::Tensor("rms_norm.output");
  base::Status status = op::rmsNorm(input, weight, param, output);
  if (status != base::kStatusCodeOk) {
    ss << "nndeploy::op::rms_norm failed: error code "
       << base::statusCodeToString(status.getStatusCode());
    pybind11::pybind11_fail(ss.str());
  }

  return output;
}

device::Tensor* convFunc(device::Tensor* input, device::Tensor* weight,
                         device::Tensor* bias,
                         std::shared_ptr<ir::ConvParam> param) {
  std::stringstream ss;

  device::Tensor* output = new device::Tensor("conv.output");
  base::Status status = op::conv(input, weight, bias, param, output);
  if (status != base::kStatusCodeOk) {
    ss << "nndeploy::op::conv failed: error code "
       << base::statusCodeToString(status.getStatusCode());
    pybind11::pybind11_fail(ss.str());
  }

  return output;
}

device::Tensor* concatFunc(std::vector<device::Tensor *> inputs,
                         std::shared_ptr<ir::ConcatParam> param) {
  std::stringstream ss;

  device::Tensor* output = new device::Tensor("concat.output");
  base::Status status = op::concat(inputs, param, output);
  if (status != base::kStatusCodeOk) {
    ss << "nndeploy::op::concat failed: error code "
       << base::statusCodeToString(status.getStatusCode());
    pybind11::pybind11_fail(ss.str());
  }

  return output;
}

device::Tensor* batchNormFunc(
    device::Tensor* input, device::Tensor* scale, device::Tensor* shift,
    device::Tensor* mean, device::Tensor* var,
    std::shared_ptr<ir::BatchNormalizationParam> param) {
  std::stringstream ss;

  device::Tensor* output = new device::Tensor("batch_norm.output");
  base::Status status =
      op::batchNorm(input, scale, shift, mean, var, param, output);
  if (status != base::kStatusCodeOk) {
    ss << "nndeploy::op::batch_norm failed: error code "
       << base::statusCodeToString(status.getStatusCode());
    pybind11::pybind11_fail(ss.str());
  }

  return output;
}

device::Tensor* reluFunc(device::Tensor* input) {
  std::stringstream ss;

  device::Tensor* output = new device::Tensor("relu.output");
  base::Status status = op::relu(input, output);
  if (status != base::kStatusCodeOk) {
    ss << "nndeploy::op::relu failed: error code "
       << base::statusCodeToString(status.getStatusCode());
    pybind11::pybind11_fail(ss.str());
  }

  return output;
}

device::Tensor* addFunc(device::Tensor* input1, device::Tensor* input2) {
  std::stringstream ss;
  device::Tensor* result = new device::Tensor("add.output");
  base::Status status = op::add(input1, input2, result);
  if (status != base::kStatusCodeOk) {
    ss << "nndeploy::op::add failed: error code "
       << base::statusCodeToString(status.getStatusCode());
    pybind11::pybind11_fail(ss.str());
  }
  return result;
}

device::Tensor* flattenFunc(device::Tensor* input,
                            std::shared_ptr<ir::FlattenParam> param) {
  std::stringstream ss;
  device::Tensor* result = new device::Tensor("flatten.output");
  base::Status status = op::flatten(input, param, result);
  if (status != base::kStatusCodeOk) {
    ss << "nndeploy::op::flatten failed: error code "
       << base::statusCodeToString(status.getStatusCode());
    pybind11::pybind11_fail(ss.str());
  }
  return result;
}

device::Tensor* gatherFunc(device::Tensor* input, device::Tensor* index, 
                           std::shared_ptr<ir::GatherParam> param) {
  std::stringstream ss;

  device::Tensor* output = new device::Tensor("gather.output");
  base::Status status = op::gather(input, index, param, output);
  if (status != base::kStatusCodeOk) {
    ss << "nndeploy::op::gather failed: error code "
       << base::statusCodeToString(status.getStatusCode());
    pybind11::pybind11_fail(ss.str());
  }
  return output;
}

device::Tensor* gemmFunc(device::Tensor* inputs_a, device::Tensor* inputs_b,
                         device::Tensor* inputs_c,
                         std::shared_ptr<ir::GemmParam> param) {
  std::stringstream ss;
  device::Tensor* result = new device::Tensor("gemm.output");
  base::Status status = op::gemm(inputs_a, inputs_b, inputs_c, param, result);
  if (status != base::kStatusCodeOk) {
    ss << "nndeploy::op::gemm failed: error code "
       << base::statusCodeToString(status.getStatusCode());
    pybind11::pybind11_fail(ss.str());
  }
  return result;
}

device::Tensor* geluFunc(device::Tensor* input) {
  std::stringstream ss;

  device::Tensor* output = new device::Tensor("gelu.output");
  base::Status status = op::gelu(input, output);
  if (status != base::kStatusCodeOk) {
    ss << "nndeploy::op::gelu failed: error code "
       << base::statusCodeToString(status.getStatusCode());
    pybind11::pybind11_fail(ss.str());
  }

  return output;
}

device::Tensor* globalAveragepoolFunc(device::Tensor* input) {
  std::stringstream ss;
  device::Tensor* result = new device::Tensor("global_averagepool.output");
  base::Status status = op::globalAveragepool(input, result);
  if (status != base::kStatusCodeOk) {
    ss << "nndeploy::op::globalAveragepool failed: error code "
       << base::statusCodeToString(status.getStatusCode());
    pybind11::pybind11_fail(ss.str());
  }
  return result;
}

device::Tensor* maxPoolFunc(device::Tensor* input,
                            std::shared_ptr<ir::MaxPoolParam> param) {
  std::stringstream ss;
  device::Tensor* result = new device::Tensor("maxpool.output");
  base::Status status = op::maxPool(input, param, result);
  if (status != base::kStatusCodeOk) {
    ss << "nndeploy::op::maxPool failed: error code "
       << base::statusCodeToString(status.getStatusCode());
    pybind11::pybind11_fail(ss.str());
  }
  return result;
}

device::Tensor* matMulFunc(device::Tensor* input1, device::Tensor* input2, 
                           std::shared_ptr<ir::MatMulParam> param,
                           device::Tensor* bias = nullptr) {
  std::stringstream ss;
  device::Tensor* result = new device::Tensor("mat_mul.output");
  base::Status status;
  if (bias != nullptr) {
      status = op::matmul(input1, input2, param, result, bias); 
  } else {
      status = op::matmul(input1, input2, param, result); 
  }
  if (status != base::kStatusCodeOk) {
    ss << "nndeploy::op::matmul failed: error code "
       << base::statusCodeToString(status.getStatusCode());
    pybind11::pybind11_fail(ss.str());
  }
  return result;
}

device::Tensor* mulFunc(device::Tensor* input1, device::Tensor* input2) {
  std::stringstream ss;
  device::Tensor* result = new device::Tensor("mul.output");
  base::Status status = op::mul(input1, input2, result);
  if (status != base::kStatusCodeOk) {
    ss << "nndeploy::op::mul failed: error code "
       << base::statusCodeToString(status.getStatusCode());
    pybind11::pybind11_fail(ss.str());
  }
  return result;
}

device::Tensor* softmaxFunc(device::Tensor* input1,
                            std::shared_ptr<ir::SoftmaxParam> param) {
  std::stringstream ss;
  device::Tensor* result = new device::Tensor("softmax.output");
  base::Status status = op::softmax(input1, param, result);
  if (status != base::kStatusCodeOk) {
    ss << "nndeploy::op::softmax failed: error code "
       << base::statusCodeToString(status.getStatusCode());
    pybind11::pybind11_fail(ss.str());
  }
  return result;
}

device::Tensor* sigmoidFunc(device::Tensor* input1) {
  std::stringstream ss;
  device::Tensor* result = new device::Tensor("sigmoid.output");
  base::Status status = op::sigmoid(input1, result);
  if (status != base::kStatusCodeOk) {
    ss << "nndeploy::op::sigmoid failed: error code "
       << base::statusCodeToString(status.getStatusCode());
    pybind11::pybind11_fail(ss.str());
  }
  return result;
}


device::Tensor* quantizeLinearFunc(
    device::Tensor* input, device::Tensor* scale, device::Tensor* zero_point,
    std::shared_ptr<ir::QuantizeLinearParam> param) {
  std::stringstream ss;

  device::Tensor* output = new device::Tensor("quantize_linear.output");
  base::Status status =
      op::quantizeLinear(input, scale, zero_point, param, output);
  if (status != base::kStatusCodeOk) {
    ss << "nndeploy::op::quantize_linear failed: error code "
       << base::statusCodeToString(status.getStatusCode());
    pybind11::pybind11_fail(ss.str());
  }

  return output;
}

device::Tensor* dequantizeLinearFunc(
    device::Tensor* input, device::Tensor* scale, device::Tensor* zero_point,
    std::shared_ptr<ir::DequantizeLinearParam> param) {
  std::stringstream ss;

  device::Tensor* output = new device::Tensor("dequantize_linear.output");
  base::Status status =
      op::dequantizeLinear(input, scale, zero_point, param, output);
  if (status != base::kStatusCodeOk) {
    ss << "nndeploy::op::dequantize_linear failed: error code "
       << base::statusCodeToString(status.getStatusCode());
    pybind11::pybind11_fail(ss.str());
  }

  return output;
}

device::Tensor* qlinearConvFunc(device::Tensor* x, device::Tensor* x_scale,
                                device::Tensor* x_zero_point, device::Tensor* w,
                                device::Tensor* w_scale,
                                device::Tensor* w_zero_point,
                                device::Tensor* y_scale,
                                device::Tensor* y_zero_point, device::Tensor* B,
                                std::shared_ptr<ir::QLinearConvParam> param) {
  std::stringstream ss;

  device::Tensor* output = new device::Tensor("qlinear_conv.output");
  base::Status status =
      op::qLinearConv(x, x_scale, x_zero_point, w, w_scale, w_zero_point,
                       y_scale, y_zero_point, B, param, output);
  if (status != base::kStatusCodeOk) {
    ss << "nndeploy::op::qLinearConv failed: error code "
       << base::statusCodeToString(status.getStatusCode());
    pybind11::pybind11_fail(ss.str());
  }

  return output;
}

device::Tensor* whereFunc(device::Tensor* input1, device::Tensor* input2, device::Tensor* condition) {
  std::stringstream ss;

  device::Tensor* output = new device::Tensor("where.output");
  base::Status status = op::where(input1, input2, condition, output);
  if (status != base::kStatusCodeOk) {
    ss << "nndeploy::op::where failed: error code "
       << base::statusCodeToString(status.getStatusCode());
    pybind11::pybind11_fail(ss.str());
  }
  return output;
}

device::Tensor* transposeFunc(device::Tensor* input, std::shared_ptr<ir::TransposeParam> param) {
  std::stringstream ss;

  device::Tensor* output = new device::Tensor("transpose.output");
  base::Status status = op::transpose(input, param, output);
  if (status != base::kStatusCodeOk) {
    ss << "nndeploy::op::transpose failed: error code "
       << base::statusCodeToString(status.getStatusCode());
    pybind11::pybind11_fail(ss.str());
  }
  return output;
}

}  // namespace nndeploy