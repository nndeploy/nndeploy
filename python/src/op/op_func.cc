#include "op/op_func.h"

namespace nndeploy {

device::Tensor* rmsNormFunc(device::Tensor* input, device::Tensor* weight,
                            device::Tensor* residual,
                            std::shared_ptr<ir::RMSNormParam> param) {
  std::stringstream ss;

  device::Tensor* output = new device::Tensor("rms_norm.output");
  base::Status status = op::rmsNorm(input, weight, residual, param, output);
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

device::Tensor* batchNormFunc(
    device::Tensor* input, device::Tensor* scale, device::Tensor* bias,
    device::Tensor* mean, device::Tensor* var,
    std::shared_ptr<ir::BatchNormalizationParam> param) {
  std::stringstream ss;

  device::Tensor* output = new device::Tensor("batch_norm.output");
  base::Status status =
      op::batchNorm(input, scale, bias, mean, var, param, output);
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

}  // namespace nndeploy