#include "op/op_func.h"

namespace nndeploy {

device::Tensor* rmsNormFunc(device::Tensor* input1, device::Tensor* input2,
                            device::Tensor* input3) {
  std::stringstream ss;

  device::Tensor* output = new device::Tensor("rms_norm.output");
  base::Status status = op::rmsNorm(input1, input2, input3, output);
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

}  // namespace nndeploy