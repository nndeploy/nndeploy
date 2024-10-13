#include "op/op_func.h"

namespace nndeploy {
    
device::Tensor* rmsNormFunc(device::Tensor* input1, device::Tensor* input2,
                            device::Tensor* input3) {
  std::stringstream ss;

  auto output_tensor_desc = input1->getDesc();
  auto output_tensor_device = input1->getDevice();

  device::Tensor* output =
      new device::Tensor(output_tensor_desc, "rms_norm_output");
  base::Status status = op::rmsNorm(input1, input2, input3, output);
  if (status != base::kStatusCodeOk) {
    ss << "nndeploy::op::rms_norm failed: error code "
       << base::statusCodeToString(status.getStatusCode());
    pybind11::pybind11_fail(ss.str());
  }

  return output;
}
}