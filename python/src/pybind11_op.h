#ifndef _PYBIND11_OP_H_
#define _PYBIND11_OP_H_

#include "nndeploy/op/op_rmsnorm.h"

/*
 * op的func层，创建输出Tensor（不分配内存）、输入检查
 */

using namespace nndeploy;

std::unique_ptr<device::Tensor> rmsNormFunc(device::Tensor* input1,
                                            device::Tensor* input2) {
  std::stringstream ss;
  if (input1->getDesc() != input2->getDesc()) {
    ss << "nndeploy::op::add failed:  input1 and input2 are different in "
          "shape、stride or data type!";
    pybind11::pybind11_fail(ss.str());
  }

  auto output_tensor_desc = input1->getDesc();
  auto output_tensor_device = input1->getDevice();

  device::Tensor* output = new device::Tensor(output_tensor_device, output_tensor_desc);
  base::Status status = op::rmsNorm(input1, input2, output);
    if(status!=base::kStatusCodeOk){
    ss << "nndeploy::op::add failed: error code " << status.desc();
    pybind11::pybind11_fail(ss.str());
    }

    return std::unique_ptr<device::Tensor>(output);
}

#endif