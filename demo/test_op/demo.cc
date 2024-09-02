#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_rmsnorm.h"

using namespace nndeploy;

int main(int argc, char* argv[]) {
  base::DeviceType cuda_device_type;
  // cuda_device_type.code_ = base::kDeviceTypeCodeX86;
  cuda_device_type.code_ = base::kDeviceTypeCodeCuda;
  cuda_device_type.device_id_ = 0;
  device::Device* cuda_device = device::getDevice(cuda_device_type);
  device::TensorDesc desc;
  desc.data_type_ = base::dataTypeOf<float>();
  desc.data_format_ = base::kDataFormatNCHW;
  desc.shape_ = {1, 3, 8, 8};

  device::Tensor* input_tensor = new device::Tensor(cuda_device, desc);
  std::mt19937 generator;

  device::randnTensor(generator, 1.0f, 100.f, input_tensor);

  input_tensor->print();

  device::Tensor* output_tensor = new device::Tensor(cuda_device, desc);
  device::randnTensor(generator, 1.0f, 100.f, output_tensor);
  output_tensor->print();

  // 新增算子测试

  return 0;
}
