/**
 * nndeploy Op Demo:
 * test op
 */

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_rmsnorm.h"

using namespace nndeploy;

double rsqrt_cpu(double x) {
  if (x <= 0.0) {
    return NAN;
  }
  return 1.0 / sqrt(x);
}

void CPUfusedresidandRMSNorm(float* h_decoder_out, float* h_scale, float eps,
                             int hidden_units, int num_tokens) {
  for (int b = 0; b < num_tokens; b++) {
    float inv_fenmu = 0.0f;
    float mean = 0.0f;
    float input = 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < hidden_units; i++) {
      input = h_decoder_out[b * hidden_units + i];
      sum += input * input;
    }
    mean = (float)sum / hidden_units;
    inv_fenmu = rsqrt_cpu(mean + eps);
    // inv_fenmu = rsqrt(mean + eps);

    for (int i = 0; i < hidden_units; i++) {
      h_decoder_out[b * hidden_units + i] =
          h_decoder_out[b * hidden_units + i] * inv_fenmu * h_scale[i];
    }
  }
}

bool CheckResult(float* CPUoutput, float* GPUoutput, int output_size) {
  float fp32GPUoutput = 0.0f;
  for (int i = 0; i < output_size; i++) {
    fp32GPUoutput = (float)GPUoutput[i];
    if (fabs(CPUoutput[i] - fp32GPUoutput) > 1e-6) {
      printf("the %dth res is wrong, CPUoutput = %f, GPUoutput = %f\n", i,
             CPUoutput[i], fp32GPUoutput);
      return false;
    }
    /* else {
      printf("the %dth res is true, CPUoutput = %f, GPUoutput = %f\n", i,
             CPUoutput[i], fp32GPUoutput);
    }*/
  }
  return true;
}

int main(int argc, char* argv[]) {
  base::DeviceType cuda_device_type;
  base::DeviceType cpu_device_type;
  // cuda_device_type.code_ = base::kDeviceTypeCodeX86;
  cuda_device_type.code_ = base::kDeviceTypeCodeCuda;
  cpu_device_type.code_ = base::kDeviceTypeCodeCpu;
  cuda_device_type.device_id_ = 0;
  device::Device* cuda_device = device::getDevice(cuda_device_type);
  device::Device* cpu_device = device::getDevice(cpu_device_type);
  device::TensorDesc desc;
  const int num_tokens = 32;
  const int hidden_units = 4096;
  const int total_size = num_tokens * hidden_units;
  float eps = 1e-6;
  desc.data_type_ = base::dataTypeOf<float>();
  desc.data_format_ = base::kDataFormatNC;
  desc.shape_ = {num_tokens, hidden_units};

  device::Tensor* d_out = new device::Tensor(cuda_device, desc);
  device::Tensor* h_out = new device::Tensor(cpu_device, desc);
  device::Tensor* d_input = new device::Tensor(cuda_device, desc);
  // test data prepare
  float* data_test = (float*)malloc(sizeof(float) * total_size);
  for (int i = 0; i < total_size; i++) {
    data_test[i] = (float)(i % 2 + 1);
  }
  device::Tensor* h_input =
      new device::Tensor(cpu_device, desc, (void*)data_test);
  h_input->copyTo(d_input);

  // to save residual
  device::Tensor* d_decoder_rsd = new device::Tensor(cuda_device, desc);
  // rmsnorm weights
  device::Tensor* d_scale = new device::Tensor(cuda_device, desc);
  // weight copy
  h_input->copyTo(d_scale);

  std::cout << "before launch kernel" << std::endl;
  std::shared_ptr<base::Param> rmsnorm_param =
      ir::createOpParam(ir::kOpTypeRMSNorm);
  op::rmsNorm(d_input, d_scale, rmsnorm_param, d_out);
  std::cout << "after launch kernel" << std::endl;
  d_out->copyTo(h_out);
  std::cout << "cuda memcpy device to host" << std::endl;

  float* CPUout = (float*)malloc(sizeof(float) * total_size);
  for (int i = 0; i < total_size; i++) {
    CPUout[i] = (float)(i % 2 + 1);
  }
  float* cpu_scale = (float*)malloc(sizeof(float) * hidden_units);
  for (int i = 0; i < hidden_units; i++) {
    cpu_scale[i] = (float)(i % 2 + 1);
  }
  CPUfusedresidandRMSNorm(CPUout, cpu_scale, eps, hidden_units, num_tokens);
  bool is_right = CheckResult(CPUout, (float*)h_out->getData(), total_size);
  if (is_right) {
    std::cout << "rmsnorm passed" << std::endl;
  }
  free(data_test);

  return 0;
}
