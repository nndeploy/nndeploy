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

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

void CPUfusedresidandRMSNorm(float* h_decoder_out, 
                                    float* h_scale, float eps, int hidden_units, int num_tokens) {
    for(int b = 0; b < num_tokens; b++) {
        float inv_fenmu = 0.0f;
        float mean = 0.0f;
        float input = 0.0f;
        float sum = 0.0f;
	for (int i = 0; i < hidden_units; i++) {
            input = h_decoder_out[b * hidden_units + i];
	    sum += input * input;
        }
        mean = (float)sum / hidden_units;
        inv_fenmu = rsqrt(mean + eps);
        
        for (int i = 0; i < hidden_units; i++) {
            h_decoder_out[b * hidden_units + i] = h_decoder_out[b * hidden_units + i] * inv_fenmu * h_scale[i];
        }
    }
}

bool CheckResult(float* CPUoutput, float* GPUoutput, int output_size) {
    float fp32GPUoutput = 0.0f;
    for(int i = 0; i < output_size; i++) {
        fp32GPUoutput = (float)GPUoutput[i];
        if(fabs(CPUoutput[i] - fp32GPUoutput) > 1e-6){
            printf("the %dth res is wrong, CPUoutput = %f, GPUoutput = %f\n", i, CPUoutput[i], fp32GPUoutput);
            return false;
        }

    }
    return true;
}

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
  const int num_tokens = 64;
  const int hidden_units = 4096;
  const int total_size = num_tokens * hidden_units;
  float eps = 1e-6;
  device::Tensor* input_tensor = new device::Tensor(cuda_device, desc);
  std::mt19937 generator;

  device::randnTensor(generator, 1.0f, 100.f, input_tensor);

  input_tensor->print();

  device::Tensor* output_tensor = new device::Tensor(cuda_device, desc);
  device::randnTensor(generator, 1.0f, 100.f, output_tensor);
  output_tensor->print();


  // 我写的
  float* h_decoder_out = (float*) malloc(sizeof(float) * total_size);
  float* decoder_out = (float*) malloc(sizeof(float) * total_size);
  float* d_decoder_out;
  cudaMalloc((void**)&d_decoder_out, sizeof(float) * total_size);
  for(int i = 0; i < total_size; i++) { 
      h_decoder_out[i] = (float)(i % 2 + 1);
  }
  // to save residual used by fusedResidualAndRmsnorm
  float* d_decoder_rsd;
  cudaMalloc((void**)&d_decoder_rsd, sizeof(float) * total_size);
  //rmsnorm weights
  float* h_scale = (float*) malloc(sizeof(float) * hidden_units);
  float* d_scale;
  cudaMalloc((void**)&d_scale, sizeof(float) * hidden_units);
  for(int i = 0; i < hidden_units; i++) { 
      h_scale[i] = (float)(i % 2 + 1);
  }

  CHECK(cudaMemcpy(d_decoder_out, h_decoder_out, sizeof(float) * total_size, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_scale, h_scale, sizeof(float) * hidden_units, cudaMemcpyHostToDevice));

  std::cout << "before launch kernel" << std::endl;
  launchRMSNorm(decoder_out_tensor, decoder_rsd, scale, eps);
  std::cout << "after launch kernel" << std::endl;
  std::cout << "cuda memcpy device to host" << std::endl;
  CHECK(cudaMemcpy(decoder_out, d_decoder_out, sizeof(float) * total_size, cudaMemcpyDeviceToHost));
  float* CPUout = (float*) malloc(sizeof(float) * total_size);
  for(int i = 0; i < total_size; i++){
      CPUout[i] = (float)(i % 2 + 1);
  }
  float* cpu_scale = (float*) malloc(sizeof(float) * hidden_units);
  for(int i = 0; i < hidden_units; i++) { 
      cpu_scale[i] = (float)(i % 2 + 1);
  }
  CPUfusedresidandRMSNorm(CPUout, cpu_scale, eps, hidden_units, num_tokens);
  bool is_right = CheckResult<float>(CPUout, decoder_out, total_size);
  std::cout << "rmsnorm passed" << std::endl;
  free(h_decoder_out);
  free(h_scale);
  free(cpu_scale);
  free(CPUout);
  free(decoder_out);
  cudaFree(d_decoder_out);
  cudaFree(d_scale);

  // 新增算子测试

  return 0;
}
