#include <stdio.h>

#include "nndeploy/device/cuda/cuda_device.h"
#include "nndeploy/ir/ir.h"
#include "nndeploy/op/op.h"
#include "nndeploy/op/op_rmsnorm.h"

namespace nndeploy {
namespace op {

static __device__ float warpReduceSum(float val) {
  for (int i = 32 / 2; i > 0; i >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, i);
  }
  return val;  // 32 threads return val, but only 0th thread is sum val
}

static __device__ float blockReduceSum(float val) {
  int tid = threadIdx.x;
  int wid = tid / 32;
  int laneid = tid % 32;
  int warpnum = (blockDim.x + 31) / 32;
  static __shared__ float warpsum[64];
  val = warpReduceSum(val);
  if (laneid == 0) {
    warpsum[wid] = val;
  }
  __syncthreads();
  float sum = tid < warpnum ? warpsum[tid] : 0.0f;
  sum =
      warpReduceSum(sum);  // though 0th own the sum, but dont need to shfl sync
  return sum;
}

static __global__ void RMSNormKernel(float* decoder_out,
                                     float* normalized_out,
                                     float* decoder_residual, 
                                     float* scale,
                                     float eps, 
                                     int num_tokens,
                                     int hidden_units) {
  int vec_size = 4;
  float thread_sum = 0.0f;
<<<<<<< HEAD
  float4* dout = (float4*)(decoder_out + blockIdx.x * hidden_units / vec_size);
  float4* nout = (float4*)(normalized_out + blockIdx.x * hidden_units / vec_size);
  float4* rsd = (float4*)(decoder_residual + blockIdx.x * hidden_units / vec_size);
=======
  float4* dout = (float4*)(decoder_out + blockIdx.x * hidden_units);
  float4* nout = (float4*)(normalized_out + blockIdx.x * hidden_units);
  float4* rsd = (float4*)(decoder_residual + blockIdx.x * hidden_units);
  //float4* dout = (float4*)(decoder_out + blockIdx.x * hidden_units / vec_size);
  //float4* nout =
      //(float4*)(normalized_out + blockIdx.x * hidden_units / vec_size);
  //float4* rsd =
      //(float4*)(decoder_residual + blockIdx.x * hidden_units / vec_size);
>>>>>>> main
  for (int idx = threadIdx.x; idx < hidden_units / vec_size;
       idx += blockDim.x) {
    float4 vec = dout[idx];
    rsd[idx] = vec;
    thread_sum += vec.x * vec.x;
    thread_sum += vec.y * vec.y;
    thread_sum += vec.z * vec.z;
    thread_sum += vec.w * vec.w;
  }
  thread_sum = blockReduceSum(thread_sum);
  __shared__ float inv_mean;
  if (threadIdx.x == 0) {
    inv_mean = rsqrtf((float)thread_sum / hidden_units + eps);
  }
  __syncthreads();

  float4* s = (float4*)scale;
  for (int idx = threadIdx.x; idx < hidden_units / vec_size;
       idx += blockDim.x) {
    float4 out = dout[idx];
    nout[idx].x = out.x * inv_mean * s[idx].x;
    nout[idx].y = out.y * inv_mean * s[idx].y;
    nout[idx].z = out.z * inv_mean * s[idx].z;
    nout[idx].w = out.w * inv_mean * s[idx].w;
  }
}


class CudaOpRMSNorm : public OpRMSNorm {
 public:
  CudaOpRMSNorm() : OpRMSNorm() {}
  virtual ~CudaOpRMSNorm() {}

  // virtual base::Status init() {}
  virtual base::Status run() {
    base::Status status = base::kStatusCodeOk;

    // auto param = dynamic_cast<ir::RMSNormParam*>(op_desc_.op_param_.get());
    // NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param,
    //                                      "op_desc_.op_param_ is nullptr");
    // float eps = param->eps_;
    float eps = 1e-6;

    int num_tokens = inputs_[0]->getShapeIndex(0);
    int hidden_units = inputs_[0]->getShapeIndex(1);
    // vec size // assume head size can be divided by 4 and 2
    int num_threads = hidden_units / 4;

    void* input_data = inputs_[0]->getData();
    void* scale_data = inputs_[1]->getData();  // RMSNorm weights
    void* rsd = inputs_[2]->getData();  

    void* output_data = outputs_[0]->getData();

    // base::DataType data_type = inputs_[0]->getDataType();
<<<<<<< HEAD

=======
>>>>>>> main
    dim3 grid(num_tokens);
    dim3 block(num_threads);
    RMSNormKernel<<<grid, block>>>((float*)input_data, (float*)output_data,
                                   (float*)rsd,
                                   (float*)scale_data,  // RMSNorm weights
                                   eps,                 // RMSNorm eps
                                   num_tokens, hidden_units);

    return status;
  }
};

REGISTER_OP_IMPLEMENTION(base::DeviceTypeCode::kDeviceTypeCodeCuda,
                         ir::kOpTypeRMSNorm, CudaOpRMSNorm)

}  // namespace op
}  // namespace nndeploy
