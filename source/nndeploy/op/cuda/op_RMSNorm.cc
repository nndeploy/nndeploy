#include <stdio.h>
#include "nndeploy/op/op_rmsnorm.h"
#include "nndeploy/op/op.h"


namespace nndeploy {
namespace op {

template<typename T>
class CudaOpRMSNormf32 : public OpRMSNorm {
 public:
  virtual base::Status init() {
    int num_tokens = decoder_out->shape[0];
    int hidden_units = decoder_out->shape[1];
    int vec_size = Vec<T>::size;
    int num_threads = hidden_units / 4; //vec size // assume head size can be divided by 4 and 2
    T* rsd = decoder_residual->data;
  }
  virtual base::Status run() {
    dim3 grid(num_tokens);
    dim3 block(num_threads);
    RMSNorm<T><<<grid, block>>>(decoder_out->data,
                            rsd,
                            scale,//RMSNorm weights
                            eps,//RMSNorm eps
                            num_tokens,
                            hidden_units);
  }
  virtual base::Status postRun() {
    
  }
 private:
  TensorWrapper<T>* decoder_out; // [num tokens, hidden_units]
  TensorWrapper<T>* decoder_residual;
  LayerNormWeight<T>& attn_norm_weight; //RMSNorm weights
  float eps; //RMSNorm eps
  bool is_last; // for print last rmsnorm output to debug

  static __global__ void RMSNorm(T* decoder_out, T* decoder_residual, T* scale, float eps, int num_tokens, int hidden_units){
    int vec_size = Vec<T>::size;
    using Vec_t = typename Vec<T>::Type;
    float thread_sum = 0.0f;
    Vec_t* dout = reinterpret_cast<Vec_t*>(decoder_out + blockIdx.x * hidden_units);
    Vec_t* rsd = reinterpret_cast<Vec_t*>(decoder_residual + blockIdx.x * hidden_units);
    for (int idx = threadIdx.x; idx < hidden_units / vec_size; idx += blockDim.x) {
        Vec_t vec = dout[idx];
        rsd[idx] = vec;
        thread_sum += vec.x * vec.x;
        thread_sum += vec.y * vec.y;
        thread_sum += vec.z * vec.z;
        thread_sum += vec.w * vec.w;
    }
    thread_sum = blockReduceSum<float>(thread_sum);
    __shared__ float inv_mean;
    if (threadIdx.x == 0) {
        inv_mean = rsqrtf((float)thread_sum / hidden_units + eps);
    }
    __syncthreads();
    Vec_t* s = reinterpret_cast<Vec_t*>(scale);
    for (int idx = threadIdx.x; idx < hidden_units / vec_size; idx += blockDim.x) {
        Vec_t out = dout[idx];
        dout[idx].x = out.x * inv_mean * s[idx].x;
        dout[idx].y = out.y * inv_mean * s[idx].y;
        dout[idx].z = out.z * inv_mean * s[idx].z;
        dout[idx].w = out.w * inv_mean * s[idx].w;
    }
  }

  static __device__ T warpReduceSum(T val){
    for(int i = 32 / 2; i > 0; i >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    return val; // 32 threads return val, but only 0th thread is sum val
  }

  static __device__ T blockReduceSum(T val){
    int tid = threadIdx.x;
    int wid = tid / 32;
    int laneid = tid % 32;
    int warpnum = (blockDim.x + 31) / 32;
    static __shared__ T warpsum[64];
    val = warpReduceSum<T>(val);
    if(laneid == 0){
        warpsum[wid] = val;
    }
    __syncthreads();
    T sum = tid < warpnum ? warpsum[tid] : (T)0;
    sum = warpReduceSum<T>(sum); //though 0th own the sum, but dont need to shfl sync
    return sum;
  }
}
}  // namespace op
}  // namespace nndeploy