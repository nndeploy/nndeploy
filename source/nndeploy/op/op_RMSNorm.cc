#include <stdio.h>
// #include "cuda_debug_utils.cuh"
// #include "rmsnorm_kernel.h"
template<typename T>
__device__ T warpReduceSum(T val){
    for(int i = 32 / 2; i > 0; i >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    return val; // 32 threads return val, but only 0th thread is sum val
}

template <typename T>
__global__ void RMSNorm(T* decoder_out, T* decoder_residual, T* scale, float eps, int num_tokens, int hidden_units){
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