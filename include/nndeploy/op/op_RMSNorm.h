#ifndef _NNDEPLOY_OP_OP_CONV_H_
#define _NNDEPLOY_OP_OP_CONV_H_

#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"
#include <cuda_runtime.h>
#include <cuda.h>

namespace nndeploy {

namespace op {
template<typename T>

class OpRMSNorm : public Op {
 public:
  virtual base::Status init()
  virtual base::Status run()
  virtual base::Status postRun()
 private:
  TensorWrapper<T>* decoder_out; // [num tokens, hidden_units]
  TensorWrapper<T>* decoder_residual;
  LayerNormWeight<T>& attn_norm_weight; //RMSNorm weights
  float eps; //RMSNorm eps
  bool is_last; // for print last rmsnorm output to debug

  static __global__ void RMSNorm(T* decoder_out, T* decoder_residual, T* scale, float eps, int num_tokens, int hidden_units);
  static __device__ T warpReduceSum(T val);
  static __device__ T blockReduceSum(T val);
};

// NNDEPLOY_CC_API base::Status RMSNorm(device::Tensor *input, device::Tensor *weight,
//                                   device::Tensor *bias,
//                                   std::shared_ptr<ConvParam> param,
//                                   device::Tensor *output);

}  // namespace op
}  // namespace nndeploy

template<typename T>
struct Vec {
    using Type = T;
    static constexpr int size = 0;
};
template<>
struct Vec<half> {
    using Type = half2; 
    static constexpr int size = 2;
};
template<>
struct Vec<float> {
    using Type = float4;
    static constexpr int size = 4;
};

#endif