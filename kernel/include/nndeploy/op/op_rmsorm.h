
// // TODO: @Leonisux:
// //
// 文件名需要修改，应该是op_rmsnorm.h，这里是op_rmsorm.h，在source/op文件夹中应该是op_rmsnorm.cc
// // TODO: @Leonisux: 宏需要修改
// #ifndef _NNDEPLOY_OP_OP_RMSNORM_H_
// #define _NNDEPLOY_OP_OP_RMSNORM_H_

// // TODO: @Leonisux: 该接口是接口类，需移除平台相关的头文件
// // #include <cuda.h>
// // #include <cuda_runtime.h>

// #include "nndeploy/op/ir.h"
// #include "nndeploy/op/op.h"

// namespace nndeploy {

// namespace op {
// template <typename T>

// class OpRMSNorm : public Op {
//  public:
//   OpRMSNorm() : Op() {}
//   virtual ~OpRMSNorm() {}

//   virtual base::Status inferShape();
//   // TODO: @Leonisux: 因为这一级的抽象主要是为了实现inferShape，
//   // TODO: @Leonisux: init、run、postRun是由真正的平台实现类实现的
//   //   virtual base::Status init();
//   //   virtual base::Status run();
//   //   virtual base::Status postRun();

//  private:
//   // TODO: @Leonisux: 输入、输出、权重是使用基类的的Tensor，此处不需要再定义
//   //   TensorWrapper<T>* decoder_out;  // [num tokens, hidden_units]
//   //   TensorWrapper<T>* decoder_residual;
//   //   LayerNormWeight<T>& attn_norm_weight;  // RMSNorm weights

//   // TODO: @Leonisux: 参数在ir中定义RMSNormParam
//   //   float eps;                             // RMSNorm eps
//   //   bool is_last;  // for print last rmsnorm output to debug

//   // TODO: @Leonisux: 该接口是接口类，需移除平台相关的头文件
//   //   static __global__ void RMSNorm(T* decoder_out, T* decoder_residual, T*
//   //   scale,
//   //                                  float eps, int num_tokens, int
//   //                                  hidden_units);
//   //   static __device__ T warpReduceSum(T val);
//   //   static __device__ T blockReduceSum(T val);
// };

// }  // namespace op
// }  // namespace nndeploy

// // TODO: @Leonisux: 该接口是接口类，需移除平台相关的头文件
// // template <typename T>
// // struct Vec {
// //   using Type = T;
// //   static constexpr int size = 0;
// // };
// // template <>
// // struct Vec<half> {
// //   using Type = half2;
// //   static constexpr int size = 2;
// // };
// // template <>
// // struct Vec<float> {
// //   using Type = float4;
// //   static constexpr int size = 4;
// // };

// #endif
