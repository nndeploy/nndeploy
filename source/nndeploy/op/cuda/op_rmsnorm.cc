#include "nndeploy/op/op_rmsnorm.h"

#include <stdio.h>

#include "nndeploy/device/cuda/cuda_device.h"
#include "nndeploy/op/ir.h"
#include "nndeploy/op/op.h"

namespace nndeploy {
namespace op {

template <typename T>
struct Vec {
  using Type = T;
  static constexpr int size = 0;
};

// TODO: @Leonisux cuda 有 half类型吗？
template <>
struct Vec<half> {  // 到底是fp16 or bfp16
  using Type = half2;
  static constexpr int size = 2;
};
template <>
struct Vec<float> {
  using Type = float4;
  static constexpr int size = 4;
};

// 这些函数定义为文件内部函数，不需要暴露给外部，也不需要与类绑定
static __global__ void RMSNorm(T* decoder_out, T* decoder_residual, T* scale,
                               float eps, int num_tokens, int hidden_units) {
  int vec_size = Vec<T>::size;
  using Vec_t = typename Vec<T>::Type;
  float thread_sum = 0.0f;
  Vec_t* dout =
      reinterpret_cast<Vec_t*>(decoder_out + blockIdx.x * hidden_units);
  Vec_t* rsd =
      reinterpret_cast<Vec_t*>(decoder_residual + blockIdx.x * hidden_units);
  for (int idx = threadIdx.x; idx < hidden_units / vec_size;
       idx += blockDim.x) {
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
  for (int idx = threadIdx.x; idx < hidden_units / vec_size;
       idx += blockDim.x) {
    Vec_t out = dout[idx];
    dout[idx].x = out.x * inv_mean * s[idx].x;
    dout[idx].y = out.y * inv_mean * s[idx].y;
    dout[idx].z = out.z * inv_mean * s[idx].z;
    dout[idx].w = out.w * inv_mean * s[idx].w;
  }
}

static __device__ T warpReduceSum(T val) {
  for (int i = 32 / 2; i > 0; i >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, i);
  }
  return val;  // 32 threads return val, but only 0th thread is sum val
}

static __device__ T blockReduceSum(T val) {
  int tid = threadIdx.x;
  int wid = tid / 32;
  int laneid = tid % 32;
  int warpnum = (blockDim.x + 31) / 32;
  static __shared__ T warpsum[64];
  val = warpReduceSum<T>(val);
  if (laneid == 0) {
    warpsum[wid] = val;
  }
  __syncthreads();
  T sum = tid < warpnum ? warpsum[tid] : (T)0;
  sum = warpReduceSum<T>(
      sum);  // though 0th own the sum, but dont need to shfl sync
  return sum;
}

// template <typename T>// 这个模板不需要用
class CudaOpRMSNorm : public OpRMSNorm {
 public:
  // 需要写构造函数
  CudaOpRMSNorm() : OpRMSNorm() {}
  // 需要写析构函数
  virtual ~CudaOpRMSNorm() {}

  // virtual base::Status init() {}
  virtual base::Status run() {
    base::Status status = base::kStatusCodeOk;

    // 写在run函数内部即可
    // int num_tokens = decoder_out->shape[0];
    // int hidden_units = decoder_out->shape[1];
    // int vec_size = Vec<T>::size;
    // // vec size // assume head size can be divided by 4 and 2
    // int num_threads = hidden_units / 4;
    // T* rsd = decoder_residual->data;

    // dim3 grid(num_tokens);
    // dim3 block(num_threads);
    // RMSNorm<T><<<grid, block>>>(decoder_out->data, rsd,
    //                             scale,  // RMSNorm weights
    //                             eps,    // RMSNorm eps
    //                             num_tokens, hidden_units);

    auto param = dynamic_cast<RMSNormParam*>(op_desc_.op_param_.get());
    NNDEPLOY_CHECK_PARAM_NULL_RET_STATUS(param,
                                         "op_desc_.op_param_ is nullptr");
    float eps = param->eps_;

    int num_tokens = inputs_[0]->getShapeIndex(0);
    int hidden_units = inputs_[0]->getShapeIndex(1);
    // vec size // assume head size can be divided by 4 and 2
    int num_threads = hidden_units / 4;

    void* input_data = inputs_[0]->getData();
    void* scale_data = inputs_[1]->getData();  // RMSNorm weights
    void* output_data = outputs_[0]->getData();

    base::DataType data_type = inputs_[0]->getDataType();

    dim3 grid(num_tokens);
    dim3 block(num_threads);

    if (data_type.code_ == base::kDataTypeCodeFp && data_type.bits_ == 16) {
      RMSNorm<half><<<grid, block>>>((half*)input_data, (half*)output_data,
                                     (half*)scale_data,  // RMSNorm weights
                                     eps,                // RMSNorm eps
                                     num_tokens, hidden_units);
    } else if (data_type.code_ == base::kDataTypeCodeFp &&
               data_type.bits_ == 32) {
      RMSNorm<float><<<grid, block>>>((float*)input_data, (float*)output_data,
                                      (float*)scale_data,  // RMSNorm weights
                                      eps,                 // RMSNorm eps
                                      num_tokens, hidden_units);
    } else {
      NNDEPLOY_LOGE("Unsupported precision type: %d\n", precision_type_);
      return base::kStatusCodeErrorInvalidParam;
    }

    return status;
  }
  // virtual base::Status postRun() {}

  //  private:
  // 这个是输入，对应inputs_[0]
  // TensorWrapper<T>* decoder_out;  // [num tokens, hidden_units]
  // 这个是输出，对应outputs_[0]
  // TensorWrapper<T>* decoder_residual;
  // 这个是参数，对应inputs_[1]
  // LayerNormWeight<T>& attn_norm_weight;  // RMSNorm weights，这个没有被用到呀
  // float eps;                             // RMSNorm eps
  // bool is_last;  // for print last rmsnorm output to debug
}

}  // namespace op
}  // namespace nndeploy
