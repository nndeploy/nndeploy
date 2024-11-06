
#ifndef _NNDEPLOY_INFERENCE_NCNN_NCNN_INFERENCE_PARAM_H_
#define _NNDEPLOY_INFERENCE_NCNN_NCNN_INFERENCE_PARAM_H_

#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/ncnn/ncnn_include.h"

namespace nndeploy {
namespace inference {

class NcnnInferenceParam : public InferenceParam {
 public:
  NcnnInferenceParam();
  virtual ~NcnnInferenceParam();

  NcnnInferenceParam(const NcnnInferenceParam &param) = default;
  NcnnInferenceParam &operator=(const NcnnInferenceParam &param) = default;

  PARAM_COPY(NcnnInferenceParam)
  PARAM_COPY_TO(NcnnInferenceParam)

  // zh
  base::Status parse(const std::string &json, bool is_path = true);
  // zh
  virtual base::Status set(const std::string &key, base::Any &any);
  // zh
  virtual base::Status get(const std::string &key, base::Any &any);

  // light mode
  // intermediate blob will be recycled when enabled
  // enabled by default
  bool lightmode_ = true;

  // the time openmp threads busy-wait for more work before going to sleep
  // default value is 20ms to keep the cores enabled
  // without too much extra power consumption afterwards
  int openmp_blocktime_ = 20;

  // enable winograd convolution optimization
  // improve convolution 3x3 stride1 performance, may consume more memory
  // changes should be applied before loading network structure and weight
  // enabled by default
  bool use_winograd_convolution_ = true;

  // enable sgemm convolution optimization
  // improve convolution 1x1 stride1 performance, may consume more memory
  // changes should be applied before loading network structure and weight
  // enabled by default
  bool use_sgemm_convolution_ = true;

  // enable quantized int8 inference
  // use low-precision int8 path for quantized model
  // changes should be applied before loading network structure and weight
  // enabled by default
  bool use_int8_inference_ = true;

  // enable bf16 data type for storage
  // improve most operator performance on all arm devices, may consume more
  // memory
  bool use_bf16_storage_ = false;

  // enable options for gpu inference
  bool use_fp16_packed_ = true;
  bool use_fp16_storage_ = true;
  bool use_fp16_arithmetic_ = true;
  bool use_int8_packed_ = true;
  bool use_int8_storage_ = true;
  bool use_int8_arithmetic_ = false;

  // enable simd-friendly packed memory layout
  // improve all operator performance on all arm devices, will consume more
  // memory changes should be applied before loading network structure and
  // weight enabled by default
  bool use_packing_layout_ = true;
  bool use_shader_pack8_ = true;

  // subgroup option
  bool use_subgroup_basic_ = false;
  bool use_subgroup_vote_ = false;
  bool use_subgroup_ballot_ = false;
  bool use_subgroup_shuffle_ = false;

  // turn on for adreno
  bool use_image_storage_ = false;
  bool use_tensor_storage_ = false;

  bool use_reserved_0_ = false;

  // enable DAZ(Denormals-Are-Zero) and FTZ(Flush-To-Zero)
  // default value is 3
  // 0 = DAZ OFF, FTZ OFF
  // 1 = DAZ ON , FTZ OFF
  // 2 = DAZ OFF, FTZ ON
  // 3 = DAZ ON,  FTZ ON
  int flush_denormals_ = 3;

  bool use_local_pool_allocator_ = true;

  // enable local memory optimization for gpu inference
  bool use_shader_local_memory_ = true;

  // enable cooperative matrix optimization for gpu inference
  bool use_cooperative_matrix_ = true;

  // more fine-grained control of winograd convolution
  bool use_winograd23_convolution_ = true;
  bool use_winograd43_convolution_ = true;
  bool use_winograd63_convolution_ = true;

  // this option is turned on for A53/A55 automatically
  // but you can force this on/off if you wish
  bool use_a53_a55_optimized_kernel_ =
      ncnn::is_current_thread_running_on_a53_a55();
};

}  // namespace inference
}  // namespace nndeploy

#endif
