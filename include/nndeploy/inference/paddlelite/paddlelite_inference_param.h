#ifndef _NNDEPLOY_INFERENCE_PADDLELITE_PADDLELITE_INFERENCE_PARAM_H_
#define _NNDEPLOY_INFERENCE_PADDLELITE_PADDLELITE_INFERENCE_PARAM_H_

#include "nndeploy/device/device.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/paddlelite/paddlelite_include.h"

namespace nndeploy {
namespace inference {

class PaddleLiteInferenceParam : public InferenceParam {
 public:
  PaddleLiteInferenceParam();
  virtual ~PaddleLiteInferenceParam();

  PaddleLiteInferenceParam(const PaddleLiteInferenceParam &param) = default;
  PaddleLiteInferenceParam &operator=(const PaddleLiteInferenceParam &param) =
      default;

  PARAM_COPY(PaddleLiteInferenceParam)
  PARAM_COPY_TO(PaddleLiteInferenceParam)

  base::Status parse(const std::string &json, bool is_path = true);

  virtual base::Status set(const std::string &key, base::Value &value);

  virtual base::Status get(const std::string &key, base::Value &value);

  // // 参考fastdeploy
  // /// kunlunxin_l3_workspace_size
  // int kunlunxin_l3_workspace_size_ = 0xfffc00;
  // /// kunlunxin_locked
  // bool kunlunxin_locked_ = false;
  // /// kunlunxin_autotune
  // bool kunlunxin_autotune_ = true;
  // /// kunlunxin_autotune_file
  // std::string kunlunxin_autotune_file_ = "";
  // /// kunlunxin_precision
  // std::string kunlunxin_precision_ = "int16";
  // /// kunlunxin_adaptive_seqlen
  // bool kunlunxin_adaptive_seqlen_ = false;
  // /// kunlunxin_enable_multi_stream
  // bool kunlunxin_enable_multi_stream_ = false;

  // /// Optimized model dir for CxxConfig
  // std::string optimized_model_dir_ = "";
  // /// nnadapter_subgraph_partition_config_path
  // std::string nnadapter_subgraph_partition_config_path_ = "";
  // /// nnadapter_subgraph_partition_config_buffer
  // std::string nnadapter_subgraph_partition_config_buffer_ = "";
  // /// nnadapter_context_properties
  // std::string nnadapter_context_properties_ = "";
  // /// nnadapter_model_cache_dir
  // std::string nnadapter_model_cache_dir_ = "";
  // /// nnadapter_mixed_precision_quantization_config_path
  // std::string nnadapter_mixed_precision_quantization_config_path_ = "";
  // /// nnadapter_dynamic_shape_info
  // std::map<std::string, std::vector<std::vector<int64_t>>>
  //     nnadapter_dynamic_shape_info_ = {{"", {{0}}}};
  // /// nnadapter_device_names
  // std::vector<std::string> nnadapter_device_names_ = {};
};

}  // namespace inference
}  // namespace nndeploy
#endif