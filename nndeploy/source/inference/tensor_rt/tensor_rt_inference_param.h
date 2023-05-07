
#ifndef _NNDEPLOY_SOURCE_INFERENCE_TENSOR_RT_TENSOR_RT_INFERENCE_PARAM_H_
#define _NNDEPLOY_SOURCE_INFERENCE_TENSOR_RT_TENSOR_RT_INFERENCE_PARAM_H_

#include "nndeploy/source/inference/inference_param.h"
#include "nndeploy/source/inference/tensor_rt/tensor_rt_include.h"

namespace nndeploy {
namespace inference {

class TensorRtInferenceParam : public InferenceParam {
 public:
  TensorRtInferenceParam();
  virtual ~TensorRtInferenceParam();

  base::Status parse(const std::string &json, bool is_path = true);

  virtual base::Status set(const std::string &key, base::Value &value);

  virtual base::Status get(const std::string &key, base::Value &value);

  int32_t max_batch_size_ = 1;
  size_t workspace_size_ = 1 << 30;
  bool is_quant_ = false;
  std::string int8_calibration_table_path_ = "";
  std::string model_save_path_ = "";
};

}  // namespace inference
}  // namespace nndeploy

#endif
