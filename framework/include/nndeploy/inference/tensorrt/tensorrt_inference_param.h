
#ifndef _NNDEPLOY_INFERENCE_TENSORRT_TENSORRT_INFERENCE_PARAM_H_
#define _NNDEPLOY_INFERENCE_TENSORRT_TENSORRT_INFERENCE_PARAM_H_

#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/tensorrt/tensorrt_include.h"

namespace nndeploy {
namespace inference {

class TensorRtInferenceParam : public InferenceParam {
 public:
  TensorRtInferenceParam();
  virtual ~TensorRtInferenceParam();

  TensorRtInferenceParam(const TensorRtInferenceParam &param) = default;
  TensorRtInferenceParam &operator=(const TensorRtInferenceParam &param) =
      default;

  PARAM_COPY(TensorRtInferenceParam)
  PARAM_COPY_TO(TensorRtInferenceParam)

  virtual base::Status set(const std::string &key, base::Any &any);

  virtual base::Status get(const std::string &key, base::Any &any);

  int max_batch_size_ = 1;
  size_t workspace_size_ = 1 << 30;
  bool is_quant_ = false;
  std::string int8_calibration_table_path_ = "";
  std::string model_save_path_ = "";
};

}  // namespace inference
}  // namespace nndeploy

#endif
