#ifndef _NNDEPLOY_INFERENCE_SNPE_SNPE_INFERENCE_PARAM_H_
#define _NNDEPLOY_INFERENCE_SNPE_SNPE_INFERENCE_PARAM_H_

#include "nndeploy/device/device.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/snpe/snpe_include.h"

namespace nndeploy {
namespace inference {
class SnpeInferenceParam : public InferenceParam {
 public:
  // new feature
  std::string runtime_;
  int32_t perf_mode_;
  int32_t profiling_level_;
  int32_t buffer_type_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_tensor_names_;
  std::vector<std::string> output_layer_names_;

 public:
  SnpeInferenceParam();
  SnpeInferenceParam(base::InferenceType type);
  virtual ~SnpeInferenceParam();

  SnpeInferenceParam(const SnpeInferenceParam &param) = default;
  SnpeInferenceParam &operator=(const SnpeInferenceParam &param) = default;

  PARAM_COPY(SnpeInferenceParam)
  PARAM_COPY_TO(SnpeInferenceParam)

  virtual base::Status set(const std::string &key, base::Any &any);

  virtual base::Status get(const std::string &key, base::Any &any);

  std::vector<std::string> save_tensors_;
};

}  // namespace inference
}  // namespace nndeploy

#endif