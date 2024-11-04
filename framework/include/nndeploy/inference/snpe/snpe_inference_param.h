#ifndef _NNDEPLOY_INFERENCE_SNPE_SNPE_INFERENCE_PARAM_H_
#define _NNDEPLOY_INFERENCE_SNPE_SNPE_INFERENCE_PARAM_H_

#include "nndeploy/device/device.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/snpe/snpe_include.h"

namespace nndeploy {
namespace inference {
class SnpeInferenceParam : public InferenceParam {
 public:
  SnpeInferenceParam();
  virtual ~SnpeInferenceParam();

  SnpeInferenceParam(const SnpeInferenceParam &param) = default;
  SnpeInferenceParam &operator=(const SnpeInferenceParam &param) = default;

  PARAM_COPY(SnpeInferenceParam)
  PARAM_COPY_TO(SnpeInferenceParam)

  base::Status parse(const std::string &json, bool is_path = true);

  virtual base::Status set(const std::string &key, base::Any &any);

  virtual base::Status get(const std::string &key, base::Any &any);

  std::vector<std::string> save_tensors_;
};

}  // namespace inference
}  // namespace nndeploy

#endif