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
};

}  // namespace inference
}  // namespace nndeploy
#endif