
#ifndef _NNDEPLOY_INFERENCE_OPENVINO_OPENVINO_INFERENCE_PARAM_H_
#define _NNDEPLOY_INFERENCE_OPENVINO_OPENVINO_INFERENCE_PARAM_H_

#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/openvino/openvino_include.h"

namespace nndeploy {
namespace inference {

class OpenVinoInferenceParam : public InferenceParam {
 public:
  OpenVinoInferenceParam();
  virtual ~OpenVinoInferenceParam();

  OpenVinoInferenceParam(const OpenVinoInferenceParam &param) = default;
  OpenVinoInferenceParam &operator=(const OpenVinoInferenceParam &param) =
      default;

  PARAM_COPY(OpenVinoInferenceParam)
  PARAM_COPY_TO(OpenVinoInferenceParam)

  base::Status parse(const std::string &json, bool is_path = true);

  virtual base::Status set(const std::string &key, base::Any &any);

  virtual base::Status get(const std::string &key, base::Any &any);

  std::vector<base::DeviceType> device_types_;
  /// Number of streams while use OpenVINO
  int num_streams_ = 1;
  /// Affinity mode
  std::string affinity_ = "YES";
  /// Performance hint mode
  std::string hint_ = "UNDEFINED";
};

}  // namespace inference
}  // namespace nndeploy

#endif
