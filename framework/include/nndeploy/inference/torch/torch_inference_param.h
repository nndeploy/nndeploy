
#ifndef _NNDEPLOY_INFERENCE_ONNXRUNTIME_ONNXRUNTIME_INFERENCE_PARAM_H_
#define _NNDEPLOY_INFERENCE_ONNXRUNTIME_ONNXRUNTIME_INFERENCE_PARAM_H_

#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/onnxruntime/onnxruntime_include.h"

namespace nndeploy {
namespace inference {

class OnnxRuntimeInferenceParam : public InferenceParam {
 public:
  OnnxRuntimeInferenceParam();
  virtual ~OnnxRuntimeInferenceParam();

  OnnxRuntimeInferenceParam(const OnnxRuntimeInferenceParam &param) = default;
  OnnxRuntimeInferenceParam &operator=(const OnnxRuntimeInferenceParam &param) =
      default;

  PARAM_COPY(OnnxRuntimeInferenceParam)
  PARAM_COPY_TO(OnnxRuntimeInferenceParam)

  virtual base::Status set(const std::string &key, base::Any &any);

  virtual base::Status get(const std::string &key, base::Any &any);
};

}  // namespace inference
}  // namespace nndeploy

#endif
