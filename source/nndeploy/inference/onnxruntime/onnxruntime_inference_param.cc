
#include "nndeploy/inference/onnxruntime/onnxruntime_inference_param.h"

namespace nndeploy {
namespace inference {

static TypeInferenceParamRegister<
    TypeInferenceParamCreator<OnnxRuntimeInferenceParam>>
    g_onnxruntime_inference_param_register(base::kInferenceTypeOnnxRuntime);

OnnxRuntimeInferenceParam::OnnxRuntimeInferenceParam() : InferenceParam() {
  model_type_ = base::kModelTypeOnnx;
  device_type_ = device::getDefaultHostDeviceType();
}
OnnxRuntimeInferenceParam::~OnnxRuntimeInferenceParam() {}

base::Status OnnxRuntimeInferenceParam::parse(const std::string &json,
                                              bool is_path) {
  base::Status status = InferenceParam::parse(json, is_path);
  if (status != base::kStatusCodeOk) {
    // TODO: log
    return status;
  }

  return base::kStatusCodeOk;
}

base::Status OnnxRuntimeInferenceParam::set(const std::string &key,
                                            base::Value &value) {
  return base::kStatusCodeOk;
}

base::Status OnnxRuntimeInferenceParam::get(const std::string &key,
                                            base::Value &value) {
  base::Status status = base::kStatusCodeOk;
  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy
