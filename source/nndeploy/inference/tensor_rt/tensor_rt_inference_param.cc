
#include "nndeploy/inference/tensor_rt/tensor_rt_inference_param.h"

namespace nndeploy {
namespace inference {

static TypeInferenceParamRegister<
    TypeInferenceParamCreator<TensorRtInferenceParam>>
    g_tensor_rt_inference_param_register(base::kInferenceTypeTensorRt);

TensorRtInferenceParam::TensorRtInferenceParam() : InferenceParam() {}
TensorRtInferenceParam::~TensorRtInferenceParam() {}

base::Status TensorRtInferenceParam::parse(const std::string &json,
                                           bool is_path) {
  base::Status status = InferenceParam::parse(json, is_path);
  if (status != base::kStatusCodeOk) {
    // TODO: log
    return status;
  }

  return base::kStatusCodeOk;
}

base::Status TensorRtInferenceParam::set(const std::string &key,
                                         base::Value &value) {
  return base::kStatusCodeOk;
}

base::Status TensorRtInferenceParam::get(const std::string &key,
                                         base::Value &value) {
  base::Status status = base::kStatusCodeOk;
  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy
