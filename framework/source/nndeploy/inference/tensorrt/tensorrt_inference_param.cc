
#include "nndeploy/inference/tensorrt/tensorrt_inference_param.h"

namespace nndeploy {
namespace inference {

static TypeInferenceParamRegister<
    TypeInferenceParamCreator<TensorRtInferenceParam>>
    g_tensorrt_inference_param_register(base::kInferenceTypeTensorRt);

TensorRtInferenceParam::TensorRtInferenceParam() : InferenceParam() {
  model_type_ = base::kModelTypeOnnx;
  device_type_.code_ = base::kDeviceTypeCodeCuda;
  device_type_.device_id_ = 0;
  gpu_tune_kernel_ = 1;
}
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
                                         base::Any &any) {
  return base::kStatusCodeOk;
}

base::Status TensorRtInferenceParam::get(const std::string &key,
                                         base::Any &any) {
  base::Status status = base::kStatusCodeOk;
  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy
