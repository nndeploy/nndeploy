
#include "nndeploy/inference/coreml/coreml_inference_param.h"

namespace nndeploy {
namespace inference {

static TypeInferenceParamRegister<TypeInferenceParamCreator<CoremlInferenceParam>>
    g_coreml_inference_param_register(base::kInferenceTypeCoreML);

CoremlInferenceParam::CoremlInferenceParam() : InferenceParam() {
  model_type_ = base::kModelTypeCoreML;
  // device_type_ = device::getDefaultHostDeviceType();
  // num_thread_ = 4;
}

CoremlInferenceParam::~CoremlInferenceParam() {}

base::Status CoremlInferenceParam::set(const std::string &key, base::Any &any) {
  base::Status status = base::kStatusCodeOk;
  return base::kStatusCodeOk;
}

base::Status CoremlInferenceParam::get(const std::string &key, base::Any &any) {
  base::Status status = base::kStatusCodeOk;
  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy
