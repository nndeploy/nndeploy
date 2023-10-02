
#include "nndeploy/inference/paddlelite/paddlelite_inference_param.h"

namespace nndeploy {
namespace inference {

static TypeInferenceParamRegister<
    TypeInferenceParamCreator<PaddleLiteInferenceParam>>
    g_paddlelite_inference_param_register(base::kInferenceTypePaddleLite);

PaddleLiteInferenceParam::PaddleLiteInferenceParam() : InferenceParam() {
  model_type_ = base::kModelTypePaddleLite;
  device_type_ = device::getDefaultHostDeviceType();
  num_thread_ = 4;
}
PaddleLiteInferenceParam::~PaddleLiteInferenceParam() {}

base::Status PaddleLiteInferenceParam::parse(const std::string &json,
                                           bool is_path) {
  base::Status status = InferenceParam::parse(json, is_path);
  if (status != base::kStatusCodeOk) {
    // TODO: log
    return status;
  }

  return base::kStatusCodeOk;
}

base::Status PaddleLiteInferenceParam::set(const std::string &key,
                                         base::Value &value) {
  return base::kStatusCodeOk;
}

base::Status PaddleLiteInferenceParam::get(const std::string &key,
                                         base::Value &value) {
  base::Status status = base::kStatusCodeOk;
  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy