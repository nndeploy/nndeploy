
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

PaddleLiteInferenceParam::PaddleLiteInferenceParam(base::InferenceType type)
    : InferenceParam(type) {
  model_type_ = base::kModelTypePaddleLite;
  device_type_ = device::getDefaultHostDeviceType();
  num_thread_ = 4;
}

PaddleLiteInferenceParam::~PaddleLiteInferenceParam() {}

base::Status PaddleLiteInferenceParam::set(const std::string &key,
                                           base::Any &any) {
  return base::kStatusCodeOk;
}

base::Status PaddleLiteInferenceParam::get(const std::string &key,
                                           base::Any &any) {
  base::Status status = base::kStatusCodeOk;
  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy