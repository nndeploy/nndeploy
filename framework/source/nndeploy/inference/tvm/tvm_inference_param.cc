#include "nndeploy/inference/tvm/tvm_inference_param.h"

namespace nndeploy {
namespace inference {
static TypeInferenceParamRegister<TypeInferenceParamCreator<TvmInferenceParam>>
    g_tvm_inference_param_register(base::kInferenceTypeTvm);

TvmInferenceParam::TvmInferenceParam() : InferenceParam() {
  model_type_ = base::kModelTypeTvm;
  device_type_ = device::getDefaultHostDeviceType();
  num_thread_ = 4;
}
TvmInferenceParam::~TvmInferenceParam() {}

base::Status TvmInferenceParam::set(const std::string &key, base::Any &any) {
  base::Status status = base::kStatusCodeOk;
  return base::kStatusCodeOk;
}

base::Status TvmInferenceParam::get(const std::string &key, base::Any &any) {
  base::Status status = base::kStatusCodeOk;
  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy