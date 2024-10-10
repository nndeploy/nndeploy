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

base::Status TvmInferenceParam::parse(const std::string &json, bool is_path) {
  std::string json_content = "";
  base::Status status = InferenceParam::parse(json_content, false);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "parse json failed!");

  return base::kStatusCodeOk;
}

base::Status TvmInferenceParam::set(const std::string &key,
                                    base::Value &value) {
  base::Status status = base::kStatusCodeOk;
  return base::kStatusCodeOk;
}

base::Status TvmInferenceParam::get(const std::string &key,
                                    base::Value &value) {
  base::Status status = base::kStatusCodeOk;
  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy