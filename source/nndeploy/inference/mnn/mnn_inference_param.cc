
#include "nndeploy/inference/mnn/mnn_inference_param.h"

namespace nndeploy {
namespace inference {

static TypeInferenceParamRegister<TypeInferenceParamCreator<MnnInferenceParam>>
    g_mnn_inference_param_register(base::kInferenceTypeMnn);

MnnInferenceParam::MnnInferenceParam() : InferenceParam() {}
MnnInferenceParam::~MnnInferenceParam() {}

base::Status MnnInferenceParam::parse(const std::string &json, bool is_path) {
  std::string json_content = "";
  base::Status status = InferenceParam::parse(json_content, false);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "parse json failed!");

  return base::kStatusCodeOk;
}

base::Status MnnInferenceParam::set(const std::string &key,
                                    base::Value &value) {
  base::Status status = base::kStatusCodeOk;
  return base::kStatusCodeOk;
}

base::Status MnnInferenceParam::get(const std::string &key,
                                    base::Value &value) {
  base::Status status = base::kStatusCodeOk;
  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy
