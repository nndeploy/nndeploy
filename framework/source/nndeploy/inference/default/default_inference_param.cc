
#include "nndeploy/inference/default/default_inference_param.h"

#include "nndeploy/device/device.h"

namespace nndeploy {
namespace inference {

static TypeInferenceParamRegister<TypeInferenceParamCreator<DefaultInferenceParam>>
    g_default_inference_param_register(base::kInferenceTypeDefault);

DefaultInferenceParam::DefaultInferenceParam() : InferenceParam() {
  model_type_ = base::kModelTypeDefault;
  device_type_ = device::getDefaultHostDeviceType();
  num_thread_ = 4;
  model_desc_ = nullptr;
  tensor_pool_type_ = net::kTensorPool1DSharedObjectTypeGreedyBySizeImprove;
}
DefaultInferenceParam::~DefaultInferenceParam() {}

base::Status DefaultInferenceParam::parse(const std::string &json, bool is_path) {
  std::string json_content = "";
  base::Status status = InferenceParam::parse(json_content, false);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "parse json failed!");

  return base::kStatusCodeOk;
}

base::Status DefaultInferenceParam::set(const std::string &key,
                                    base::Value &value) {
  base::Status status = base::kStatusCodeOk;
  return base::kStatusCodeOk;
}

base::Status DefaultInferenceParam::get(const std::string &key,
                                    base::Value &value) {
  base::Status status = base::kStatusCodeOk;
  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy
