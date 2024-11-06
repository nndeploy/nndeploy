
#include "nndeploy/inference/tnn/tnn_inference_param.h"

#include "nndeploy/device/device.h"

namespace nndeploy {
namespace inference {

static TypeInferenceParamRegister<TypeInferenceParamCreator<TnnInferenceParam>>
    g_tnn_inference_param_register(base::kInferenceTypeTnn);

TnnInferenceParam::TnnInferenceParam() : InferenceParam() {
  model_type_ = base::kModelTypeTnn;
  device_type_ = device::getDefaultHostDeviceType();
  num_thread_ = 4;
}
TnnInferenceParam::~TnnInferenceParam() {}

base::Status TnnInferenceParam::parse(const std::string &json, bool is_path) {
  std::string json_content = "";
  base::Status status = InferenceParam::parse(json_content, false);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "parse json failed!");

  return base::kStatusCodeOk;
}

base::Status TnnInferenceParam::set(const std::string &key,
                                    base::Any &any) {
  base::Status status = base::kStatusCodeOk;
  return base::kStatusCodeOk;
}

base::Status TnnInferenceParam::get(const std::string &key,
                                    base::Any &any) {
  base::Status status = base::kStatusCodeOk;
  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy
