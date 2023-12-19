
#include "nndeploy/inference/ascend_cl/ascend_cl_inference_param.h"

#include "nndeploy/device/device.h"

namespace nndeploy {
namespace inference {

static TypeInferenceParamRegister<TypeInferenceParamCreator<AscendclInferenceParam>>
    g_mdc_inference_param_register(base::kInferenceTypeASCENDCL);

AscendclInferenceParam::AscendclInferenceParam() : InferenceParam() {
  model_type_ = base::kModelTypeASCENDCL;
  device_type_.code_ = base::kDeviceTypeCodeASCENDCL;
  device_type_.device_id_ = 0;
  num_thread_ = 4;
}
AscendclInferenceParam::~AscendclInferenceParam() {}

base::Status AscendclInferenceParam::parse(const std::string &json, bool is_path) {
  base::Status status = InferenceParam::parse(json, is_path);
  if (status != base::kStatusCodeOk) {
    // TODO: log
    return status;
  }

  return base::kStatusCodeOk;
}

base::Status AscendclInferenceParam::set(const std::string &key,
                                    base::Value &value) {
  return base::kStatusCodeOk;
}

base::Status AscendclInferenceParam::get(const std::string &key,
                                    base::Value &value) {
  base::Status status = base::kStatusCodeOk;
  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy
