
#include "nndeploy/inference/ascend_cl/ascend_cl_inference_param.h"

#include "nndeploy/device/device.h"

namespace nndeploy {
namespace inference {

static TypeInferenceParamRegister<
    TypeInferenceParamCreator<AscendCLInferenceParam>>
    g_ascend_cl_inference_param_register(base::kInferenceTypeAscendCL);

AscendCLInferenceParam::AscendCLInferenceParam() : InferenceParam() {
  model_type_ = base::kModelTypeAscendCL;
  device_type_.code_ = base::kDeviceTypeCodeAscendCL;
  device_type_.device_id_ = 0;
  num_thread_ = 4;
}
AscendCLInferenceParam::~AscendCLInferenceParam() {}

base::Status AscendCLInferenceParam::parse(const std::string &json,
                                           bool is_path) {
  base::Status status = InferenceParam::parse(json, is_path);
  if (status != base::kStatusCodeOk) {
    // TODO: log
    return status;
  }

  return base::kStatusCodeOk;
}

base::Status AscendCLInferenceParam::set(const std::string &key,
                                         base::Any &any) {
  return base::kStatusCodeOk;
}

base::Status AscendCLInferenceParam::get(const std::string &key,
                                         base::Any &any) {
  base::Status status = base::kStatusCodeOk;
  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy
