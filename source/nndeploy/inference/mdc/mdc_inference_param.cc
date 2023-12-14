
#include "nndeploy/inference/mdc/mdc_inference_param.h"

#include "nndeploy/device/device.h"

namespace nndeploy {
namespace inference {

static TypeInferenceParamRegister<TypeInferenceParamCreator<MdcInferenceParam>> g_mdc_inference_param_register(
    base::kInferenceTypeMdc);

MdcInferenceParam::MdcInferenceParam() : InferenceParam() {
  model_type_ = base::kModelTypeMdc;
  device_type_ = device::getDefaultHostDeviceType();
  num_thread_ = 4;
}
MdcInferenceParam::~MdcInferenceParam() {}

base::Status MdcInferenceParam::parse(const std::string &json, bool is_path) {
  base::Status status = InferenceParam::parse(json, is_path);
  if (status != base::kStatusCodeOk) {
    // TODO: log
    return status;
  }

  return base::kStatusCodeOk;
}

base::Status MdcInferenceParam::set(const std::string &key, base::Value &value) { return base::kStatusCodeOk; }

base::Status MdcInferenceParam::get(const std::string &key, base::Value &value) {
  base::Status status = base::kStatusCodeOk;
  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy
