
#include "nndeploy/inference/mnn/mnn_inference_param.h"

namespace nndeploy {
namespace inference {

static TypeInferenceParamRegister<TypeInferenceParamCreator<MnnInferenceParam>>
    g_mnn_inference_param_register(base::kInferenceTypeMnn);

MnnInferenceParam::MnnInferenceParam() : InferenceParam() {
  model_type_ = base::kModelTypeMnn;
  device_type_ = device::getDefaultHostDeviceType();
  num_thread_ = 4;
  backup_device_type_ = device::getDefaultHostDeviceType();
}

MnnInferenceParam::MnnInferenceParam(base::InferenceType type)
    : InferenceParam(type) {
  model_type_ = base::kModelTypeMnn;
  device_type_ = device::getDefaultHostDeviceType();
  num_thread_ = 4;
  backup_device_type_ = device::getDefaultHostDeviceType();
}

MnnInferenceParam::~MnnInferenceParam() {}

base::Status MnnInferenceParam::set(const std::string &key, base::Any &any) {
  base::Status status = base::kStatusCodeOk;
  return base::kStatusCodeOk;
}

base::Status MnnInferenceParam::get(const std::string &key, base::Any &any) {
  base::Status status = base::kStatusCodeOk;
  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy
