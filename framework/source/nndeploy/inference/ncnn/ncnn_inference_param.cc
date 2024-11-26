
#include "nndeploy/inference/ncnn/ncnn_inference_param.h"

#include "nndeploy/device/device.h"

namespace nndeploy {
namespace inference {

static TypeInferenceParamRegister<TypeInferenceParamCreator<NcnnInferenceParam>>
    g_ncnn_inference_param_register(base::kInferenceTypeNcnn);

NcnnInferenceParam::NcnnInferenceParam() : InferenceParam() {
  model_type_ = base::kModelTypeNcnn;
  device_type_ = device::getDefaultHostDeviceType();
  num_thread_ = ncnn::get_physical_big_cpu_count();
}
NcnnInferenceParam::~NcnnInferenceParam() {}

base::Status NcnnInferenceParam::set(const std::string &key, base::Any &any) {
  base::Status status = base::kStatusCodeOk;
  return base::kStatusCodeOk;
}

base::Status NcnnInferenceParam::get(const std::string &key, base::Any &any) {
  base::Status status = base::kStatusCodeOk;
  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy
