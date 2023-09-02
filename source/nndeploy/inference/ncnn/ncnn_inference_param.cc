
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

base::Status NcnnInferenceParam::parse(const std::string &json, bool is_path) {
  std::string json_content = "";
  base::Status status = InferenceParam::parse(json_content, false);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "parse json failed!");

  return base::kStatusCodeOk;
}

base::Status NcnnInferenceParam::set(const std::string &key,
                                     base::Value &value) {
  base::Status status = base::kStatusCodeOk;
  return base::kStatusCodeOk;
}

base::Status NcnnInferenceParam::get(const std::string &key,
                                     base::Value &value) {
  base::Status status = base::kStatusCodeOk;
  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy
