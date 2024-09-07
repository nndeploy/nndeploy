
#include "nndeploy/inference/rknn/rknn_inference_param.h"

namespace nndeploy {
namespace inference {

static TypeInferenceParamRegister<TypeInferenceParamCreator<RknnInferenceParam>>
    g_rknn_inference_param_register(base::kInferenceTypeRknn);

RknnInferenceParam::RknnInferenceParam() : InferenceParam() {
  model_type_ = base::kModelTypeRknn;
  input_data_format_ = RKNN_TENSOR_NHWC;
  input_data_type_ = RKNN_TENSOR_UINT8;
  input_pass_through_ = false;
  num_thread_ = 4;
  device_type_ = device::getDefaultHostDeviceType();
}
RknnInferenceParam::~RknnInferenceParam() {}

base::Status RknnInferenceParam::parse(const std::string &json, bool is_path) {
  base::Status status = InferenceParam::parse(json, is_path);
  if (status != base::kStatusCodeOk) {
    // TODO: log
    return status;
  }

  return base::kStatusCodeOk;
}

base::Status RknnInferenceParam::set(const std::string &key,
                                     base::Value &value) {
  return base::kStatusCodeOk;
}

base::Status RknnInferenceParam::get(const std::string &key,
                                     base::Value &value) {
  base::Status status = base::kStatusCodeOk;
  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy