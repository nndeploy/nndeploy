
#include "nndeploy/inference/tnn/tnn_inference_param.h"

namespace nndeploy {
namespace inference {

static TypeInferenceParamRegister<TypeInferenceParamCreator<TnnInferenceParam>>
    g_tnn_inference_param_register(base::kInferenceTypeTnn);

TnnInferenceParam::TnnInferenceParam() : InferenceParam() {
  model_type_ = base::kModelTypeTnn;
  is_path_ = false;
  device_type_ = base::kDeviceTypeArm;
  num_thread_ = 4;
}
TnnInferenceParam::~TnnInferenceParam() {}

}  // namespace inference
}  // namespace nndeploy
