
#include "nndeploy/inference/onnxruntime/onnxruntime_inference_param.h"

#include "nndeploy/device/device.h"

namespace nndeploy {
namespace inference {

static TypeInferenceParamRegister<
    TypeInferenceParamCreator<OnnxRuntimeInferenceParam>>
    g_onnxruntime_inference_param_register(base::kInferenceTypeOnnxRuntime);

OnnxRuntimeInferenceParam::OnnxRuntimeInferenceParam() : InferenceParam() {
  model_type_ = base::kModelTypeOnnx;
  device_type_ = device::getDefaultHostDeviceType();
  num_thread_ = 4;
  graph_optimization_level_ = 1;
  inter_op_num_threads_ = -1;
  execution_mode_ = 0;
}
OnnxRuntimeInferenceParam::~OnnxRuntimeInferenceParam() {}

base::Status OnnxRuntimeInferenceParam::set(const std::string &key,
                                            base::Any &any) {
  return base::kStatusCodeOk;
}

base::Status OnnxRuntimeInferenceParam::get(const std::string &key,
                                            base::Any &any) {
  base::Status status = base::kStatusCodeOk;
  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy
