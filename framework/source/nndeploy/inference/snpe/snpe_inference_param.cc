#include "nndeploy/inference/snpe/snpe_inference_param.h"

namespace nndeploy {
namespace inference {

static TypeInferenceParamRegister<TypeInferenceParamCreator<SnpeInferenceParam>>
    g_snpe_inference_param_register(base::kInferenceTypeSnpe);

SnpeInferenceParam::SnpeInferenceParam() : InferenceParam() {
  model_type_ = base::kModelTypeSnpe;
  device_type_ = device::getDefaultHostDeviceType();
  num_thread_ = 4;

  // snpe_runtime_ = "dsp";
  // snpe_perf_mode_ = 5;
  // snpe_profiling_level_ = 0;
  // snpe_buffer_type_ = 0;
  // input_names_ = {"images"};
  // output_tensor_names_ = {"output0", "output1", "output2"};
  // output_layer_names_  = {"Conv_199", "Conv_200", "Conv_201"};
}

SnpeInferenceParam::~SnpeInferenceParam() {}

base::Status SnpeInferenceParam::set(const std::string &key,
                                     base::Any &any) {
  base::Status status = base::kStatusCodeOk;

  return base::kStatusCodeOk;
}

base::Status SnpeInferenceParam::get(const std::string &key,
                                     base::Any &any) {
  base::Status status = base::kStatusCodeOk;

  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy