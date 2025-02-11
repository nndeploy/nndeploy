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

SnpeInferenceParam::SnpeInferenceParam(base::InferenceType type)
    : InferenceParam(type) {
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

base::Status SnpeInferenceParam::set(const std::string &key, base::Any &any) {
  base::Status status = base::kStatusCodeOk;

  if (key == "runtime") {
    runtime_ = any.get<std::string>();
  } else if (key == "perf_mode") {
    perf_mode_ = any.get<int32_t>();
  } else if (key == "profiling_level") {
    profiling_level_ = any.get<int32_t>();
  } else if (key == "buffer_type") {
    buffer_type_ = any.get<int32_t>();
  } else if (key == "input_names") {
    input_names_ = any.get<std::vector<std::string>>();
  } else if (key == "output_tensor_names") {
    output_tensor_names_ = any.get<std::vector<std::string>>();
  } else if (key == "output_layer_names") {
    output_layer_names_ = any.get<std::vector<std::string>>();
  }

  return base::kStatusCodeOk;
}

base::Status SnpeInferenceParam::get(const std::string &key, base::Any &any) {
  base::Status status = base::kStatusCodeOk;

  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy