
#include "nndeploy/source/inference/mnn/mnn_inference_param.h"

namespace nndeploy {
namespace inference {

static TypeInferenceParamRegister<TypeInferenceParamCreator<MnnInferenceParam>>
    g_mnn_inference_param_register(base::kInferenceTypeMnn);

base::Status MnnInferenceParam::parse(const std::string &json, bool is_path) {
  std::string json_content = "";
  base::Status status = InferenceParam::parse(json_content, false);
  if (status != base::kStatusCodeOk) {
    // TODO: log
    return status;
  }

  return base::kStatusCodeOk;
}

base::Status MnnInferenceParam::set(const std::string &key,
                                    base::Value &value) {
  base::Status status = base::kStatusCodeOk;
  if (key == "opencl_library_path_") {
    uint8_t *tmp = nullptr;
    if (value.get(&tmp)) {
      opencl_library_path_ = std::string(reinterpret_cast<char *>(tmp));
    } else {
      status = base::kStatusCodeErrorInvalidParam;
    }
  }
  return base::kStatusCodeOk;
}

base::Status MnnInferenceParam::get(const std::string &key,
                                    base::Value &value) {
  base::Status status = base::kStatusCodeOk;
  return base::kStatusCodeOk;
}

}  // namespace inference
}  // namespace nndeploy
