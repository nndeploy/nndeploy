
#ifndef _NNDEPLOY_INFERENCE_COREML_COREML_INFERENCE_PARAM_H_
#define _NNDEPLOY_INFERENCE_COREML_COREML_INFERENCE_PARAM_H_

#include "nndeploy/device/device.h"
#include "nndeploy/inference/coreml/coreml_include.h"
#include "nndeploy/inference/inference_param.h"

namespace nndeploy {
namespace inference {

class CoremlInferenceParam : public InferenceParam {
 public:
  CoremlInferenceParam();
  virtual ~CoremlInferenceParam();

  CoremlInferenceParam(const CoremlInferenceParam &param) = default;
  CoremlInferenceParam &operator=(const CoremlInferenceParam &param) = default;

  PARAM_COPY(CoremlInferenceParam)
  PARAM_COPY_TO(CoremlInferenceParam)

  virtual base::Status parse(const std::string &json, bool is_path = true);
  virtual base::Status set(const std::string &key, base::Value &value);
  virtual base::Status get(const std::string &key, base::Value &value);

  /// @brief A Boolean value that determines whether to allow low-precision
  /// accumulation on a GPU.
  bool low_precision_acceleration_ =
      false;  // reivew： 可否使用InferenceParam::precision_type_
  enum inferenceUnits {  // reivew： 可否使用InferenceParam::device_type_,
                         // 参照openvino的做法，继续沿用该做法的法，需要inferenceUnits->InferenceUnits
    ALL_UNITS = 0,
    CPU_ONLY = 1,
    CPU_AND_GPU = 2,
    CPU_AND_NPU
  };
  inferenceUnits inference_units_ = CPU_ONLY;
};

}  // namespace inference
}  // namespace nndeploy

#endif
