
#ifndef _NNDEPLOY_INFERENCE_MDC_ASCEND_CL_INFERENCE_PARAM_H_
#define _NNDEPLOY_INFERENCE_MDC_ASCEND_CL_INFERENCE_PARAM_H_

#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/ascend_cl/ascend_cl_include.h"

namespace nndeploy {
namespace inference {

class AscendclInferenceParam : public InferenceParam {
 public:
  AscendclInferenceParam();
  virtual ~AscendclInferenceParam();

  AscendclInferenceParam(const AscendclInferenceParam &param) = default;
  AscendclInferenceParam &operator=(const AscendclInferenceParam &param) = default;

  PARAM_COPY(AscendclInferenceParam)
  PARAM_COPY_TO(AscendclInferenceParam)

  base::Status parse(const std::string &json, bool is_path = true);

  virtual base::Status set(const std::string &key, base::Value &value);

  virtual base::Status get(const std::string &key, base::Value &value);
};

}  // namespace inference
}  // namespace nndeploy

#endif
