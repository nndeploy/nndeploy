
#ifndef _NNDEPLOY_INFERENCE_ASCEND_CL_INFERENCE_PARAM_H_
#define _NNDEPLOY_INFERENCE_ASCEND_CL_INFERENCE_PARAM_H_

#include "nndeploy/inference/ascend_cl/ascend_cl_include.h"
#include "nndeploy/inference/inference_param.h"

namespace nndeploy {
namespace inference {

class AscendCLInferenceParam : public InferenceParam {
 public:
  AscendCLInferenceParam();
  virtual ~AscendCLInferenceParam();

  AscendCLInferenceParam(const AscendCLInferenceParam &param) = default;
  AscendCLInferenceParam &operator=(const AscendCLInferenceParam &param) =
      default;

  PARAM_COPY(AscendCLInferenceParam)
  PARAM_COPY_TO(AscendCLInferenceParam)

  base::Status parse(const std::string &json, bool is_path = true);

  virtual base::Status set(const std::string &key, base::Value &value);

  virtual base::Status get(const std::string &key, base::Value &value);
};

}  // namespace inference
}  // namespace nndeploy

#endif
