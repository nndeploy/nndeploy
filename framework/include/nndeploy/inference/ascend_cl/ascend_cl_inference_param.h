
#ifndef _NNDEPLOY_INFERENCE_ASCEND_CL_INFERENCE_PARAM_H_
#define _NNDEPLOY_INFERENCE_ASCEND_CL_INFERENCE_PARAM_H_

#include "nndeploy/inference/ascend_cl/ascend_cl_include.h"
#include "nndeploy/inference/inference_param.h"

namespace nndeploy {
namespace inference {

class AscendCLInferenceParam : public InferenceParam {
 public:
  AscendCLInferenceParam();
  AscendCLInferenceParam(base::InferenceType type);
  virtual ~AscendCLInferenceParam();

  AscendCLInferenceParam(const AscendCLInferenceParam &param) = default;
  AscendCLInferenceParam &operator=(const AscendCLInferenceParam &param) =
      default;

  PARAM_COPY(AscendCLInferenceParam)
  PARAM_COPY_TO(AscendCLInferenceParam)

  virtual base::Status set(const std::string &key, base::Any &any);

  virtual base::Status get(const std::string &key, base::Any &any);
};

}  // namespace inference
}  // namespace nndeploy

#endif
