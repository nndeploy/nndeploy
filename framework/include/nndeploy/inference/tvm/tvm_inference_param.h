#ifndef _NNDEPLOY_INFERENCE_TVM_TVM_INFERENCE_PARAM_H_
#define _NNDEPLOY_INFERENCE_TVM_TVM_INFERENCE_PARAM_H_

#include "nndeploy/device/device.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/tvm/tvm_include.h"

namespace nndeploy {
namespace inference {

class NNDEPLOY_CC_API TvmInferenceParam : public InferenceParam {
 public:
  TvmInferenceParam();
  TvmInferenceParam(base::InferenceType type);
  virtual ~TvmInferenceParam();

  TvmInferenceParam(const TvmInferenceParam &param) = default;
  TvmInferenceParam &operator=(const TvmInferenceParam &param) = default;

  PARAM_COPY(TvmInferenceParam)
  PARAM_COPY_TO(TvmInferenceParam)

  virtual base::Status set(const std::string &key, base::Any &any);

  virtual base::Status get(const std::string &key, base::Any &any);
};

}  // namespace inference
}  // namespace nndeploy

#endif