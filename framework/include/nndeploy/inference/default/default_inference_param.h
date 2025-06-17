
#ifndef _NNDEPLOY_INFERENCE_DEFAULT_DEFAULT_INFERENCE_PARAM_H_
#define _NNDEPLOY_INFERENCE_DEFAULT_DEFAULT_INFERENCE_PARAM_H_

#include "nndeploy/inference/default/default_include.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/net/tensor_pool.h"

namespace nndeploy {
namespace inference {

class NNDEPLOY_CC_API DefaultInferenceParam : public InferenceParam {
 public:
  DefaultInferenceParam();
  DefaultInferenceParam(base::InferenceType type);
  virtual ~DefaultInferenceParam();

  DefaultInferenceParam(const DefaultInferenceParam &param) = default;
  DefaultInferenceParam &operator=(const DefaultInferenceParam &param) =
      default;

  PARAM_COPY(DefaultInferenceParam)
  PARAM_COPY_TO(DefaultInferenceParam)

  virtual base::Status set(const std::string &key, base::Any &any);
  virtual base::Status get(const std::string &key, base::Any &any);

  ir::ModelDesc *model_desc_ = nullptr;
  net::TensorPoolType tensor_pool_type_ =
      net::kTensorPool1DSharedObjectTypeGreedyBySizeImprove;
  std::vector<base::DeviceType> device_types_;
};

}  // namespace inference
}  // namespace nndeploy

#endif
