#ifndef _NNDEPLOY_INFERENCE_RKNN_RKNN_INFERENCE_PARAM_H_
#define _NNDEPLOY_INFERENCE_RKNN_RKNN_INFERENCE_PARAM_H_

#include "nndeploy/device/device.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/rknn/rknn_include.h"

namespace nndeploy {
namespace inference {

class NNDEPLOY_CC_API RknnInferenceParam : public InferenceParam {
 public:
  RknnInferenceParam();
  RknnInferenceParam(base::InferenceType type);
  virtual ~RknnInferenceParam();

  RknnInferenceParam(const RknnInferenceParam &param) = default;
  RknnInferenceParam &operator=(const RknnInferenceParam &param) = default;

  PARAM_COPY(RknnInferenceParam)
  PARAM_COPY_TO(RknnInferenceParam)

  virtual base::Status set(const std::string &key, base::Any &any);

  virtual base::Status get(const std::string &key, base::Any &any);

  rknn_tensor_format input_data_format_;
  rknn_tensor_type input_data_type_;
  bool input_pass_through_;
};

}  // namespace inference
}  // namespace nndeploy
#endif
