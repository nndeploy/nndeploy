/**
 * @brief
 * 基类InferenceParam已经定义了推理的基本参数，这里只需要定义tnn的特有参数，推理的基本参数已经可以包含tnn的所有参数了，故这里不需要再额外增加参数了
 *
 */
#ifndef _NNDEPLOY_INFERENCE_TNN_TNN_INFERENCE_PARAM_H_
#define _NNDEPLOY_INFERENCE_TNN_TNN_INFERENCE_PARAM_H_

#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/tnn/tnn_include.h"

namespace nndeploy {
namespace inference {

class TnnInferenceParam : public InferenceParam {
 public:
  TnnInferenceParam();
  virtual ~TnnInferenceParam();

  TnnInferenceParam(const TnnInferenceParam &param) = default;
  TnnInferenceParam &operator=(const TnnInferenceParam &param) = default;

  PARAM_COPY(TnnInferenceParam)
  PARAM_COPY_TO(TnnInferenceParam)
};

}  // namespace inference
}  // namespace nndeploy

#endif
