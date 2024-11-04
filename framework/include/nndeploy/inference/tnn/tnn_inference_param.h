
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

  // zh
  base::Status parse(const std::string &json, bool is_path = true);
  // zh
  virtual base::Status set(const std::string &key, base::Any &any);
  // zh
  virtual base::Status get(const std::string &key, base::Any &any);

  // zh:
  // 移除InferenceParam中data_format的配置，tnn的data_format_主要是来选择推理的是tensor的data_format_
  base::DataFormat data_format_ = base::kDataFormatAuto;
};

}  // namespace inference
}  // namespace nndeploy

#endif
