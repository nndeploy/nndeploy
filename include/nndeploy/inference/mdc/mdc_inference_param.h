
#ifndef _NNDEPLOY_INFERENCE_MDC_MDC_INFERENCE_PARAM_H_
#define _NNDEPLOY_INFERENCE_MDC_MDC_INFERENCE_PARAM_H_

#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/mdc/mdc_include.h"

namespace nndeploy {
namespace inference {

class MdcInferenceParam : public InferenceParam {
 public:
  MdcInferenceParam();
  virtual ~MdcInferenceParam();

  MdcInferenceParam(const MdcInferenceParam &param) = default;
  MdcInferenceParam &operator=(const MdcInferenceParam &param) = default;

  PARAM_COPY(MdcInferenceParam)
  PARAM_COPY_TO(MdcInferenceParam)

  base::Status parse(const std::string &json, bool is_path = true);

  virtual base::Status set(const std::string &key, base::Value &value);

  virtual base::Status get(const std::string &key, base::Value &value);
};

}  // namespace inference
}  // namespace nndeploy

#endif
