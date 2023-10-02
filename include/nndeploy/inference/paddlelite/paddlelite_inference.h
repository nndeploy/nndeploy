#ifndef _NNDEPLOY_INFERENCE_PADDLELITE_PADDLELITE_INFERENCE_H_
#define _NNDEPLOY_INFERENCE_PADDLELITE_PADDLELITE_INFERENCE_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/shape.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/inference/inference.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/paddlelite/paddlelite_convert.h"
#include "nndeploy/inference/paddlelite/paddlelite_include.h"
#include "nndeploy/inference/paddlelite/paddlelite_inference_param.h"

namespace nndeploy {
namespace inference {

class PaddleLiteInference : public Inference {
 public:
  PaddleLiteInference(base::InferenceType type);
  virtual ~PaddleLiteInference();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status reshape(base::ShapeMap &shape_map);

  virtual base::Status run();

private:
    paddle::lite_api::MobileConfig config_;
    std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor_;
    size_t power_mode_ = 0;

};

}  // namespace inference
}  // namespace nndeploy


#endif