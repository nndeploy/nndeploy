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

  virtual device::TensorDesc getInputTensorAlignDesc(const std::string &name);
  virtual device::TensorDesc getOutputTensorAlignDesc(const std::string &name);

  virtual base::Status run();

  virtual device::Tensor *getOutputTensorAfterRun(
      const std::string &name, base::DeviceType device_type, bool is_copy,
      base::DataFormat data_format = base::kDataFormatAuto);

 private:
  base::Status allocateInputOutputTensor();
  base::Status deallocateInputOutputTensor();

 private:
  paddle::lite_api::CxxConfig config_;
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor_;
  std::map<std::string, int> io_name_index_;
};

}  // namespace inference
}  // namespace nndeploy

#endif