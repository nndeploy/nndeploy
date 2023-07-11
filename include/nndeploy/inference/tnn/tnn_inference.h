
#ifndef _NNDEPLOY_INFERENCE_TNN_TNN_INFERENCE_H_
#define _NNDEPLOY_INFERENCE_TNN_TNN_INFERENCE_H_

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
#include "nndeploy/inference/tnn/tnn_convert.h"
#include "nndeploy/inference/tnn/tnn_include.h"
#include "nndeploy/inference/tnn/tnn_inference_param.h"

namespace nndeploy {
namespace inference {

class TnnInference : public Inference {
 public:
  TnnInference(base::InferenceType type);
  virtual ~TnnInference();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status reshape(base::ShapeMap &shape_map);

  virtual int64_t getMemorySize();

  virtual float getGFLOPs();

  virtual device::TensorDesc getInputTensorAlignDesc(const std::string &name);
  virtual device::TensorDesc getOutputTensorAlignDesc(const std::string &name);

  virtual base::Status run();

  TNN::ScheduleConfig *getInternalInferenceParam();
  TNN::Interpreter *getInternalInterpreter();
  TNN::Session *getInternalSession();

 private:
  base::Status allocateInputOutputTensor();
  base::Status deallocateInputOutputTensor();

 private:
  TNN::ScheduleConfig *internal_inference_param_ = nullptr;
  TNN::Interpreter *internal_interpreter_ = nullptr;
  TNN::Session *internal_session_ = nullptr;
};

}  // namespace inference
}  // namespace nndeploy

#endif