
#ifndef _NNDEPLOY_INFERENCE_MNN_MNN_INFERENCE_H_
#define _NNDEPLOY_INFERENCE_MNN_MNN_INFERENCE_H_

#include "nndeploy/base/basic.h"
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
#include "nndeploy/inference/mnn/mnn_convert.h"
#include "nndeploy/inference/mnn/mnn_include.h"
#include "nndeploy/inference/mnn/mnn_inference_param.h"

namespace nndeploy {
namespace inference {

class MnnInference : public Inference {
 public:
  MnnInference(base::InferenceType type);
  virtual ~MnnInference();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status reshape(base::ShapeMap &shape_map);

  virtual int64_t getMemorySize();

  virtual float getGFLOPs();

  virtual bool canOpInputTensor();
  virtual bool canOpOutputTensor();

  virtual device::TensorDesc getInputTensorAlignDesc(const std::string &name);
  virtual device::TensorDesc getOutputTensorAlignDesc(const std::string &name);

  virtual base::Status run();

  MNN::ScheduleConfig *getInternalInferenceParam();
  MNN::Interpreter *getInternalInterpreter();
  MNN::Session *getInternalSession();

 private:
  base::Status allocateInputOutputTensor();
  base::Status deallocateInputOutputTensor();

 private:
  MNN::ScheduleConfig *internal_inference_param_ = nullptr;
  MNN::Interpreter *internal_interpreter_ = nullptr;
  MNN::Session *internal_session_ = nullptr;
};

}  // namespace inference
}  // namespace nndeploy

#endif