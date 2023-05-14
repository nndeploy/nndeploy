
#ifndef _NNDEPLOY_SOURCE_INFERENCE_MNN_MNN_INFERENCE_H_
#define _NNDEPLOY_SOURCE_INFERENCE_MNN_MNN_INFERENCE_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/shape.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/base/value.h"
#include "nndeploy/source/device/device.h"
#include "nndeploy/source/device/tensor.h"
#include "nndeploy/source/inference/inference.h"
#include "nndeploy/source/inference/inference_param.h"
#include "nndeploy/source/inference/mnn/mnn_convert.h"
#include "nndeploy/source/inference/mnn/mnn_include.h"
#include "nndeploy/source/inference/mnn/mnn_inference_param.h"

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

  virtual device::TensorDesc getInputTensorAlignDesc(const std::string &name);
  virtual device::TensorDesc getOutputTensorAlignDesc(const std::string &name);

  virtual base::Status setInputTensor(const std::string &name,
                                      device::Tensor *input_tensor);
  virtual base::Status setOutputTensor(const std::string &name,
                                       device::Tensor *output_tensor);

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