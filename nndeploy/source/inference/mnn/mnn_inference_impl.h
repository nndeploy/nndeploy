
#ifndef _NNDEPLOY_SOURCE_INFERENCE_MNN_MNN_INFERENCE_IMPL_H_
#define _NNDEPLOY_SOURCE_INFERENCE_MNN_MNN_INFERENCE_IMPL_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/base/value.h"
#include "nndeploy/source/device/device.h"
#include "nndeploy/source/device/tensor.h"
#include "nndeploy/source/inference/abstract_inference_impl.h"
#include "nndeploy/source/inference/config.h"
#include "nndeploy/source/inference/mnn/mnn_include.h"

namespace nndeploy {
namespace inference {

class MnnInferenceImpl : public AbstractInferenceImpl {
 public:
  MnnInferenceImpl();
  virtual ~MnnInferenceImpl();

  virtual base::Status init(std::shared_ptr<Config> config);
  virtual base::Status deinit();

  virtual base::Status preRun(base::ShapeMap min_shape = base::ShapeMap(),
                              base::ShapeMap opt_shape = base::ShapeMap(),
                              base::ShapeMap max_shape = base::ShapeMap());
  virtual base::Status postRun();

  virtual base::Status reShape(base::ShapeMap &shape_map);

  virtual int64_t getMemorySize();

  virtual base::Status setInputTensor(
      const std::string &name,
      const std::shared_ptr<device::Tensor> input_tensor);
  //
  virtual std::shared_ptr<device::Tensor> getOutputTensor(
      const std::string &name, std::vector<int32_t> config);

  virtual base::Status run();
  virtual base::Status asyncRun();

  MNN::ScheduleConfig *getInternalConfig();
  MNN::Interpreter *getInternalInterpreter();
  MNN::Session *getInternalSession();

 private:
  base::Status convertInternalConfig();
  base::ShapeMap getInternalInputShapeMap();
  base::Status createInternalInputsOutputs();

 private:
  MNN::ScheduleConfig *internal_config_;
  MNN::Interpreter *internal_interpreter_ = nullptr;
  MNN::Session *internal_session_ = nullptr;
};

}  // namespace inference
}  // namespace nndeploy

#endif