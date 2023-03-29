
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
  MnnInferenceImpl() = default;
  virtual ~MnnInferenceImpl();

  virtual base::Status init(const Config &config);
  virtual base::Status deinit();

  virtual base::Status preRun(base::ShapeMap min_shape = base::ShapeMap(),
                              base::ShapeMap opt_shape = base::ShapeMap(),
                              base::ShapeMap max_shape = base::ShapeMap());
  virtual base::Status postRun();

  virtual base::Status reShape(base::ShapeMap &shape_map);

  virtual int64_t getMemorySize();
  virtual int64_t getMemorySize(int index);
  virtual base::Status setMemory(device::Buffer *buffer);

  virtual base::Status setInputTensor(
      const std::string &name,
      const std::shared_ptr<device::Tensor> input_tensor);
  //
  virtual std::shared_ptr<device::Tensor> getOutputTensor(
      const std::string &name, std::vector<int32_t> config);

  virtual base::Status run();
  virtual base::Status asyncRun();

  MNN::ScheduleConfig getMnnConfig();
  MNN::Interpreter *getMnnInterpreter();
  MNN::Session *getMnnSession();

 private:
  base::Status convertConfigInternal();
  base::ShapeMap getInputShapeMapInternal();
  base::Status createInputsOutputsInternal();

 private:
  MNN::ScheduleConfig mnn_config_;
  MNN::Interpreter *mnn_interpreter_ = nullptr;
  MNN::Session *mnn_session_ = nullptr;
};

}  // namespace inference
}  // namespace nndeploy

#endif