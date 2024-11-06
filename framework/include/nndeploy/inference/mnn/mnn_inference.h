
#ifndef _NNDEPLOY_INFERENCE_MNN_MNN_INFERENCE_H_
#define _NNDEPLOY_INFERENCE_MNN_MNN_INFERENCE_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/shape.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/any.h"
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
  MNN::ScheduleConfig *schedule_config_ = nullptr;
  MNN::Interpreter *interpreter_ = nullptr;
  MNN::Session *session_ = nullptr;
};

}  // namespace inference
}  // namespace nndeploy

#endif