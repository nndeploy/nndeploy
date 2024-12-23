
#ifndef _NNDEPLOY_INFERENCE_TNN_TNN_INFERENCE_H_
#define _NNDEPLOY_INFERENCE_TNN_TNN_INFERENCE_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/shape.h"
#include "nndeploy/base/status.h"
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
  virtual base::Status setMemory(device::Buffer *buffer);

  virtual base::Status run();

  virtual device::Tensor *getOutputTensorAfterRun(
      const std::string &name, base::DeviceType device_type, bool is_copy,
      base::DataFormat data_format = base::kDataFormatAuto);

 private:
  base::Status allocateInputOutputTensor();
  base::Status deallocateInputOutputTensor();

 private:
  tnn::ModelConfig model_config_;
  tnn::NetworkConfig network_config_;
  tnn::TNN *tnn_ = nullptr;
  std::shared_ptr<tnn::Instance> instance_;

  std::map<std::string, std::shared_ptr<tnn::Mat>> output_mat_map_;
};

}  // namespace inference
}  // namespace nndeploy

#endif