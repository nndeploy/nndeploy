#ifndef _NNDEPLOY_INFERENCE_RKNN_RKNN_INFERENCE_H
#define _NNDEPLOY_INFERENCE_RKNN_RKNN_INFERENCE_H

#include "nndeploy/base/common.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/any.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/inference/inference.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/rknn/rknn_include.h"
#include "nndeploy/inference/rknn/rknn_inference_param.h"

namespace nndeploy {
namespace inference {

class RknnInference : public Inference {
 public:
  RknnInference(base::InferenceType type);
  virtual ~RknnInference();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status reshape(base::ShapeMap &shape_map);

  virtual base::Status setInputTensor(const std::string &name,
                                      device::Tensor *input_tensor);

  virtual base::Status run();

  virtual device::Tensor *getOutputTensorAfterRun(
      const std::string &name, base::DeviceType device_type, bool is_copy,
      base::DataFormat data_format = base::kDataFormatAuto);

 private:
  rknn_context rknn_ctx_;
  std::vector<rknn_input> rknn_inputs_;
  std::vector<rknn_output> rknn_outputs_;
  std::map<int, std::string> inputs_index_name_;
};

}  // namespace inference
}  // namespace nndeploy
#endif
