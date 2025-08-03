
#ifndef _NNDEPLOY_INFERENCE_OPENVINO_OPENVINO_INFERENCE_H_
#define _NNDEPLOY_INFERENCE_OPENVINO_OPENVINO_INFERENCE_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/inference/inference.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/openvino/openvino_include.h"
#include "nndeploy/inference/openvino/openvino_inference_param.h"

namespace nndeploy {
namespace inference {

class NNDEPLOY_CC_API OpenVinoInference : public Inference {
 public:
  OpenVinoInference(base::InferenceType type);
  virtual ~OpenVinoInference();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status reshape(base::ShapeMap &shape_map);

  virtual base::Status run();

  virtual device::Tensor *getOutputTensorAfterRun(
      const std::string &name, base::DeviceType device_type, bool is_copy,
      base::DataFormat data_format = base::kDataFormatAuto);

 public:
  static ov::Core core_;

 private:
  bool initialized_ = false;
  std::map<std::string, int> input_index_map_;
  std::map<std::string, int> output_index_map_;
  ov::CompiledModel compiled_model_;
  ov::InferRequest infer_request_;
};

}  // namespace inference
}  // namespace nndeploy

#endif
