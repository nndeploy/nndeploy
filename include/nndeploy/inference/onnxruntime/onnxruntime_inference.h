
#ifndef _NNDEPLOY_INFERENCE_ONNXRUNTIME_ONNXRUNTIME_INFERENCE_H_
#define _NNDEPLOY_INFERENCE_ONNXRUNTIME_ONNXRUNTIME_INFERENCE_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/inference/inference.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/onnxruntime/onnxruntime_include.h"
#include "nndeploy/inference/onnxruntime/onnxruntime_inference_param.h"

namespace nndeploy {
namespace inference {

struct OrtValueInfo {
  std::string name;
  std::vector<int64_t> shape;
  ONNXTensorElementDataType dtype;
};

class OnnxRuntimeInference : public Inference {
 public:
  OnnxRuntimeInference(base::InferenceType type);
  virtual ~OnnxRuntimeInference();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status reshape(base::ShapeMap &shape_map);

  virtual base::Status run();

  virtual device::Tensor *getOutputTensorAfterRun(
      const std::string &name, base::DeviceType device_type, bool is_copy,
      base::DataFormat data_format = base::kDataFormatAuto);

 private:
  bool isDynamic(std::vector<int64_t> &shape);

 private:
  int batch_size_ = 1;

  Ort::Env env_;
  Ort::Session session_{nullptr};
  Ort::SessionOptions session_options_;
  std::shared_ptr<Ort::IoBinding> binding_;

  std::vector<OrtValueInfo> inputs_desc_;
  std::vector<OrtValueInfo> outputs_desc_;

  std::map<std::string, device::Tensor *> max_input_tensors_;
  std::map<std::string, device::Tensor *> max_output_tensors_;

  std::vector<Ort::Value> internal_outputs_;
};

}  // namespace inference
}  // namespace nndeploy

#endif
