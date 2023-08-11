
#ifndef _NNDEPLOY_INFERENCE_TENSORRT_TENSORRT_INFERENCE_H_
#define _NNDEPLOY_INFERENCE_TENSORRT_TENSORRT_INFERENCE_H_

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
#include "nndeploy/inference/tensorrt/tensorrt_include.h"
#include "nndeploy/inference/tensorrt/tensorrt_inference_param.h"

namespace nndeploy {
namespace inference {

class TensorRtInference : public Inference {
 public:
  TensorRtInference(base::InferenceType type);
  virtual ~TensorRtInference();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status reshape(base::ShapeMap &shape_map);

  virtual int64_t getMemorySize();
  virtual base::Status setMemory(device::Buffer *buffer);

  virtual base::Status run();

 private:
  base::Status initWithOnnxModel(const std::string &model_buffer,
                                 TensorRtInferenceParam *config);

  base::Status initWithTensorRtModel(const std::string &model_buffer,
                                     TensorRtInferenceParam *config);

  bool checkDynamicShape();

  base::Status CreateExecuteContext();

  char const *getBindingName(int32_t binding_index);
  int getNbBindings();
  bool bindingIsInput(int32_t binding_index);
  nvinfer1::DataType getBindingDataType(int32_t binding_index);
  nvinfer1::TensorFormat getBindingFormat(int32_t binding_index);
  nvinfer1::Dims getBindingDimensions(int32_t binding_index);
  bool setBindingDimensions(int32_t binding_index, nvinfer1::Dims dimensions);

 private:
  base::UniquePtr<nvinfer1::IBuilder> builder_;
  base::UniquePtr<nvinfer1::INetworkDefinition> network_;
  base::UniquePtr<nvonnxparser::IParser> parser_;

  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_;

  std::vector<void *> bindings_;
  std::map<std::string, int> io_name_index_;
  std::map<int, std::string> io_index_name_;

  size_t forward_memory_size_;
  device::Buffer *inner_forward_buffer_;

  std::map<std::string, device::Tensor *> max_input_tensors_;
  std::map<std::string, device::Tensor *> max_output_tensors_;
};

}  // namespace inference
}  // namespace nndeploy

#endif
