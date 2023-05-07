
#ifndef _NNDEPLOY_SOURCE_INFERENCE_TENSOR_RT_TENSOR_RT_INFERENCE_H_
#define _NNDEPLOY_SOURCE_INFERENCE_TENSOR_RT_TENSOR_RT_INFERENCE_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/base/value.h"
#include "nndeploy/source/device/device.h"
#include "nndeploy/source/device/tensor.h"
#include "nndeploy/source/inference/inference.h"
#include "nndeploy/source/inference/inference_param.h"
#include "nndeploy/source/inference/tensor_rt/tensor_rt_include.h"
#include "nndeploy/source/inference/tensor_rt/tensor_rt_inference_param.h"
#include "nndeploy/source/inference/tensor_rt/tensor_rt_util.h"

namespace nndeploy {
namespace inference {

class TensorRtInference : public Inference {
 public:
  TensorRtInference();
  virtual ~TensorRtInference();

  virtual base::Status init();
  virtual base::Status deinit();

  /**
   * @brief reShape
   *
   * @param shape_map
   * @return base::Status
   * @note
   * # 检查shape_map是否合法
   * ## 是否是input
   * ## 大于最小shape，小于最大shape
   * # 检擦是否等于当前shape
   * ## 等于，continue
   * ## 不等于，reshape
   * ### 更新current shape和current input tensor
   */
  virtual base::Status reShape(base::ShapeMap &shape_map);

  virtual int64_t getMemorySize();
  virtual base::Status setMemory(device::Buffer *buffer);

  virtual base::Status setInputTensor(
      const std::string &name,
      const std::shared_ptr<device::Tensor> input_tensor);
  //
  virtual std::shared_ptr<device::Tensor> getOutputTensor(
      const std::string &name, std::vector<int32_t> config);

  virtual base::Status run();

 public:
  base::Status preRunWithOnnxModel(std::string model_buffer,
                                   TensorRtInferenceParam *config);

  base::Status preRunWithTensorRtModel(std::string model_buffer,
                                       TensorRtInferenceParam *config);

  bool checkDynamicShape();

  base::Status CreateExecuteContext();

 private:
  base::UniquePtr<nvinfer1::IBuilder> builder_;
  base::UniquePtr<nvinfer1::INetworkDefinition> network_;
  base::UniquePtr<nvonnxparser::IParser> parser_;

  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_;

  std::vector<void *> bindings_;
  std::map<std::string, int> io_name_index_;

  static TensorRtLogger logger_;

  size_t forward_memory_size_;
  device::Buffer *inner_forward_buffer_;
};

}  // namespace inference
}  // namespace nndeploy

#endif
