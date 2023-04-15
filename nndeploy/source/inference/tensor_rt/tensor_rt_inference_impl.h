#ifndef F5864D0E_2D33_4FA9_B279_2CAC7777F20A
#define F5864D0E_2D33_4FA9_B279_2CAC7777F20A

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
#include "nndeploy/source/inference/tensor_rt/tensor_rt_config.h"
#include "nndeploy/source/inference/tensor_rt/tensor_rt_include.h"
#include "nndeploy/source/inference/tensor_rt/tensor_rt_util.h"

namespace nndeploy {
namespace inference {

class TensorRtInferenceImpl : public AbstractInferenceImpl {
 public:
  TensorRtInferenceImpl();
  virtual ~TensorRtInferenceImpl();

  virtual base::Status init(std::shared_ptr<Config> config);
  virtual base::Status deinit();

  /**
   * @brief preRun
   *
   * @param min_shape
   * @param opt_shape
   * @param max_shape
   * @return base::Status
   * @note
   * # nndeploy的config转为 MNN的config
   * # 通过init函数创建的interpreter和转化后的MNN::ScheduleConfig创建session
   * # 更新所有shape和tensor
   */
  virtual base::Status preRun(base::ShapeMap min_shape = base::ShapeMap(),
                              base::ShapeMap opt_shape = base::ShapeMap(),
                              base::ShapeMap max_shape = base::ShapeMap());
  virtual base::Status postRun();

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
  virtual base::Status asyncRun();

 public:
  base::Status preRunWithOnnxModel(std::string model_buffer,
                                   TensorRtConfigImpl *config);

  base::Status preRunWithTensorRtModel(std::string model_buffer,
                                       TensorRtConfigImpl *config);

  bool checkDynamicShape();

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

#endif /* F5864D0E_2D33_4FA9_B279_2CAC7777F20A */
