
#ifndef _NNDEPLOY_SOURCE_INFERENCE_MNN_MNN_INFERENCE_H_
#define _NNDEPLOY_SOURCE_INFERENCE_MNN_MNN_INFERENCE_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/shape.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/base/value.h"
#include "nndeploy/source/device/device.h"
#include "nndeploy/source/device/tensor.h"
#include "nndeploy/source/inference/inference.h"
#include "nndeploy/source/inference/inference_param.h"
#include "nndeploy/source/inference/mnn/mnn_convert.h"
#include "nndeploy/source/inference/mnn/mnn_include.h"
#include "nndeploy/source/inference/mnn/mnn_inference_param.h"

namespace nndeploy {
namespace inference {

class MnnInference : public Inference {
 public:
  MnnInference(base::InferenceType type);
  virtual ~MnnInference();

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

  virtual float getGFLOPs();

  virtual base::Status setInputTensor(
      const std::string &name,
      const std::shared_ptr<device::Tensor> input_tensor);
  //
  virtual std::shared_ptr<device::Tensor> getOutputTensor(
      const std::string &name, std::vector<int32_t> config);

  virtual base::Status run();

  MNN::ScheduleConfig *getInternalInferenceParam();
  MNN::Interpreter *getInternalInterpreter();
  MNN::Session *getInternalSession();

 private:
  MNN::ScheduleConfig *internal_inference_param_;
  MNN::Interpreter *internal_interpreter_ = nullptr;
  MNN::Session *internal_session_ = nullptr;

  MNN::Tensor::DimensionType internal_current_output_type = MNN::Tensor::CAFFE;
  std::map<std::string, MNN::Tensor *> internal_current_output_tensors_;
};

}  // namespace inference
}  // namespace nndeploy

#endif