
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
#include "nndeploy/source/inference/mnn/mnn_config.h"
#include "nndeploy/source/inference/mnn/mnn_include.h"

namespace nndeploy {
namespace inference {

class MnnInferenceImpl : public AbstractInferenceImpl {
 public:
  MnnInferenceImpl();
  virtual ~MnnInferenceImpl();

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

  virtual base::Status setInputTensor(
      const std::string &name,
      const std::shared_ptr<device::Tensor> input_tensor);
  //
  virtual std::shared_ptr<device::Tensor> getOutputTensor(
      const std::string &name, std::vector<int32_t> config);

  virtual base::Status run();
  virtual base::Status asyncRun();

  MNN::ScheduleConfig *getInternalConfig();
  MNN::Interpreter *getInternalInterpreter();
  MNN::Session *getInternalSession();

 private:
  MNN::ScheduleConfig *internal_config_ = nullptr;
  MNN::Interpreter *internal_interpreter_ = nullptr;
  MNN::Session *internal_session_ = nullptr;

  MNN::Tensor::DimensionType internal_current_output_type = MNN::Tensor::CAFFE;
  std::map<std::string, MNN::Tensor *> internal_current_output_tensors_;
};

}  // namespace inference
}  // namespace nndeploy

#endif