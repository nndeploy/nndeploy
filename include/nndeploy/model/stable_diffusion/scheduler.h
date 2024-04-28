
#ifndef _NNDEPLOY_MODEL_STABLE_DIFFUSION_SCHEDULER_H_
#define _NNDEPLOY_MODEL_STABLE_DIFFUSION_SCHEDULER_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/dag/dag::Edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/model/convert_to.h"

namespace nndeploy {
namespace model {

enum SchedulerType : public int {
  kSchedulerDDIM = 0x0000,
  kSchedulerDPM,
  kSchedulerEulerA,
  kSchedulerLMSD,
  kSchedulerPNDM,
  kSchedulerNotSupport,
};

class NNDEPLOY_CC_API SchedulerParam : public base::Param {
 public:
  std::string version_ = "v2.1";
  int num_train_timesteps_ = 1000;  // 训练时间步数
  float beta_start_ = 0.00085;      // beta起始值
  float beta_end_ = 0.012;          // beta结束值
  bool clip_sample_ = false;        // 是否裁剪样本
  bool set_alpha_to_one_ = false;  // 是否将alpha的累积乘积的最后一个元素设置为1
  int steps_offset_ = 1;  // 时间步偏移
  /**
   * @brief
   * v_prediction or epsilon
   */
  std::string prediction_type_ = "v_prediction";  // 预测噪声的方法

  int num_inference_steps = 50;
};

/**
 * @brief
 * @note 主要是如下4个目的：
 * 1- 迭代次数
 * 2- 构建latent
 * 3- timestep
 * 4- convert_to encoder_hidden_states
 */
class NNDEPLOY_CC_API Scheduler : public dag::Loop {
 public:
  Scheduler(const std::string &name, SchedulerType scheduler_type,
            dag::Edge *input, dag::Edge *output)
      : Loop(name, input, output), scheduler_type_(scheduler_type) {}
  Scheduler(const std::string &name, std::initializer_list<dag::Edge *> inputs,
            std::initializer_list<dag::Edge *> outputs)
      : Loop(name, inputs, outputs), scheduler_type_(scheduler_type) {}
  virtual ~Scheduler() {}

  /**
   * @brief Set the Timesteps object
   *
   * @param num_inference_steps
   * @return base::Status
   * @note 设置推断过程中的时间步
   */
  virtual base::Status setTimesteps() = 0;

  /**
   * @brief
   *
   * @return base::Status
   * @note 配置调度器，计算推断步骤中的方差
   */
  virtual base::Status configure() = 0;

  virtual device::Tensor *scaleModelInput(device::Tensor *sample,
                                          int index) = 0;

  /**
   * @brief
   *
   * @param sample
   * @param latents
   * @param index
   * @param timestep
   * @return base::Status
   * @note
   *  生成过程中的每一步都会调用此方法，它根据当前的时间步计算并更新生成的样本。
   */
  virtual base::Status step(device::Tensor *sample, device::Tensor *latents,
                            int index, int timestep) = 0;

  /**
   * @brief
   *
   * @param init_latents 初始的潜在表示
   * @param noise 要添加的噪声
   * @param idx 当前的时间步索引
   * @param latent_timestep 潜在表示的时间步
   * @return base::Status
   * @note 此方法用于将噪声添加到初始潜在表示中，生成扩散过程中的一个步骤
   */
  virtual base::Status addNoise(device::Tensor *init_latents,
                                device::Tensor *noise, int idx,
                                int latent_timestep) = 0;

 protected:
  SchedulerType scheduler_type_ = kSchedulerNotSupport;
};

class NNDEPLOY_CC_API SchedulerDDIM : public Scheduler {
 public:
  SchedulerDDIM(const std::string &name, SchedulerType scheduler_type,
                dag::Edge *input, dag::Edge *output);
  SchedulerDDIM(const std::string &name,
                std::initializer_list<dag::Edge *> inputs,
                std::initializer_list<dag::Edge *> outputs);
  virtual ~SchedulerDDIM() {}

  virtual base::Status setTimesteps(int num_inference_steps);

  virtual device::Tensor *scaleModelInput(device::Tensor *sample, int index);

  float getVariance(int64_t timesteps, int64_t prev_timestep);

  virtual base::Status configure();

  virtual base::Status step(device::Tensor *sample, device::Tensor *latents,
                            int index, int timestep);

  virtual base::Status addNoise(device::Tensor *init_latents,
                                device::Tensor *noise, int idx,
                                int latent_timestep);

  virtual base::Status init();
  // virtual base::Status deinit() {}

  virtual int loops();
  virtual base::Status run();

 private:
  std::vector<float> alphas_cumprod_;
  float final_alpha_cumprod_ = 1.0;
  // standard deviation of the initial noise distribution
  float init_noise_sigma_ = 1.0;

  std::vector<float> variance_;

  std::vector<int64_t> timesteps_;

  device::Tensor *latent_tensor_ = nullptr;
  device::Tensor *timestep_tensor_ = nullptr;
};

Scheduler *createScheduler(const std::string &name,
                           SchedulerType scheduler_type, dag::Edge *input,
                           dag::Edge *output) {
  switch (scheduler_type) {
    case kSchedulerDDIM:
      return new SchedulerDDIM(name, scheduler_type, input, output);
    default:
      return nullptr;
  }
}

extern NNDEPLOY_CC_API dag::Graph *createSchedulerUNetGraph(
    const std::string &name, dag::Edge *input, dag::Edge *output,
    SchedulerType scheduler_type, base::InferenceType inference_type,
    std::vector<base::Param *> &param);

}  // namespace model
}  // namespace nndeploy

#endif
