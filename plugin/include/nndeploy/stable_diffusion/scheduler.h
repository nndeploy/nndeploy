
#ifndef _NNDEPLOY_MODEL_STABLE_DIFFUSION_SCHEDULER_H_
#define _NNDEPLOY_MODEL_STABLE_DIFFUSION_SCHEDULER_H_

#include <random>

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/loop.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/preprocess/convert_to.h"
#include "nndeploy/stable_diffusion/type.h"

namespace nndeploy {
namespace stable_diffusion {

class NNDEPLOY_CC_API SchedulerParam : public base::Param {
 public:
  virtual SchedulerParam *clone() const = 0;

 public:
  std::string version_ = "v1.5";
  int num_train_timesteps_ = 1000;  // 训练时间步数
  bool clip_sample_ = false;        // 是否裁剪样本
  int num_inference_steps_ = 50;    // 推断步数
  int unet_channels_ = 4;           // channel
  int image_height_ = 512;          // height
  int image_width_ = 512;           // width
};

class NNDEPLOY_CC_API Scheduler {
 public:
  Scheduler(SchedulerType type) : scheduler_type_(type) {}
  virtual ~Scheduler() {}

  virtual base::Status init(SchedulerParam *param) = 0;
  virtual base::Status deinit() = 0;

  /**
   * @brief Set the Timesteps object
   *
   * @return base::Status
   * @note 设置推断过程中的时间步
   */
  virtual base::Status setTimesteps() = 0;

  /**
   * @brief
   *
   * @param sample
   * @param index
   * @return device::Tensor*
   */
  virtual base::Status scaleModelInput(device::Tensor *sample, int index) = 0;
  /**
   * @brief
   *
   * @param sample
   * @param timestep
   * @param latents
   * @param pre_sample
   * @return base::Status
   */
  base::Status step(device::Tensor *sample, device::Tensor *timestep,
                    device::Tensor *latents, device::Tensor *pre_sample) = 0;

  /**
   * @brief Get the Timestep object
   *
   * @return std::vector<float>&
   */
  virtual std::vector<int> &getTimestep() = 0;

 protected:
  SchedulerType scheduler_type_ = kSchedulerTypeNotSupport;
  SchedulerParam *scheduler_param_;
};

/**
 * @brief 推理框架的创建类
 *
 */
class SchedulerCreator {
 public:
  virtual ~SchedulerCreator(){};
  virtual Scheduler *createScheduler(SchedulerType type) = 0;
};

/**
 * @brief 推理框架的创建类模板
 *
 * @tparam T
 */
template <typename T>
class TypeSchedulerCreator : public SchedulerCreator {
  virtual Scheduler *createScheduler(SchedulerType type) { return new T(type); }
};

/**
 * @brief Get the Global Scheduler Creator Map object
 *
 * @return std::map<SchedulerType, std::shared_ptr<SchedulerCreator>>&
 */
std::map<SchedulerType, std::shared_ptr<SchedulerCreator>>
    &getGlobalSchedulerCreatorMap();

/**
 * @brief 推理框架的创建类的注册类模板
 *
 * @tparam T
 */
template <typename T>
class TypeSchedulerRegister {
 public:
  explicit TypeSchedulerRegister(SchedulerType type) {
    getGlobalSchedulerCreatorMap()[type] = std::shared_ptr<T>(new T());
  }
};

/**
 * @brief Create a Scheduler object
 *
 * @param type
 * @return Scheduler*
 */
extern NNDEPLOY_CC_API Scheduler *createScheduler(SchedulerType type);

/**
 * @brief
 *
 * @param generator
 * @param init_noise_sigma
 * @param latents
 * @return base::Status
 */
base::Status initializeLatents(std::mt19937 &generator, float init_noise_sigma,
                               device ::Tensor *latents);

}  // namespace stable_diffusion
}  // namespace nndeploy

#endif
