
#ifndef _NNDEPLOY_MODEL_STABLE_DIFFUSION_DDIM_SCHEDULER_H_
#define _NNDEPLOY_MODEL_STABLE_DIFFUSION_DDIM_SCHEDULER_H_

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
#include "nndeploy/stable_diffusion/scheduler.h"
#include "nndeploy/stable_diffusion/type.h"

namespace nndeploy {
namespace stable_diffusion {

class NNDEPLOY_CC_API DDIMSchedulerParam : public SchedulerParam {
 public:
  virtual DDIMSchedulerParam *clone() const;

 public:
  float beta_start_ = 0.00085;                   // beta起始值
  float beta_end_ = 0.012;                       // beta结束值
  std::string beta_schedule_ = "scaled_linear";  // beta调度方式
  bool set_alpha_to_one_ = false;  // 是否将alpha的累积乘积的最后一个元素设置为1
};

class NNDEPLOY_CC_API DDIMScheduler : public Scheduler {
 public:
  DDIMScheduler(SchedulerType scheduler_type);
  virtual ~DDIMScheduler();

  virtual base::Status init(SchedulerParam *param);
  virtual base::Status deinit();

  virtual base::Status setTimesteps();

  virtual device::Tensor *scaleModelInput(device::Tensor *sample, int index);

  virtual base::Status DDIMScheduler::step(device::Tensor *sample,
                                           device::Tensor *latents,
                                           int timestep);

  virtual std::vector<float> &getTimestep();

 public:
  float final_alpha_cumprod_ = 1.0;

  std::vector<double> betas_;
  std::vector<double> alphas_;
  std::vector<double> alphas_cumprod_;

  std::vector<int> timesteps_;   // 时间步序列
  std::vector<float> variance_;  // 方差

  DDIMSchedulerParam *ddim_scheduler_param_ = nullptr;
};

}  // namespace stable_diffusion
}  // namespace nndeploy

#endif
