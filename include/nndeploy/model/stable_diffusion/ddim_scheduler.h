
#ifndef _NNDEPLOY_MODEL_STABLE_DIFFUSION_DDIM_SCHEDULER_H_
#define _NNDEPLOY_MODEL_STABLE_DIFFUSION_DDIM_SCHEDULER_H_

#include <random>

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
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/loop.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/model/convert_to.h"
#include "nndeploy/model/stable_diffusion/scheduler.h"
#include "nndeploy/model/stable_diffusion/type.h"

namespace nndeploy {
namespace model {

class NNDEPLOY_CC_API DDIMScheduler : public Scheduler {
 public:
  DDIMScheduler(SchedulerType scheduler_type);
  virtual ~DDIMScheduler();

  virtual base::Status setTimesteps();

  virtual device::Tensor *scaleModelInput(device::Tensor *sample, int index);

  float getVariance(int64_t timesteps, int64_t prev_timestep);

  virtual base::Status configure();

  virtual base::Status step(device::Tensor *model_output,
                            device::Tensor *sample, int idx,
                            std::vector<int64_t> &timestep, float eta,
                            bool use_clipped_model_output,
                            std::mt19937 &generator,
                            device::Tensor *variance_noise);

  virtual base::Status addNoise(device::Tensor *init_latents,
                                device::Tensor *noise, int idx,
                                int latent_timestep);

 public:
  std::vector<float> alphas_cumprod_;  // alpha的累积乘积
  float final_alpha_cumprod_ = 1.0;
  // standard deviation of the initial noise distribution
  float init_noise_sigma_ = 1.0f;  // 初始噪声的标准差

  std::vector<int64_t> timesteps_;  // 时间步序列
  std::vector<float> variance_;     // 方差
  device::Tensor *noise_pred_uncond_ = nullptr;
  device::Tensor *noise_pred_text_ = nullptr;
};

}  // namespace model
}  // namespace nndeploy

#endif
