
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
#include "nndeploy/stable_diffusion/scheduler.h"
#include "nndeploy/stable_diffusion/type.h"

namespace nndeploy {
namespace stable_diffusion {

class NNDEPLOY_CC_API DDIMScheduler : public Scheduler {
 public:
  DDIMScheduler(SchedulerType scheduler_type);
  virtual ~DDIMScheduler();

  virtual base::Status init();
  virtual base::Status deinit();

  virtual base::Status setTimesteps();

  virtual device::Tensor *scaleModelInput(device::Tensor *sample, int index);

  float getVariance(int64_t timesteps, int64_t prev_timestep);

  virtual base::Status configure();

  virtual base::Status step(device::Tensor *output, device::Tensor *sample,
                            int idx, float timestep, float eta = 0,
                            bool use_clipped_model_output = false,
                            std::mt19937 generator = std::mt19937(),
                            device::Tensor *variance_noise = nullptr);

  virtual base::Status addNoise(device::Tensor *init_latents,
                                device::Tensor *noise, int idx,
                                int latent_timestep);

  virtual std::vector<float> &getTimestep();

 public:
  std::vector<float> alphas_cumprod_;  // alpha的累积乘积
  float final_alpha_cumprod_ = 1.0;

  std::vector<float> timesteps_;  // 时间步序列
  std::vector<float> variance_;   // 方差
};

}  // namespace stable_diffusion
}  // namespace nndeploy

#endif
