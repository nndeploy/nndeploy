
#include "nndeploy/stable_diffusion/scheduler.h"

#include "nndeploy/infer/infer.h"
#include "nndeploy/op/op_binary.h"

namespace nndeploy {
namespace stable_diffusion {

std::map<SchedulerType, std::shared_ptr<SchedulerCreator>>
    &getGlobalSchedulerCreatorMap() {
  static std::once_flag once;
  static std::shared_ptr<
      std::map<SchedulerType, std::shared_ptr<SchedulerCreator>>>
      creators;
  std::call_once(once, []() {
    creators.reset(
        new std::map<SchedulerType, std::shared_ptr<SchedulerCreator>>);
  });
  return *creators;
}

Scheduler *createScheduler(SchedulerType type) {
  Scheduler *temp = nullptr;
  auto &creater_map = getGlobalSchedulerCreatorMap();
  if (creater_map.count(type) > 0) {
    temp = creater_map[type]->createScheduler(type);
  }
  return temp;
}

void customLinspace(float start, float end, int steps,
                    std::vector<float> &values) {
  float step_size = (end - start) / (steps - 1);
  for (int i = 0; i < steps; ++i) {
    values[i] = start + i * step_size;
  }
}

base::Status initializeLatents(std::mt19937 &generator, float init_noise_sigma,
                               device ::Tensor *latents) {
  base::Status status = base::kStatusCodeOk;

  // # Scale the initial noise by the standard deviation required by the
  // scheduler
  status = device::randnTensor(generator, 0.0, 1.0, latents);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "randn failed!");
  // TODO:op::mul
  // status = op::mul(latents, init_noise_sigma, latents);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "mul failed!");

  return status;
}

}  // namespace stable_diffusion
}  // namespace nndeploy
