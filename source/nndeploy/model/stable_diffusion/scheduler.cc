
#include "nndeploy/model/stable_diffusion/scheduler.h"

#include "nndeploy/model/infer.h"
#include "nndeploy/op/function.h"

namespace nndeploy {
namespace model {

void Scheduler::setParam(SchedulerParam *param) { scheduler_param_ = param; }

void Scheduler::customLinspace(float start, float end, int steps,
                               std::vector<float> &values) {
  float step_size = (end - start) / (steps - 1);
  for (int i = 0; i < steps; ++i) {
    values[i] = start + i * step_size;
  }
}

base::Status Scheduler::initializeLatents(std::mt19937 &generator,
                                          float init_noise_sigma,
                                          device ::Tensor *latents) {
  base::Status status = base::kStatusCodeOk;

  // # Scale the initial noise by the standard deviation required by the
  // scheduler
  // status = latents->randn(generator);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "randn failed!");
  // status = op::mul(latents, init_noise_sigma, latents);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "mul failed!");

  return status;
}

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

}  // namespace model
}  // namespace nndeploy
