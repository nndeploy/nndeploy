
#include "nndeploy/model/stable_diffusion/scheduler.h"

#include "nndeploy/model/infer.h"
#include "nndeploy/op/function.h"

namespace nndeploy {
namespace model {

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
  // status = latents->randn(generator);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "randn failed!");
  // status = op::mul(latents, init_noise_sigma, latents);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "mul failed!");

  return status;
}

std::map<SchedulerType, std::shared_ptr<SchedulerCreator>> &
getGlobalSchedulerCreatorMap() {
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

Scheduler *createScheduler(const std::string &name, SchedulerType type,
                           std::vector<dag::Edge *> &input,
                           std::vector<dag::Edge *> &output) {
  Scheduler *temp = nullptr;
  auto &creater_map = getGlobalSchedulerCreatorMap();
  if (creater_map.count(type) > 0) {
    temp = creater_map[type]->createScheduler(name, type, input, output);
  }
  return temp;
}

dag::Graph *createSchedulerUNetGraph(const std::string &name, dag::Edge *input,
                                     dag::Edge *output,
                                     SchedulerType scheduler_type,
                                     base::InferenceType inference_type,
                                     std::vector<base::Param *> &param) {
  std::vector<dag::Edge *> inputs;
  inputs.emplace_back(input);
  std::vector<dag::Edge *> outputs;
  inputs.emplace_back(output);
  Scheduler *scheduler = createScheduler(name, scheduler_type, inputs, outputs);
  scheduler->setParam(param[0]);

  dag::Node *unet_infer = scheduler->createInfer<Infer>(
      "unet_infer", inference_type,
      {"sample", "timestep", "encoder_hidden_states"}, {output});
  unet_infer->setParam(param[1]);
  scheduler->addNode(unet_infer, false);

  return scheduler;
}

}  // namespace model
}  // namespace nndeploy
