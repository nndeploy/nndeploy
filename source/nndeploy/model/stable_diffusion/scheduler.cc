
#include "nndeploy/model/stable_diffusion/scheduler.h"

namespace nndeploy {
namespace model {

std::vector<float> custom_linspace(float start, float end, int steps) {
  std::vector<float> values(steps);
  float step_size = (end - start) / (steps - 1);
  for (int i = 0; i < steps; ++i) {
    values[i] = start + i * step_size;
  }
  return values;
}

SchedulerDDIM::SchedulerDDIM(const std::string &name,
                             SchedulerType scheduler_type, dag::Edge *input,
                             dag::Edge *output)
    : Scheduler(name, scheduler_type, input, output) {
  param_ = std::make_shared<SchedulerParam>();
}
SchedulerDDIM::SchedulerDDIM(const std::string &name,
                             std::initializer_list<dag::Edge *> inputs,
                             std::initializer_list<dag::Edge *> outputs)
    : Scheduler(name, scheduler_type, inputs, outputs) {
  param_ = std::make_shared<SchedulerParam>();
}
SchedulerDDIM::~SchedulerDDIM() {}

base::Status SchedulerDDIM::setTimesteps() {
  base::Status status = base::kStatusCodeOk;
  float step_ratio =
      (float)param_->num_train_timesteps / (float)param_->num_inference_steps;
  timesteps_.clear();
  timesteps_.resize(param_->num_inference_steps);
  for (int i = 0; i < param_->num_inference_steps; i++) {
    timesteps_[i] =
        (int64_t)(std::lround((float)(param_->num_inference_steps - 1 - i) *
                              step_ratio)) +
        param_->steps_offset_;
  }
  return status;
}

device::Tensor *SchedulerDDIM::scaleModelInput(device::Tensor *sample,
                                               int index) {
  return sample;
}

float SchedulerDDIM::getVariance(int64_t timesteps, int64_t prev_timestep) {
  float alpha_prod_t = alphas_cumprod_[timesteps];
  float alpha_prod_t_prev = (prev_timestep >= 0)
                                ? alphas_cumprod_[prev_timestep]
                                : final_alpha_cumprod_;
  float beta_prod_t = 1.0 - alpha_prod_t;
  float beta_prod_t_prev = 1.0 - alpha_prod_t_prev;

  float variance = (beta_prod_t_prev / beta_prod_t) *
                   (1.0 - alpha_prod_t / alpha_prod_t_prev);
  return variance;
}

base::Status SchedulerDDIM::configure() {
  base::Status status = base::kStatusCodeOk;
  timesteps_.resize(param_->num_inference_steps);
  float step_ratio =
      (float)param_->num_train_timesteps / (float)param_->num_inference_steps;
  for (int i = 0; i < param_->num_inference_steps; i++) {
    int64_t prev_timestep = timesteps_[i] - (int64_t)(step_ratio);
    variance[i] = getVariance(timesteps_[i], prev_timestep);
  }
  return status;
}

base::Status SchedulerDDIM::step(device::Tensor *sample,
                                 device::Tensor *latents, int index,
                                 int timestep);

base::Status SchedulerDDIM::addNoise(device::Tensor *init_latents,
                                     device::Tensor *noise, int idx,
                                     int latent_timestep);

base::Status SchedulerDDIM::init() {
  base::Status status = base::kStatusCodeOk;

  // 计算betas，它们是方差的平方根，从beta_start的平方根到beta_end的平方根
  std::vector<float> betas = custom_linspace(std::sqrtf(param_->beta_start_),
                                             std::sqrtf(param_->beta_end_),
                                             param_->num_train_timesteps_);
  // 计算alphas，它们是1减去beta的平方
  std::vector<float> alphas(param_->num_train_timesteps_, 0.0f);
  for (int i = 0; i < param_->num_train_timesteps_; i++) {
    alphas[i] = 1 - betas[i] * betas[i];
  }
  // alphas_cumprod_
  alphas_cumprod_.resize(param_->num_train_timesteps_, 0.0f);
  alphas_cumprod_[0] = alphas[0];
  for (int i = 1; i < param_->num_train_timesteps_; i++) {
    alphas_cumprod_[i] = alphas_cumprod_[i - 1] * alphas[i];
  }
  // final_alpha_cumprod_
  if (param_->set_alpha_to_one_) {
    final_alpha_cumprod_ = 1.0;
  } else {
    final_alpha_cumprod = alphas_cumprod_.front();
  }
  // timesteps_
  timesteps_.resize(param_->num_train_timesteps_, 0);
  for (int i = 0; i < param_->num_train_timesteps_; i++) {
    timesteps_[i] = param_->num_train_timesteps_ - 1 - i;
  }

  status = this->setTimesteps(param_->denoising_steps_);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "setTimesteps failed!");

  status = this->configure();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "configure failed!");

  status = Loop::init();
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "Loop::init failed!");

  return status;
}
// base::Status SchedulerDDIM::deinit() {}

int SchedulerDDIM::loops() { return param_->denoising_steps_; }

base::Status SchedulerDDIM::run() {
  base::Status status = base::kStatusCodeOk;

  setRunningFlag(true);

  int size = loops();
  if (size < 1) {
    NNDEPLOY_LOGE("loops size is invalid!\n");
    return base::kStatusCodeErrorInvalidValue;
  }
  // build sample
  dag::Edge *sample = this->getEdge("sample");
  latent_tensor_ = sample->createTensor();
  // build timestep
  dag::Edge *timestep = this->getEdge("timestep");
  timestep_tensor_ = sample->createTensor();
  for (int i = 0; i < size; i++) {
    // set sample
    // set timestep
    sample->set(latent_, index);
    device::Tensor *timestep_tensor = new device::Tensor();
    status = executor_->run();
    NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "executor run failed!");
    // update sample
    // update timestep
  }

  setRunningFlag(false);
  return status;
}

dag::Graph *createSchedulerUNetGraphs(const std::string &name, dag::Edge *input,
                                      dag::Edge *output,
                                      SchedulerType scheduler_type,
                                      base::InferenceType inference_type,
                                      std::vector<base::Param *> &param) {
  dag::Scheduler *scheduler =
      createScheduler(name, scheduler_type, input, output);
  scheduler->setParam(param[0]);

  dag::Infer *unet_infer = scheduler->createInfer<Infer>(
      "unet_infer", inference_type,
      {"sample", "timestep", "encoder_hidden_states"}, output);
  unet_infer->setParam(param[1]);

  return scheduler;
}

}  // namespace model
}  // namespace nndeploy
