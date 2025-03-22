
#include "nndeploy/stable_diffusion/ddim_scheduler.h"

#include "nndeploy/infer/infer.h"
#include "nndeploy/op/op_binary.h"
#include "nndeploy/stable_diffusion/scheduler.h"

namespace nndeploy {
namespace stable_diffusion {

TypeSchedulerRegister<TypeSchedulerCreator<DDIMScheduler>>
    g_ddim_scheduler_register(kSchedulerTypeDDIM);

DDIMScheduler::DDIMScheduler(SchedulerType scheduler_type)
    : Scheduler(scheduler_type) {}

DDIMScheduler::~DDIMScheduler() {}

base::Status DDIMScheduler::init(SchedulerParam *param) {
  base::Status status = base::kStatusCodeOk;

  if (param == nullptr) {
    NNDEPLOY_LOGE("Scheduler param is null!\n");
    return base::kStatusCodeErrorInvalidValue;
  }

  // init scheduler param
  scheduler_param_ = dynamic_cast<SchedulerParam *>(param);

  double beta_start = scheduler_param_->beta_start_;
  double beta_end = scheduler_param_->beta_end_;
  int num_train_timesteps = scheduler_param_->num_train_timesteps_;

  betas_.resize(num_train_timesteps);
  if (scheduler_param_->beta_schedule_ == "linear") {
    for (int i = 0; i < num_train_timesteps; i++) {
      betas_[i] =
          beta_start + (beta_end - beta_start) * i / (num_train_timesteps - 1);
    }
  } else if (scheduler_param_->beta_schedule_ == "scaled_linear") {
    double sqrt_beta_start = std::sqrt(beta_start);
    double sqrt_beta_end = std::sqrt(beta_end);
    for (int i = 0; i < num_train_timesteps; i++) {
      double temp = sqrt_beta_start + (sqrt_beta_end - sqrt_beta_start) * i /
                                          (num_train_timesteps - 1);
      betas_[i] = temp * temp;
    }
  } else {
    throw std::invalid_argument("不支持的beta调度: " + beta_schedule);
  }

  // 计算 alphas = 1.0 - betas
  alphas_.resize(num_train_timesteps);
  for (int i = 0; i < num_train_timesteps; i++) {
    alphas_[i] = 1.0 - betas_[i];
  }

  // 计算 alphas 的累积乘积
  alphas_cumprod_.resize(num_train_timesteps);
  double cumprod = 1.0;
  for (int i = 0; i < num_train_timesteps; i++) {
    cumprod *= alphas_[i];
    alphas_cumprod_[i] = cumprod;
  }

  // 记录最后一个时间步的 alpha 累积乘积（在这里即 alphas_cumprod 的第一个元素）
  final_alpha_cumprod = alphas_cumprod_[0];

  // 初始化时间步数组，降序排列（例如：999, 998, ..., 0）
  timesteps_.resize(num_train_timesteps);
  for (int i = 0; i < num_train_timesteps; i++) {
    timesteps_[i] = num_train_timesteps - 1 - i;
  }

  return status;
}

base::Status DDIMScheduler::deinit() {
  base::Status status = base::kStatusCodeOk;
  return status;
}

base::Status DDIMScheduler::setTimesteps() {
  base::Status status = base::kStatusCodeOk;
  int step_ratio = scheduler_param_->num_train_timesteps_ /
                   scheduler_param_->num_inference_steps_;
  timesteps_.clear();
  // creates integer timesteps by multiplying by ratio
  // casting to int to avoid issues when num_inference_step is power of 3
  timesteps_.resize(scheduler_param_->num_inference_steps_);
  for (int i = 0; i < scheduler_param_->num_inference_steps_; i++) {
    timesteps_[i] = (int64_t)((scheduler_param_->num_inference_steps_ - 1 - i) *
                                  step_ratio +
                              scheduler_param_->steps_offset_);
  }
  return status;
}

device::Tensor *DDIMScheduler::scaleModelInput(device::Tensor *sample,
                                               int index) {
  return sample;
}

float DDIMScheduler::getVariance(int64_t timesteps, int64_t prev_timestep) {
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

base::Status DDIMScheduler::configure() {
  base::Status status = base::kStatusCodeOk;
  variance_.resize(scheduler_param_->num_inference_steps_, 0.0f);
  int64_t step_ratio = (int64_t)scheduler_param_->num_train_timesteps_ /
                       (int64_t)scheduler_param_->num_inference_steps_;
  for (int i = 0; i < scheduler_param_->num_inference_steps_; i++) {
    int64_t prev_timestep = timesteps_[i] - step_ratio;
    variance_[i] = getVariance(timesteps_[i], prev_timestep);
  }
  return status;
}
/**
 * @brief
 *
 * @param sample
 * @param latents
 * @param index
 * @param timestep
 * @return base::Status
 * # See formulas (12) and (16) of DDIM paper
 * # https://arxiv.org/pdf/2010.02502.pdf
 * # Ideally, read DDIM paper in-detail understanding
 * # Notation (<variable name> -> <name in paper>
 * # - pred_noise_t -> e_theta(x_t, t)
 * # - pred_original_sample -> f_theta(x_t, t) or x_0
 * # - std_dev_t -> sigma_t
 * # - eta -> η
 * # - pred_sample_direction -> "direction pointing to x_t"
 * # - pred_prev_sample -> "x_t-1"
 */
base::Status DDIMScheduler::step(
    device::Tensor *output, device::Tensor *sample, int idx, float timestep,
    float eta, bool use_clipped_model_output, std::mt19937 generator,
    device::Tensor *variance_noise) {  // discussion
  base::Status status = base::kStatusCodeOk;
  device::Device *host_device = device::getDefaultHostDevice();

  //
  int prev_idx = idx + 1;

  float alpha_prod_t = alphas_cumprod_[idx];
  device::TensorDesc alpha_prod_t_desc(base::dataTypeOf<float>(),
                                       base::kDataFormatN, {1});
  device::Tensor alpha_prod_t_tensor(host_device, alpha_prod_t_desc);
  alpha_prod_t_tensor.set(alpha_prod_t);
  float alpha_prod_t_sqrt = std::sqrt(alpha_prod_t);
  device::Tensor alpha_prod_t_sqrt_tensor(host_device, alpha_prod_t_desc);
  alpha_prod_t_sqrt_tensor.set(alpha_prod_t_sqrt);

  float alpha_prod_t_prev = (prev_idx < scheduler_param_->num_train_timesteps_)
                                ? alphas_cumprod_[prev_idx]
                                : final_alpha_cumprod_;
  device::TensorDesc alpha_prod_t_prev_desc(base::dataTypeOf<float>(),
                                            base::kDataFormatN, {1});
  device::Tensor alpha_prod_t_prev_tensor(host_device, alpha_prod_t_prev_desc);
  alpha_prod_t_prev_tensor.set(alpha_prod_t_prev);

  float beta_prod_t = 1.0 - alpha_prod_t;
  device::TensorDesc beta_prod_t_desc(base::dataTypeOf<float>(),
                                      base::kDataFormatN, {1});
  device::Tensor beta_prod_t_tensor(host_device, beta_prod_t_desc);
  beta_prod_t_tensor.set(beta_prod_t);
  float beta_prod_t_sqrt = std::sqrt(beta_prod_t);
  device::Tensor beta_prod_t_sqrt_tensor(host_device, beta_prod_t_desc);
  beta_prod_t_sqrt_tensor.set(beta_prod_t_sqrt);

  // 3. compute predicted original sample from predicted noise also called
  // "predicted x_0" of formula(12) from https: //arxiv.org/pdf/2010.02502.pdf
  device::Tensor pred_original_sample(output->getDevice(), output->getDesc());
  if (scheduler_param_->prediction_type_ == "epsilon") {
    device::Tensor tmp1(output->getDevice(), output->getDesc());
    op::mul(&beta_prod_t_sqrt_tensor, output, &tmp1);
    device::Tensor tmp2(output->getDevice(), output->getDesc());
    op::sub(sample, &tmp1, &tmp2);
    op::div(&tmp2, &tmp1, &alpha_prod_t_sqrt_tensor);
  } else if (scheduler_param_->prediction_type_ == "sample") {
    output->copyTo(&pred_original_sample);
  } else if (scheduler_param_->prediction_type_ == "v_prediction") {
    device::Tensor sample_tmp(sample->getDevice(), sample->getDesc());
    op::mul(&alpha_prod_t_sqrt_tensor, sample, &sample_tmp);
    device::Tensor model_output_tmp(output->getDevice(), output->getDesc());
    op::mul(&beta_prod_t_sqrt_tensor, output, &model_output_tmp);
    op::sub(&sample_tmp, &model_output_tmp, &pred_original_sample);

    op::mul(&beta_prod_t_sqrt_tensor, sample, &sample_tmp);
    op::mul(&alpha_prod_t_sqrt_tensor, output, &model_output_tmp);
    op::add(&sample_tmp, &model_output_tmp, output);
  } else {
    NNDEPLOY_LOGE("Invalid prediction type!\n");
    return base::kStatusCodeErrorInvalidValue;
  }

  // # 4. Clip "predicted x_0"
  if (scheduler_param_->clip_sample_) {
    op::clamp(&pred_original_sample, -1.0f, 1.0f, &pred_original_sample);
  }

  // # 5. compute variance : "sigma_t(η)"->see formula(16)
  // σ_t = sqrt((1 − α_t−1) / (1 − α_t)) * sqrt(1 − α_t / α_t−1)
  float variance = variance_[idx];
  float std_dev_t = eta * sqrtf(variance);

  if (use_clipped_model_output) {
    // the output is always re-derived from the clipped x_0 in Glide
    device::Tensor pred_original_sample_tmp(pred_original_sample.getDevice(),
                                            pred_original_sample.getDesc());
    op::mul(&alpha_prod_t_sqrt_tensor, &pred_original_sample,
            &pred_original_sample_tmp);
    device::Tensor sample_tmp(sample->getDevice(), sample->getDesc());
    op::sub(sample, &pred_original_sample_tmp, &sample_tmp);
    op::div(&sample_tmp, &beta_prod_t_sqrt_tensor, output);
  }

  // # 6. compute "direction pointing to x_t" of formula (12) from
  // https://arxiv.org/pdf/2010.02502.pdf
  float coff_model_output =
      std::sqrt(1 - alpha_prod_t_prev - std_dev_t * std_dev_t);
  device::Tensor coff_model_output_tensor(host_device, beta_prod_t_desc);
  coff_model_output_tensor.set(coff_model_output);
  device::Tensor pred_sample_direction(output->getDevice(), output->getDesc());
  op::mul(&coff_model_output_tensor, output, &pred_sample_direction);

  // # 7. compute x_t without "random noise" of formula(12) from https:
  // //arxiv.org/pdf/2010.02502.pdf
  float alpha_prod_t_prev_sqrt = std::sqrt(alpha_prod_t_prev);
  device::Tensor pred_original_sample_temp;
  // pred_original_sample_temp = alpha_prod_t_prev_sqrt * pred_original_sample;
  // op::mul
  device::Tensor prev_sample;
  // prev_sample = pred_original_sample_temp + pred_sample_direction;
  // op::add

  return status;
}

base::Status DDIMScheduler::addNoise(device::Tensor *init_latents,
                                     device::Tensor *noise, int idx,
                                     int latent_timestep) {
  base::Status status = base::kStatusCodeOk;
  device::Device *host_device = device::getDefaultHostDevice();

  float sqrt_alpha_prod = std::sqrt(alphas_cumprod_[idx]);
  device::TensorDesc sqrt_alpha_prod_desc(base::dataTypeOf<float>(),
                                          base::kDataFormatN, {1});
  device::Tensor sqrt_alpha_prod_tensor(host_device, sqrt_alpha_prod_desc);
  sqrt_alpha_prod_tensor.set(sqrt_alpha_prod);
  op::mul(&sqrt_alpha_prod_tensor, init_latents, init_latents);

  float sqrt_one_minus_alpha_prod = std::sqrt(1.0f - alphas_cumprod_[idx]);
  device::TensorDesc sqrt_one_minus_alpha_prod_desc(base::dataTypeOf<float>(),
                                                    base::kDataFormatN, {1});
  device::Tensor sqrt_one_minus_alpha_prod_tensor(host_device,
                                                  sqrt_alpha_prod_desc);
  sqrt_one_minus_alpha_prod_tensor.set(sqrt_one_minus_alpha_prod);
  op::mul(&sqrt_one_minus_alpha_prod_tensor, noise, noise);

  op::add(init_latents, noise, init_latents);

  return status;
}

std::vector<float> &DDIMScheduler::getTimestep() { return timesteps_; }

}  // namespace stable_diffusion
}  // namespace nndeploy
