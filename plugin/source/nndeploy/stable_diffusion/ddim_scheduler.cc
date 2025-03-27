
#include "nndeploy/stable_diffusion/ddim_scheduler.h"

#include <algorithm>  // std::reverse

#include "nndeploy/infer/infer.h"

namespace nndeploy {
namespace stable_diffusion {

TypeSchedulerRegister<TypeSchedulerCreator<DDIMScheduler>>
    g_ddim_scheduler_register(kSchedulerTypeDDIM);

DDIMScheduler::DDIMScheduler(SchedulerType scheduler_type)
    : Scheduler(scheduler_type) {}

DDIMScheduler::~DDIMScheduler() {}

DDIMSchedulerParam *DDIMSchedulerParam::clone() const {
  return new DDIMSchedulerParam(*this);
}

base::Status DDIMScheduler::init(SchedulerParam *param) {
  if (param == nullptr) {
    NNDEPLOY_LOGE("Scheduler param is null!\n");
    return base::kStatusCodeErrorInvalidValue;
  }

  // init scheduler param
  const DDIMSchedulerParam *ddim_param =
      dynamic_cast<DDIMSchedulerParam *>(param);
  scheduler_param_ = ddim_param->clone();
  ddim_scheduler_param_ = dynamic_cast<DDIMSchedulerParam *>(scheduler_param_);

  float beta_start = ddim_scheduler_param_->beta_start_;
  float beta_end = ddim_scheduler_param_->beta_end_;
  int num_train_timesteps = ddim_scheduler_param_->num_train_timesteps_;

  betas_.resize(num_train_timesteps);
  if (ddim_scheduler_param_->beta_schedule_ == "linear") {
    for (int i = 0; i < num_train_timesteps; i++) {
      betas_[i] =
          beta_start + (beta_end - beta_start) * i / (num_train_timesteps - 1);
    }
  } else if (ddim_scheduler_param_->beta_schedule_ == "scaled_linear") {
    float sqrt_beta_start = std::sqrt(beta_start);
    float sqrt_beta_end = std::sqrt(beta_end);
    for (int i = 0; i < num_train_timesteps; i++) {
      float temp = sqrt_beta_start + (sqrt_beta_end - sqrt_beta_start) * i /
                                         (num_train_timesteps - 1);
      betas_[i] = temp * temp;
    }
  } else {
    NNDEPLOY_LOGI("Unsupported beta scheduler:%s\n",
                  ddim_scheduler_param_->beta_schedule_);
    return base::kStatusCodeErrorInvalidValue;
  }

  // 计算 alphas = 1.0 - betas
  alphas_.resize(num_train_timesteps);
  for (int i = 0; i < num_train_timesteps; i++) {
    alphas_[i] = 1.0 - betas_[i];
  }

  // 计算 alphas 的累积乘积
  alphas_cumprod_.resize(num_train_timesteps);
  float cumprod = 1.0;
  for (int i = 0; i < num_train_timesteps; i++) {
    cumprod *= alphas_[i];
    alphas_cumprod_[i] = cumprod;
  }

  // 记录最后一个时间步的 alpha 累积乘积（在这里即 alphas_cumprod 的第一个元素）
  final_alpha_cumprod_ = alphas_cumprod_[0];

  // 初始化时间步数组，降序排列（例如：999, 998, ..., 0）
  timesteps_.resize(num_train_timesteps);
  for (int i = 0; i < num_train_timesteps; i++) {
    timesteps_[i] = num_train_timesteps - 1 - i;
  }

  return base::kStatusCodeOk;
}

base::Status DDIMScheduler::deinit() {
  if (scheduler_param_ != nullptr) {
    delete scheduler_param_;
    scheduler_param_ = nullptr;
  }
  return base::kStatusCodeOk;
}

base::Status DDIMScheduler::setTimesteps() {
  int num_train_timesteps = ddim_scheduler_param_->num_train_timesteps_;
  int num_inference_steps = ddim_scheduler_param_->num_inference_steps_;

  // 整除运算：确定步长
  int step_ratio = num_train_timesteps / num_inference_steps;
  // 生成 [0, num_train_timesteps) 间隔为 step_ratio 的序列
  std::vector<int> ts;
  for (int i = 0; i < num_train_timesteps; i += step_ratio) {
    ts.push_back(i);
  }
  // 反转序列，按降序排列
  std::reverse(ts.begin(), ts.end());
  timesteps_ = ts;

  return base::kStatusCodeOk;
}

base::Status DDIMScheduler::scaleModelInput(device::Tensor *sample,
                                            int timestep) {
  return base::kStatusCodeOk;
}

base::Status DDIMScheduler::step(device::Tensor *sample,
                                 device::Tensor *timestep,
                                 device::Tensor *latents,
                                 device::Tensor *pre_sample) {
  float *sample_ = (float *)(sample->getData());
  size_t sample_size = sample->getSize();
  std::vector<float> sample_v(sample_, sample_ + sample_size);

  float *timestep_ = (float *)(timestep->getData());
  size_t timestep_size = timestep->getSize();
  std::vector<float> timestep_v(timestep_, timestep_ + timestep_size);

  float *latents_ = (float *)(latents->getData());
  size_t latents_size = latents->getSize();
  std::vector<float> latents_v(latents_, latents_ + latents_size);

  float *pre_sample_ = (float *)(pre_sample->getData());
  size_t pre_sample_size = pre_sample->getSize();
  std::vector<float> pre_sample_v(pre_sample_, pre_sample_ + pre_sample_size);

  base::Status status =
      step_inner(sample_v, timestep_v[0], latents_v, pre_sample_v);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("ddim scheduler step failed.\n");
    return status;
  }

  memcpy(pre_sample_, pre_sample_v.data(), pre_sample_size);
  return base::kStatusCodeOk;
}

base::Status DDIMScheduler::step_inner(std::vector<float> &sample, int timestep,
                                       std::vector<float> &latents,
                                       std::vector<float> &prev_sample) {
  int step_index = -1;
  for (size_t i = 0; i < timesteps_.size(); i++) {
    if (timesteps_[i] == timestep) {
      step_index = static_cast<int>(i);
      break;
    }
  }
  if (step_index == -1) {
    NNDEPLOY_LOGE("The timestep is not in timesteps array.");
    return base::kStatusCodeErrorInvalidValue;
  }

  // 根据采样序列确定前一个训练时间步
  float alpha_prod_t_prev;
  if (step_index < static_cast<int>(timesteps_.size()) - 1) {
    int prev_timestep = timesteps_[step_index + 1];
    alpha_prod_t_prev = alphas_cumprod_[prev_timestep];
  } else {
    alpha_prod_t_prev = final_alpha_cumprod_;
  }

  // 当前时间步对应的alpha累积乘积, 注意直接用传入的 timestep 作为下标
  float alpha_prod_t = alphas_cumprod_[timestep];

  // 计算方差
  float variance = (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t) *
                   (1.0 - alpha_prod_t / alpha_prod_t_prev);

  // 计算噪声尺度 sigma_t
  float eta = ddim_scheduler_param_->eta_;
  float sigma_t = eta * std::sqrt(variance);

  // 计算预测的原始样本
  if (sample.size() != latents.size()) {
    NNDEPLOY_LOGE("sample size is not equal to latents size\n");
    return base::kStatusCodeErrorInvalidValue;
  }
  std::vector<float> pred_original_sample(sample.size());
  float sqrt_alpha_prod_t = std::sqrt(alpha_prod_t);
  float sqrt_one_minus_alpha_prod_t = std::sqrt(1.0 - alpha_prod_t);
  for (size_t i = 0; i < sample.size(); ++i) {
    pred_original_sample[i] =
        (sample[i] - sqrt_one_minus_alpha_prod_t * latents[i]) /
        sqrt_alpha_prod_t;
  }

  // 计算方向项
  std::vector<float> pred_sample_direction(sample.size());
  float sqrt_term = std::sqrt(1.0 - alpha_prod_t_prev - (eta * eta) * variance);
  for (size_t i = 0; i < latents.size(); ++i) {
    pred_sample_direction[i] = sqrt_term * latents[i];
  }

  // 根据是否为最后一步生成随机噪声 noise
  std::vector<float> noise(sample.size(), 0.0);
  if (step_index < static_cast<int>(timesteps_.size()) - 1) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0.0, 1.0);
    for (size_t i = 0; i < noise.size(); ++i) {
      noise[i] = d(gen);
    }
  }

  // 计算前一个样本
  // device::Tensor prev_sample(sample->getDevice(), sample->getDesc());
  float sqrt_alpha_prod_t_prev = std::sqrt(alpha_prod_t_prev);
  for (size_t i = 0; i < sample.size(); ++i) {
    prev_sample[i] = sqrt_alpha_prod_t_prev * pred_original_sample[i] +
                     pred_sample_direction[i] + sigma_t * noise[i];
  }

  // 裁剪样本
  if (ddim_scheduler_param_->clip_sample_) {
    for (size_t i = 0; i < prev_sample.size(); ++i) {
      if (prev_sample[i] < -1.0) {
        prev_sample[i] = -1.0;
      } else if (prev_sample[i] > 1.0) {
        prev_sample[i] = 1.0;
      }
    }
  }

  return base::kStatusCodeOk;
}

std::vector<int> &DDIMScheduler::getTimestep() { return timesteps_; }

}  // namespace stable_diffusion
}  // namespace nndeploy
