
#include "nndeploy/stable_diffusion/ddim_scheduler.h"

#include <algorithm>

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

base::Status DDIMSchedulerParam::serialize(
    rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
  base::Status status = SchedulerParam::serialize(json, allocator);
  if (status != base::kStatusCodeOk) return status;

  json.AddMember("beta_start_", beta_start_, allocator);
  json.AddMember("beta_end_", beta_end_, allocator);

  rapidjson::Value beta_schedule_val;
  beta_schedule_val.SetString(
      beta_schedule_.c_str(),
      static_cast<rapidjson::SizeType>(beta_schedule_.length()), allocator);
  json.AddMember("beta_schedule_", beta_schedule_val, allocator);

  json.AddMember("eta_", eta_, allocator);
  json.AddMember("set_alpha_to_one_", set_alpha_to_one_, allocator);

  return base::kStatusCodeOk;
}

base::Status DDIMSchedulerParam::deserialize(rapidjson::Value &json) {
  base::Status status = SchedulerParam::deserialize(json);
  if (status != base::kStatusCodeOk) return status;

  if (json.HasMember("beta_start_") && json["beta_start_"].IsFloat()) {
    beta_start_ = json["beta_start_"].GetFloat();
  }
  if (json.HasMember("beta_end_") && json["beta_end_"].IsFloat()) {
    beta_end_ = json["beta_end_"].GetFloat();
  }
  if (json.HasMember("beta_schedule_") && json["beta_schedule_"].IsString()) {
    beta_schedule_ = json["beta_schedule_"].GetString();
  }
  if (json.HasMember("eta_") && json["eta_"].IsFloat()) {
    eta_ = json["eta_"].GetFloat();
  }
  if (json.HasMember("set_alpha_to_one_") &&
      json["set_alpha_to_one_"].IsBool()) {
    set_alpha_to_one_ = json["set_alpha_to_one_"].GetBool();
  }
  return base::kStatusCodeOk;
}

base::Status DDIMScheduler::init(SchedulerParam *param) {
  if (param == nullptr) {
    NNDEPLOY_LOGE("Scheduler param is null!\n");
    return base::kStatusCodeErrorInvalidValue;
  }

  // init scheduler param
  const DDIMSchedulerParam *ddim_param =
      dynamic_cast<DDIMSchedulerParam *>(param);
  if (!ddim_param || ddim_param->num_train_timesteps_ < 1 ||
      ddim_param->num_inference_steps_ < 1 ||
      ddim_param->num_inference_steps_ > ddim_param->num_train_timesteps_) {
    return base::kStatusCodeErrorInvalidValue;
  }
  scheduler_param_ = ddim_param->clone();
  ddim_scheduler_param_ = dynamic_cast<DDIMSchedulerParam *>(scheduler_param_);

  float beta_start = ddim_scheduler_param_->beta_start_;
  float beta_end = ddim_scheduler_param_->beta_end_;
  int num_train_timesteps = ddim_scheduler_param_->num_train_timesteps_;

  betas_.resize(num_train_timesteps);
  if (ddim_scheduler_param_->beta_schedule_ == "linear") {
    for (int i = 0; i < num_train_timesteps; i++) {
      betas_[i] =
          beta_start + (beta_end - beta_start) * i / std::max(1, num_train_timesteps - 1);
    }
  } else if (ddim_scheduler_param_->beta_schedule_ == "scaled_linear") {
    float sqrt_beta_start = std::sqrt(beta_start);
    float sqrt_beta_end = std::sqrt(beta_end);
    for (int i = 0; i < num_train_timesteps; i++) {
      float temp = sqrt_beta_start + (sqrt_beta_end - sqrt_beta_start) * i /
                                         std::max(1, num_train_timesteps - 1);
      betas_[i] = temp * temp;
    }
  } else {
    NNDEPLOY_LOGI("Unsupported beta scheduler:%s\n",
                  (ddim_scheduler_param_->beta_schedule_).c_str());
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
    ddim_scheduler_param_ = nullptr;
  }
  return base::kStatusCodeOk;
}

base::Status DDIMScheduler::setTimesteps() {
  if (!ddim_scheduler_param_) return base::kStatusCodeErrorInvalidValue;
  int train = ddim_scheduler_param_->num_train_timesteps_;
  int steps = ddim_scheduler_param_->num_inference_steps_;
  if (steps < 1 || steps > train) return base::kStatusCodeErrorInvalidValue;
  int ratio = train / steps;
  timesteps_.resize(steps);
  for (int i = 0; i < steps; ++i) {
    timesteps_[i] = (steps - 1 - i) * ratio;
  }
  return base::kStatusCodeOk;
}

base::Status DDIMScheduler::scaleModelInput(device::Tensor *sample,
                                            int timestep) {
  return base::kStatusCodeOk;
}

base::Status DDIMScheduler::step(device::Tensor *model_output,
                                 device::Tensor *timestep,
                                 device::Tensor *sample,
                                 device::Tensor *pre_sample) {
  for (auto *tensor : {model_output, timestep, sample, pre_sample}) {
    if (!tensor || !tensor->getData() ||
        !device::isHostDeviceType(tensor->getDeviceType()) ||
        tensor->getDataType() != base::dataTypeOf<float>() ||
        tensor->getSize() < sizeof(float) || tensor->getRealSize() < tensor->getSize()) {
      return base::kStatusCodeErrorInvalidValue;
    }
  }
  if (!ddim_scheduler_param_ || model_output->getSize() != sample->getSize() ||
      pre_sample->getSize() != sample->getSize()) return base::kStatusCodeErrorInvalidValue;
  const float step = *static_cast<const float *>(timestep->getData());
  if (!(step >= 0 && step < static_cast<double>(ddim_scheduler_param_->num_train_timesteps_))
      || std::floor(step) != step) return base::kStatusCodeErrorInvalidValue;
  const size_t count = sample->getSize() / sizeof(float);
  auto *values = static_cast<const float *>(sample->getData());
  auto *prediction = static_cast<const float *>(model_output->getData());
  std::vector<float> sample_v(values, values + count);
  std::vector<float> model_output_v(prediction, prediction + count);
  std::vector<float> pre_sample_v(count);
  base::Status status = step_inner(model_output_v, static_cast<int>(step), sample_v, pre_sample_v);
  if (status != base::kStatusCodeOk) {
    NNDEPLOY_LOGE("ddim scheduler step failed.\n");
    return status;
  }

  memcpy(pre_sample->getData(), pre_sample_v.data(), count * sizeof(float));
  return base::kStatusCodeOk;
}

base::Status DDIMScheduler::step_inner(std::vector<float> &model_output,
                                       int timestep, std::vector<float> &sample,
                                       std::vector<float> &prev_sample) {
  if (!ddim_scheduler_param_ || prev_sample.size() != sample.size()) {
    return base::kStatusCodeErrorInvalidValue;
  }
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
  if (sample.size() != model_output.size()) {
    NNDEPLOY_LOGE("sample size is not equal to model_output size\n");
    return base::kStatusCodeErrorInvalidValue;
  }
  std::vector<float> pred_original_sample(sample.size());
  float sqrt_alpha_prod_t = std::sqrt(alpha_prod_t);
  float sqrt_one_minus_alpha_prod_t = std::sqrt(1.0 - alpha_prod_t);
  for (size_t i = 0; i < sample.size(); ++i) {
    pred_original_sample[i] =
        (sample[i] - sqrt_one_minus_alpha_prod_t * model_output[i]) /
        sqrt_alpha_prod_t;
  }

  // 计算方向项
  std::vector<float> pred_sample_direction(sample.size());
  float sqrt_term = std::sqrt(1.0 - alpha_prod_t_prev - (eta * eta) * variance);
  for (size_t i = 0; i < model_output.size(); ++i) {
    pred_sample_direction[i] = sqrt_term * model_output[i];
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

std::vector<int> &DDIMScheduler::getTimesteps() { return timesteps_; }

}  // namespace stable_diffusion
}  // namespace nndeploy
