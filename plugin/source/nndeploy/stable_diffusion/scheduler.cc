
#include "nndeploy/stable_diffusion/scheduler.h"

#include "nndeploy/infer/infer.h"

namespace nndeploy {
namespace stable_diffusion {

base::Status SchedulerParam::serialize(
    rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
  rapidjson::Value version_val;
  version_val.SetString(version_.c_str(),
                        static_cast<rapidjson::SizeType>(version_.length()),
                        allocator);
  json.AddMember("version_", version_val, allocator);

  json.AddMember("num_train_timesteps_", num_train_timesteps_, allocator);
  json.AddMember("clip_sample_", clip_sample_, allocator);
  json.AddMember("num_inference_steps_", num_inference_steps_, allocator);
  json.AddMember("unet_channels_", unet_channels_, allocator);
  json.AddMember("image_height_", image_height_, allocator);
  json.AddMember("image_width_", image_width_, allocator);
  json.AddMember("guidance_scale_", guidance_scale_, allocator);
  json.AddMember("vae_scale_factor_", vae_scale_factor_, allocator);
  json.AddMember("init_noise_sigma_", init_noise_sigma_, allocator);

  return base::kStatusCodeOk;
}

base::Status SchedulerParam::deserialize(rapidjson::Value &json) {
  if (json.HasMember("version_") && json["version_"].IsString()) {
    version_ = json["version_"].GetString();
  }

  if (json.HasMember("num_train_timesteps_") &&
      json["num_train_timesteps_"].IsInt()) {
    num_train_timesteps_ = json["num_train_timesteps_"].GetInt();
  }

  if (json.HasMember("clip_sample_") && json["clip_sample_"].IsBool()) {
    clip_sample_ = json["clip_sample_"].GetBool();
  }

  if (json.HasMember("num_inference_steps_") &&
      json["num_inference_steps_"].IsInt()) {
    num_inference_steps_ = json["num_inference_steps_"].GetInt();
  }

  if (json.HasMember("unet_channels_") && json["unet_channels_"].IsInt()) {
    unet_channels_ = json["unet_channels_"].GetInt();
  }

  if (json.HasMember("image_height_") && json["image_height_"].IsInt()) {
    image_height_ = json["image_height_"].GetInt();
  }

  if (json.HasMember("image_width_") && json["image_width_"].IsInt()) {
    image_width_ = json["image_width_"].GetInt();
  }

  if (json.HasMember("guidance_scale_") && json["guidance_scale_"].IsFloat()) {
    guidance_scale_ = json["guidance_scale_"].GetFloat();
  }

  if (json.HasMember("vae_scale_factor_") &&
      json["vae_scale_factor_"].IsFloat()) {
    vae_scale_factor_ = json["vae_scale_factor_"].GetFloat();
  }

  if (json.HasMember("init_noise_sigma_") &&
      json["init_noise_sigma_"].IsFloat()) {
    init_noise_sigma_ = json["init_noise_sigma_"].GetFloat();
  }

  return base::kStatusCodeOk;
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

  // Scale the initial noise by the standard deviation required by the scheduler
  status = device::randnTensor(generator, 0.0, 1.0, latents);
  NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "randn failed!");
  // TODO:op::mul
  // status = op::mul(latents, init_noise_sigma, latents);
  // NNDEPLOY_RETURN_ON_NEQ(status, base::kStatusCodeOk, "mul failed!");

  return status;
}

}  // namespace stable_diffusion
}  // namespace nndeploy
