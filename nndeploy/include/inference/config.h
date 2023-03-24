/**
 * @file param.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-12-20
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef _NNDEPLOY_INCLUDE_INFERENCE_CONFIG_H_
#define _NNDEPLOY_INCLUDE_INFERENCE_CONFIG_H_

#include "nndeploy/include/inference/default_config_impl.h"

using namespace nndeploy::base;

namespace nndeploy {
namespace inference {

base::Status JsonToDefaultConfig(const std::string &json,
                                 DefaultConfigImpl *config,
                                 bool is_path = true);

class DefaultConfigImpl {
 public:
  DefaultConfigImpl();
  DefaultConfigImpl(const std::string &json, bool is_path = true);

  virtual ~Config();

  virtual base::Status set(const std::string &key, const base::Value &value);

  virtual base::Status get(const std::string &key, base::Value &value);

  base::InferenceType model_type_;
  base::InferenceType inference_type_;
  bool is_path_ = true;
  bool is_encrypt_ = false;
  std::vector<std::string> model_value_;

  base::DeviceType device_types_;
  base::ShareMemoryType share_memory_mode_ = base::kShareMemoryTypeNoShare;
  base::PrecisionType precision_ = base::PrecisionTypeFp32;
  base::PowerType power_type = base::kPowerTypeNormal;

  bool is_dynamic_shape_ = false;
  base::ShapeMap input_shape_ = base::ShapeMap();
  bool is_quant = false;

  base::InferenceOptLevel opt_level_ = base::kInferenceOptLevelAuto;

  std::string cache_path_ = "";
  bool is_tune_kernel_ = false;
};

class Config {
 public:
  Config(base::InferenceType type);
  Config(base::InferenceType type, const std::string &json,
         bool is_path = true);

  virtual ~Config();

  base::Status set(const std::string &key, const base::Value &value);

  base::Status get(const std::string &key, base::Value &value);

  DefaultConfigImpl *config_impl_;

  base::InferenceType type_;
};

}  // namespace inference
}  // namespace nndeploy

#endif
