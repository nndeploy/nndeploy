
#ifndef _NNDEPLOY_INCLUDE_INFERENCE_CONFIG_H_
#define _NNDEPLOY_INCLUDE_INFERENCE_CONFIG_H_

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/base/value.h"

namespace nndeploy {
namespace inference {

class DefaultConfigImpl {
 public:
  DefaultConfigImpl();
  virtual ~DefaultConfigImpl();

  virtual base::Status jsonToConfig(const std::string &json,
                                    bool is_path = true);

  virtual base::Status set(const std::string &key, const base::Value &value);

  virtual base::Status get(const std::string &key, base::Value &value);

  // interprt
  base::InferenceType model_type_;
  bool is_path_ = true;
  std::vector<std::string> model_value_;
  base::EncryptType encrypt_type_ = base::kEncryptTypeNone;
  std::string license_;

  // forward
  base::DeviceType device_types_;
  base::ShareMemoryType share_memory_mode_ = base::kShareMemoryTypeNoShare;
  base::PrecisionType precision_ = base::kPrecisionTypeFp32;
  base::PowerType power_type = base::kPowerTypeNormal;
  bool is_dynamic_shape_ = false;
  base::ShapeMap input_shape_ = base::ShapeMap();
  bool is_quant = false;
  base::InferenceOptLevel opt_level_ = base::kInferenceOptLevelAuto;
  std::string cache_path_ = "";
  bool is_tune_kernel_ = false;
};

class ConfigCreator {
 public:
  virtual ~ConfigCreator(){};
  virtual DefaultConfigImpl *createConfig() = 0;
};

template <typename T>
class TypeConfigCreator : public ConfigCreator {
  virtual DefaultConfigImpl *createConfig() { return new T(); }
};

std::map<base::InferenceType, std::shared_ptr<ConfigCreator>>
    &getGlobalConfigCreatorMap();

template <typename T>
class TypeConfigRegister {
 public:
  explicit TypeConfigRegister(base::InferenceType type) {
    getGlobalConfigCreatorMap()[type] = std::shared_ptr<T>(new T());
  }
};

DefaultConfigImpl *createConfig(base::InferenceType type);

class Config {
 public:
  explicit Config(base::InferenceType type);
  virtual ~Config();

  base::Status jsonToConfig(const std::string &json, bool is_path = true);

  base::Status set(const std::string &key, const base::Value &value);

  base::Status get(const std::string &key, base::Value &value);

  DefaultConfigImpl *config_impl_;

  base::InferenceType type_;
};

}  // namespace inference
}  // namespace nndeploy

#endif
