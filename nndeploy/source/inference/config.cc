
#include "nndeploy/source/inference/config.h"

namespace nndeploy {
namespace inference {

DefaultConfigImpl::DefaultConfigImpl() {}
DefaultConfigImpl::~DefaultConfigImpl() {}

base::Status DefaultConfigImpl::jsonToConfig(const std::string &json,
                                             bool is_path) {
  return base::kStatusCodeOk;
}

base::Status DefaultConfigImpl::set(const std::string &key,
                                    const base::Value &value) {
  return base::kStatusCodeOk;
}

base::Status DefaultConfigImpl::get(const std::string &key,
                                    base::Value &value) {
  return base::kStatusCodeOk;
}

std::map<base::InferenceType, std::shared_ptr<ConfigCreator>> &
getGlobalConfigCreatorMap() {
  static std::once_flag once;
  static std::shared_ptr<
      std::map<base::InferenceType, std::shared_ptr<ConfigCreator>>>
      creators;
  std::call_once(once, []() {
    creators.reset(
        new std::map<base::InferenceType, std::shared_ptr<ConfigCreator>>);
  });
  return *creators;
}

DefaultConfigImpl *createConfig(base::InferenceType type) {
  DefaultConfigImpl *temp = nullptr;
  auto &creater_map = getGlobalConfigCreatorMap();
  if (creater_map.count(type) > 0) {
    temp = creater_map[type]->createConfig();
  }
  return temp;
}

Config::Config(base::InferenceType type) : type_(type) {
  config_impl_ = createConfig(type);
}
Config::~Config() {
  if (config_impl_ != nullptr) {
    delete config_impl_;
  }
}

base::Status Config::jsonToConfig(const std::string &json, bool is_path) {
  if (config_impl_ != nullptr) {
    return config_impl_->jsonToConfig(json, is_path);
  } else {
    // TODO: log
    return base::kStatusCodeErrorInvalidValue;
  }
}

base::Status Config::set(const std::string &key, const base::Value &value) {
  if (config_impl_ != nullptr) {
    return config_impl_->set(key, value);
  } else {
    // TODO: log
    return base::kStatusCodeErrorInvalidValue;
  }
}

base::Status Config::get(const std::string &key, base::Value &value) {
  if (config_impl_ != nullptr) {
    return config_impl_->get(key, value);
  } else {
    // TODO: log
    return base::kStatusCodeErrorInvalidValue;
  }
}

}  // namespace inference
}  // namespace nndeploy
