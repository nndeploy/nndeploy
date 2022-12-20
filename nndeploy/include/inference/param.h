
#ifndef _NNDEPLOY_INCLUDE_INFERENCE_CONFIG_H_
#define _NNDEPLOY_INCLUDE_INFERENCE_CONFIG_H_

#include "nndeploy/include/base/config.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/inference/config_impl.h"

using namespace nndeploy::base;

namespace nndeploy {
namespace inference {

class Config {
 public:
  template<typename T>
  virtual Status set(const std::string &key, const T &value);

  template<typename T>
  virtual Status get(const std::string &key, T &value);

  static Config createConfig(InferenceType model_type,
  InferenceType inference_type, std::vector<std::string> params)

 private:
  std::shared_ptr<ConfigImpl> config_impl;
};

}  // namespace inference
}  // namespace nndeploy

#endif

