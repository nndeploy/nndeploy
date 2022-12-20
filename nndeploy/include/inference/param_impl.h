

#ifndef _NNDEPLOY_INCLUDE_INFERENCE_CONFIG_IMPL_H_
#define _NNDEPLOY_INCLUDE_INFERENCE_CONFIG_IMPL_H_

#include "nndeploy/include/base/config.h"
#include "nndeploy/include/base/status.h"

using namespace nndeploy::base;

namespace nndeploy {
namespace inference {

struct Config {
 public:
  template<typename T>
  virtual Status set(const std::string &key, const T &value);

  template<typename T>
  virtual Status get(const std::string &key, T &value);

 private:
  InferenceType model_type_;
  InferenceType inference_type_;
  bool is_path_ = false;
  bool is_encrypt_ = false;
  std::vector<std::string> model_value_;

  DeviceType device_types_;
  ShareMemoryType share_memory_mode_ = SHARE_MEMORY_TYPE_NO_SHARE;
  PrecisionType precision_ = PRECISION_TYPE_NORMAL;
  PowerType power_type = POWER_TYPE_NORMAL;

  bool is_dynamic_shape_ = false;
  ShapeMap input_shape_ = ShapeMap();
  bool is_quant = false;

  InferenceOptType opt_type_ = INFERENCE_OPT_TYPE_AUTO;
  std::string cache_path_ = "";
  bool is_tune_kernel_ = false;
};

}  // namespace inference
}  // namespace nndeploy

#endif
