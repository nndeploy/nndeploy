/**
 * @file config_impl.h
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2022-12-20
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef _NNDEPLOY_INCLUDE_INFERENCE_CONFIG_IMPL_H_
#define _NNDEPLOY_INCLUDE_INFERENCE_CONFIG_IMPL_H_

#include "nndeploy/include/base/basic.h"
#include "nndeploy/include/base/log.h"
#include "nndeploy/include/base/macro.h"
#include "nndeploy/include/base/object.h"
#include "nndeploy/include/base/status.h"
#include "nndeploy/include/base/value.h"


namespace nndeploy {
namespace inference {

class ConfigImpl {
 public:
  virtual base::Status set(const std::string &key, const base::Value &value);

  virtual base::Status get(const std::string &key, base::Value &value);

 private:
  base::InferenceType model_type_;
  base::InferenceType inference_type_;
  bool is_path_ = true;
  bool is_encrypt_ = false;
  std::vector<std::string> model_value_;

  base::DeviceType device_types_;
  base::ShareMemoryType share_memory_mode_ = base::SHARE_MEMORY_TYPE_NO_SHARE;
  base::PrecisionType precision_ = base::PRECISION_TYPE_FP32;
  base::PowerType power_type = base::POWER_TYPE_NORMAL;

  bool is_dynamic_shape_ = false;
  base::ShapeMap input_shape_ = base::ShapeMap();
  bool is_quant = false;

  base::InferenceOptLevel opt_level_ = base::INFERENCE_OPT_LEVEL_AUTO;

  std::string cache_path_ = "";
  bool is_tune_kernel_ = false;
};

}  // namespace inference
}  // namespace nndeploy

#endif
