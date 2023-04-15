
#ifndef _NNDEPLOY_SOURCE_INFERENCE_TENSOR_RT_TENSOR_RT_CONFIG_IMPL_H_
#define _NNDEPLOY_SOURCE_INFERENCE_TENSOR_RT_TENSOR_RT_CONFIG_IMPL_H_

#include "nndeploy/source/inference/config.h"
#include "nndeploy/source/inference/tensor_rt/tensor_rt_include.h"

namespace nndeploy {
namespace inference {

class TensorRtConfigImpl : public DefaultConfigImpl {
 public:
  TensorRtConfigImpl();
  virtual ~TensorRtConfigImpl();

  base::Status jsonToConfig(const std::string &json, bool is_path = true);

  virtual base::Status set(const std::string &key, base::Value &value);

  virtual base::Status get(const std::string &key, base::Value &value);

  int32_t max_batch_size_ = 1;
  size_t workspace_size_ = 1 << 30;
  std::string int8_calibration_table_path_ = "";
  std::string model_save_path_ = "";
};

}  // namespace inference
}  // namespace nndeploy

#endif
