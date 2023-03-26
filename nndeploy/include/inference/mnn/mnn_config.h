
#ifndef _NNDEPLOY_INCLUDE_INFERENCE_MNN_MNN_CONFIG_IMPL_H_
#define _NNDEPLOY_INCLUDE_INFERENCE_MNN_MNN_CONFIG_IMPL_H_

#include "nndeploy/include/inference/config.h"

namespace nndeploy {
namespace inference {

class MnnConfigImpl : public DefaultConfigImpl {
 public:
  MnnConfigImpl();
  virtual ~MnnConfigImpl();

  base::Status jsonToConfig(const std::string &json, bool is_path = true);

  virtual base::Status set(const std::string &key, const base::Value &value);

  virtual base::Status get(const std::string &key, base::Value &value);

  std::string library_path_ = "";
};

}  // namespace inference
}  // namespace nndeploy

#endif
