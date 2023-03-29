
#ifndef _NNDEPLOY_SOURCE_INFERENCE_MNN_MNN_CONFIG_IMPL_H_
#define _NNDEPLOY_SOURCE_INFERENCE_MNN_MNN_CONFIG_IMPL_H_

#include "nndeploy/source/inference/config.h"
#include "nndeploy/source/inference/mnn/mnn_include.h"

namespace nndeploy {
namespace inference {

class MnnConfigImpl : public DefaultConfigImpl {
 public:
  MnnConfigImpl();
  virtual ~MnnConfigImpl();

  base::Status jsonToConfig(const std::string &json, bool is_path = true);

  virtual base::Status set(const std::string &key, base::Value &value);

  virtual base::Status get(const std::string &key, base::Value &value);

  std::vector<std::string> save_tensors_;
  int gpu_mode = 4;
  MNN::ScheduleConfig::Path path;
  base::DeviceType backup_device_type_;
  MNN::BackendConfig::MemoryMode memory_mode_ =
      MNN::BackendConfig::MemoryMode::Memory_Normal;

  std::string library_path_ = "";
};

}  // namespace inference
}  // namespace nndeploy

#endif
