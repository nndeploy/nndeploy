
#ifndef _NNDEPLOY_SOURCE_INFERENCE_MNN_MNN_INFERENCE_PARAM_H_
#define _NNDEPLOY_SOURCE_INFERENCE_MNN_MNN_INFERENCE_PARAM_H_

#include "nndeploy/source/inference/inference_param.h"
#include "nndeploy/source/inference/mnn/mnn_include.h"

namespace nndeploy {
namespace inference {

class MnnInferenceParam : public InferenceParam {
 public:
  MnnInferenceParam();
  MnnInferenceParam(std::string name);
  virtual ~MnnInferenceParam();

  base::Status parse(const std::string &json, bool is_path = true);

  virtual base::Status set(const std::string &key, base::Value &value);

  virtual base::Status get(const std::string &key, base::Value &value);

  std::vector<std::string> save_tensors_;
  MNN::ScheduleConfig::Path path_;
  base::DeviceType backup_device_type_ = {base::kDeviceTypeCodeCpu, 0};
  MNN::BackendConfig::MemoryMode memory_mode_ =
      MNN::BackendConfig::MemoryMode::Memory_Normal;
};

}  // namespace inference
}  // namespace nndeploy

#endif
