
#ifndef _NNDEPLOY_INFERENCE_MNN_MNN_INFERENCE_PARAM_H_
#define _NNDEPLOY_INFERENCE_MNN_MNN_INFERENCE_PARAM_H_

#include "nndeploy/device/device.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/mnn/mnn_include.h"

namespace nndeploy {
namespace inference {

class NNDEPLOY_CC_API MnnInferenceParam : public InferenceParam {
 public:
  MnnInferenceParam();
  MnnInferenceParam(base::InferenceType type);
  virtual ~MnnInferenceParam();

  MnnInferenceParam(const MnnInferenceParam &param) = default;
  MnnInferenceParam &operator=(const MnnInferenceParam &param) = default;

  PARAM_COPY(MnnInferenceParam)
  PARAM_COPY_TO(MnnInferenceParam)

  virtual base::Status set(const std::string &key, base::Any &any);

  virtual base::Status get(const std::string &key, base::Any &any);

  std::vector<std::string> save_tensors_;
  MNN::ScheduleConfig::Path path_;
  base::DeviceType backup_device_type_;
  MNN::BackendConfig::MemoryMode memory_mode_ =
      MNN::BackendConfig::MemoryMode::Memory_Normal;
};

}  // namespace inference
}  // namespace nndeploy

#endif
