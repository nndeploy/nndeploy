
#ifndef _NNDEPLOY_MODEL_SEGMENT_SAM_SAM_H_
#define _NNDEPLOY_MODEL_SEGMENT_SAM_SAM_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/model/segment/result.h"
#include "nndeploy/model/packet.h"
#include "nndeploy/model/pipeline.h"
#include "nndeploy/model/task.h"

namespace nndeploy {
namespace model {

#define NNDEPLOY_SAM "NNDEPLOY_SAM"

// 后处理
class NNDEPLOY_CC_API SamPostParam : public base::Param {
 public:
};

class NNDEPLOY_CC_API SamPostProcess : public model::Task {
 public:
  SamPostProcess(const std::string& name, model::Packet* input,
                 model::Packet* output)
      : Task(name, input, output) {
    param_ = std::make_shared<SamPostParam>();
  }
  virtual ~SamPostProcess() {}

  virtual base::Status run();
};

// 构建prompt_endocer和mask_decoder的输入
class NNDEPLOY_CC_API SamBuildInput : public model::Task {
 public:
  SamBuildInput(const std::string& name, model::Packet* input,
                model::Packet* output)
      : Task(name, input, output) {}
  virtual ~SamBuildInput() {}

  virtual base::Status run();
};

extern NNDEPLOY_CC_API model::Pipeline* createSamPipeline(
    const std::string& name, base::InferenceType inference_type,
    base::DeviceType device_type, Packet* input, Packet* output,
    base::ModelType model_type, bool is_path,
    std::vector<std::string> model_values);

}  // namespace model
}  // namespace nndeploy
#endif