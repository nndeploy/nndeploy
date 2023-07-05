/**
 * @github: https://github.com/facebookresearch/detr
 * @huggingface: https://github.com/facebookresearch/detr
 */
#ifndef _NNDEPLOY_MODEL_DETECT_0PENCV_DETR_DETR_H_
#define _NNDEPLOY_MODEL_DETECT_0PENCV_DETR_DETR_H_

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
#include "nndeploy/model/detect/result.h"
#include "nndeploy/pipeline/packet.h"
#include "nndeploy/pipeline/pipeline.h"
#include "nndeploy/pipeline/task.h"

namespace nndeploy {
namespace model {
namespace opencv {

class NNDEPLOY_CC_API DetrPostParam : public base::Param {
 public:
  float score_threshold_ = 0.7f;
};

class NNDEPLOY_CC_API DetrPostProcess : public pipeline::Task {
 public:
  DetrPostProcess(const std::string& name, pipeline::Packet* input,
                  pipeline::Packet* output)
      : Task(name, input, output) {
    param_ = std::make_shared<DetrPostParam>();
  }
  virtual ~DetrPostProcess() {}

  virtual base::Status run();

 private:
  DetectResults results_;
};

extern NNDEPLOY_CC_API pipeline::Pipeline* creatDetrPipeline(
    const std::string& name, base::InferenceType type, pipeline::Packet* input,
    pipeline::Packet* output, bool is_path,
    std::vector<std::string>& model_value);

}  // namespace opencv
}  // namespace model
}  // namespace nndeploy

#endif /* _NNDEPLOY_DETECT_0PENCV_POSTPROCESS_H_ */
