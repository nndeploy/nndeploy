/**
 * @github: https://github.com/facebookresearch/detr
 * @huggingface: https://github.com/facebookresearch/detr
 */
#ifndef _NNDEPLOY_DETECT_0PENCV_DETR_H_
#define _NNDEPLOY_DETECT_0PENCV_DETR_H_

#include "nndeploy/base/basic.h"
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
#include "nndeploy/task/detect/result.h"
#include "nndeploy/task/packet.h"
#include "nndeploy/task/task.h"

namespace nndeploy {
namespace task {

class DetrPostParam : public base::Param {
 public:
  float score_threshold_ = 0.7f;
};

class DetrPostProcess : public Task {
 public:
  DetrPostProcess(const std::string& name = "") : Task(name) {
    param_ = std::make_shared<DetrPostParam>();
  }
  virtual ~DetrPostProcess() {}

  virtual base::Status run();

 private:
  DetectResults results_;
};

task::Task* creatDetrTask(const std::string& name, base::InferenceType type);

}  // namespace task
}  // namespace nndeploy

#endif /* _NNDEPLOY_DETECT_0PENCV_POST_PROCESS_H_ */
