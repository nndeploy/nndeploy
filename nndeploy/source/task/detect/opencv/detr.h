/**
 * @source: git@github.com:DataXujing/TensorRT-DETR.git
 */
#ifndef _NNDEPLOY_SOURCE_DETECT_0PENCV_DETR_H_
#define _NNDEPLOY_SOURCE_DETECT_0PENCV_DETR_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/param.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/base/string.h"
#include "nndeploy/source/base/value.h"
#include "nndeploy/source/device/buffer.h"
#include "nndeploy/source/device/buffer_pool.h"
#include "nndeploy/source/device/device.h"
#include "nndeploy/source/device/tensor.h"
#include "nndeploy/source/task/detect/result.h"
#include "nndeploy/source/task/execution.h"
#include "nndeploy/source/task/opencv_include.h"
#include "nndeploy/source/task/packet.h"
#include "nndeploy/source/task/task.h"

namespace nndeploy {
namespace task {

class DetrPostParam : public base::Param {
  int32_t num_class_ = 22;
  int32_t num_qurrey_ = 100;  // detr默认是100
  float score_threshold_ = 0.7f;
};

class DetrPostProcess : public Execution {
 public:
  DetrPostProcess(const std::string& name = "") : Execution(name) {}
  virtual ~DetrPostProcess() {}

  virtual base::Status run();

 private:
  DetectResults results_;
};

task::Task* creatDetrTask(const std::string& name, base::InferenceType type);

}  // namespace task
}  // namespace nndeploy

#endif /* _NNDEPLOY_SOURCE_DETECT_0PENCV_POST_PROCESS_H_ */
