#ifndef _NNDEPLOY_SOURCE_DETECT_0PENCV_DETR_H_
#define _NNDEPLOY_SOURCE_DETECT_0PENCV_DETR_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/base/string.h"
#include "nndeploy/source/base/value.h"
#include "nndeploy/source/device/buffer.h"
#include "nndeploy/source/device/buffer_pool.h"
#include "nndeploy/source/device/device.h"
#include "nndeploy/source/device/tensor.h"
#include "nndeploy/source/task/execution.h"
#include "nndeploy/source/task/opencv_include.h"
#include "nndeploy/source/task/packet.h"
#include "nndeploy/source/task/results.h"
#include "nndeploy/source/task/task.h"

namespace nndeploy {
namespace task {

class DetrPostProcess : public Execution {
 public:
  DetrPostProcess(const std::string& name = "") : Execution(name) {}
  virtual ~DetrPostProcess() {}

  struct Bbox {
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
    int cid;
  };

  virtual base::Status run();
};

task::Task* creatDetrTask(const std::string& name, base::InferenceType type);

}  // namespace task
}  // namespace nndeploy

#endif /* _NNDEPLOY_SOURCE_DETECT_0PENCV_POST_PROCESS_H_ */
