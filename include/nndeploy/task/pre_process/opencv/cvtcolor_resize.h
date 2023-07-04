#ifndef _NNDEPLOY_TASK_PRE_PROCESS_0PENCV_CVTCOLOR_RESIZE_H_
#define _NNDEPLOY_TASK_PRE_PROCESS_0PENCV_CVTCOLOR_RESIZE_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/task/packet.h"
#include "nndeploy/task/pre_process/opencv/opencv_convert.h"
#include "nndeploy/task/pre_process/params.h"
#include "nndeploy/task/task.h"

namespace nndeploy {
namespace task {
namespace opencv {

class CvtColrResize : public Task {
 public:
  CvtColrResize(const std::string& name, Packet* input, Packet* output)
      : Task(name, input, output) {
    param_ = std::make_shared<CvtclorResizeParam>();
  }
  virtual ~CvtColrResize() {}

  virtual base::Status run();
};

}  // namespace opencv
}  // namespace task
}  // namespace nndeploy

#endif /* _NNDEPLOY_TASK_PRE_PROCESS_0PENCV_CVTCOLOR_RESIZE_H_ */
