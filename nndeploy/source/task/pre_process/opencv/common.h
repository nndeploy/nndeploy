#ifndef _NNDEPLOY_SOURCE_TASK_PRE_PROCESS_0PENCV_COMMON_H_
#define _NNDEPLOY_SOURCE_TASK_PRE_PROCESS_0PENCV_COMMON_H_

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
#include "nndeploy/source/task/params.h"
#include "nndeploy/source/task/task.h"

namespace nndeploy {
namespace task {

class OpencvConvert {
 public:
  static int convertFromCvtColorType(base::CvtColorType src);

  static int convertFromInterpType(base::InterpType src);

  /**
   * @brief cast + normalize + premute
   *
   * @return true
   * @return false
   */
  static bool convertToTensor(const cv::Mat& src, device::Tensor* dst,
                              float* mean, float* std);
};

}  // namespace task
}  // namespace nndeploy

#endif
