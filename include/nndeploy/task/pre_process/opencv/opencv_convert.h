#ifndef _NNDEPLOY_TASK_PRE_PROCESS_0PENCV_OPENCV_CONVERT_H_
#define _NNDEPLOY_TASK_PRE_PROCESS_0PENCV_OPENCV_CONVERT_H_

#include "nndeploy/base/basic.h"
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
#include "nndeploy/task/pre_process/params.h"
#include "nndeploy/task/task.h"

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
                              float* scale, float* mean, float* std);
};

}  // namespace task
}  // namespace nndeploy

#endif
