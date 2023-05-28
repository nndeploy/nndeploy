#ifndef _NNTASK_SOURCE_COMMON_PRE_PROCESS_0PENCV_COMMON_H_
#define _NNTASK_SOURCE_COMMON_PRE_PROCESS_0PENCV_COMMON_H_

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
#include "nntask/source/common/execution.h"
#include "nntask/source/common/opencv_include.h"
#include "nntask/source/common/packet.h"
#include "nntask/source/common/params.h"
#include "nntask/source/common/task.h"

namespace nntask {
namespace common {

class OpencvConvert {
 public:
  static int convertFromPixType(nndeploy::base::PixelType src);

  static int convertFromInterpType(nndeploy::base::InterpType src);

  /**
   * @brief cast + normalize + premute
   *
   * @return true
   * @return false
   */
  static bool convertToTensor(const cv::Mat& src, nndeploy::device::Tensor* dst,
                              cv::Scalar& mean, cv::Scalar& std);

}  // namespace common
}  // namespace nntask

#endif
