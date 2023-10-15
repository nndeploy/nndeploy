#ifndef _NNDEPLOY_MODEL_PREPROCESS_OPENCV_CONVERT_H_
#define _NNDEPLOY_MODEL_PREPROCESS_OPENCV_CONVERT_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/dag/packet.h"
#include "nndeploy/dag/task.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/model/preprocess/params.h"

namespace nndeploy {
namespace model {

class OpenCvConvert {
 public:
  static int convertFromCvtColorType(base::CvtColorType src);

  static int convertFromInterpType(base::InterpType src);

  static int convertFromBorderType(base::BorderType src);

  static cv::Scalar convertFromScalar(const base::Scalar2d& src);

  /**
   * @brief cast + normalize + premute
   *
   * @return true
   * @return false
   */
  static bool convertToTensor(const cv::Mat& src, device::Tensor* dst,
                              float* scale, float* mean, float* std);
};

}  // namespace model
}  // namespace nndeploy

#endif
