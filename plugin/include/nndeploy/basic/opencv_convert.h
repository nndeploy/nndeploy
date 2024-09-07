#ifndef _NNDEPLOY_BASIC_OPENCV_CONVERT_H_
#define _NNDEPLOY_BASIC_OPENCV_CONVERT_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/basic/params.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace basic {

class OpenCvConvert {
 public:
  static int convertFromCvtColorType(base::CvtColorType src);

  static int convertFromInterpType(base::InterpType src);

  static int convertFromBorderType(base::BorderType src);

  static cv::Scalar convertFromScalar(const base::Scalar2d &src);

  static int convertFromDataType(base::DataType src);

  /**
   * @brief
   *
   * @param src must uint8_t
   * @param dst
   * @param data_type
   * @param scale
   * @param mean
   * @param std
   * @return true
   * @return false
   */
  static bool normalize(const cv::Mat &src, cv::Mat &dst,
                        base::DataType data_type, float *scale, float *mean,
                        float *std);

  /**
   * @brief cast + normalize + premute
   *
   * @return true
   * @return false
   */
  static bool convertToTensor(const cv::Mat &src, device::Tensor *dst,
                              bool normalize, float *scale, float *mean,
                              float *std);
};

}  // namespace basic
}  // namespace nndeploy

#endif
