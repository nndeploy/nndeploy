#ifndef _NNDEPLOY_BASIC_OPENCV_UTIL_H_
#define _NNDEPLOY_BASIC_OPENCV_UTIL_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/basic/opencv_convert.h"
#include "nndeploy/basic/params.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace basic {

class OpenCvUtil {
 public:
  static void cvtColor(cv::InputArray src, cv::OutputArray dst,
                       base::PixelType src_pixel_type,
                       base::PixelType dst_pixel_type);
  static void cvtColor(cv::InputArray src, cv::OutputArray dst,
                       const CvtcolorParam &param);

  static void resize(cv::InputArray src, cv::OutputArray dst, int dst_w,
                     int dst_h, float scale_w, float scale_h,
                     base::InterpType interp_type);
  static void resize(cv::InputArray src, cv::OutputArray dst,
                     const ResizeParam &param);

  static void copyMakeBorder(cv::InputArray src, cv::OutputArray dst, int top,
                             int bottom, int left, int right,
                             base::BorderType border_type,
                             const base::Scalar2d &border_val);
  static void copyMakeBorder(cv::InputArray src, cv::OutputArray dst,
                             const PaddingParam &param);

  static void warpAffine(cv::InputArray src, cv::OutputArray dst,
                         float *transform, int dst_w, int dst_h,
                         base::InterpType interp_type,
                         base::BorderType border_type,
                         const base::Scalar2d &border_val);
  static void warpAffine(cv::InputArray src, cv::OutputArray dst,
                         const WarpAffineParam &param);

  static cv::Mat crop(cv::InputArray src, int x, int y, int w, int h);
  static cv::Mat crop(cv::InputArray src, const CropParam &param);
};

}  // namespace basic
}  // namespace nndeploy

#endif
