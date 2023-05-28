
#include "nntask/source/common/process/opencv/pre_process.h"

namespace nntask {
namespace common {

/**
 * @brief cast + normalize + premute
 *
 * @return true
 * @return false
 */
bool matToTensor(const cv::Mat& src, nndeploy::device::Tensor* dst,
                 cv::Scalar& scale, cv::Scalar& bias) {
  int c = dst->getShapeIndex(1);
  int h = dst->getShapeIndex(2);
  int w = dst->getShapeIndex(3);

  if (dst->getDataFormat() == nndeploy::base::kDataFormatNCHW) {
    cv::Mat tmp;
    int cv_type = CV_MAKETYPE(CV_32F, c);
    src.convertTo(tmp, cv_type);
    // tmp = tmp * scale;
    // tmp = tmp - bias;
    tmp = tmp / 255.0f;
    std::vector<cv::Mat> tmp_vec;
    for (int i = 0; i < c; ++i) {
      float* data = (float*)dst->getPtr() + w * h * i;
      tmp_vec.emplace_back(cv::Mat(cv::Size(w, h), CV_32FC1, data));
    }

    cv::split(tmp, tmp_vec);
  }
  return true;
}

}  // namespace common
}  // namespace nntask
