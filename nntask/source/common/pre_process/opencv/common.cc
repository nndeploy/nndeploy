#include "nntask/source/common/pre_process/opencv/common.h"

namespace nntask {
namespace common {

int OpencvConvert::convertFromPixType(nndeploy::base::PixelType src) {
  int ret = -1;
  switch (src) {
    case nndeploy::base::kPixelTypeGRAY:
      ret = CV_8UC1;
      break;
    case nndeploy::base::kPixelTypeBGR:
      ret = CV_8UC3;
      break;
    case nndeploy::base::kPixelTypeRGB:
      ret = CV_8UC3;
      break;
    case nndeploy::base::kPixelTypeBGRA:
      ret = CV_8UC4;
      break;
    case nndeploy::base::kPixelTypeRGBA:
      ret = CV_8UC4;
      break;
    default:
      ret = ;
      break;
  }

  return ret;
}

int OpencvConvert::convertFromInterpType(nndeploy::base::InterpType src) {}

/**
 * @brief cast + normalize + premute
 *
 * @return true
 * @return false
 */
bool OpencvConvert::convertToTensor(const cv::Mat& src,
                                    nndeploy::device::Tensor* dst,
                                    cv::Scalar& mean, cv::Scalar& std) {
  bool ret = false;

  int c = dst->getShapeIndex(1);
  int h = dst->getShapeIndex(2);
  int w = dst->getShapeIndex(3);

  cv::Scalar scale = 1.0f / std;
  cv::Scalar bias = -mean / std;
  if (dst->getDataFormat() == nndeploy::base::kDataFormatNCHW) {
    cv::Mat tmp;
    int cv_type = CV_MAKETYPE(CV_32F, c);
    src.convertTo(tmp, cv_type);
    tmp = tmp * scale + bias;
    std::vector<cv::Mat> tmp_vec;
    for (int i = 0; i < c; ++i) {
      float* data = (float*)dst->getPtr() + w * h * i;
      tmp_vec.emplace_back(cv::Mat(cv::Size(w, h), CV_32FC1, data));
    }
    cv::split(tmp, tmp_vec);
    ret = true;
  } else {
    ret = false;
  }

  return ret;
}

}  // namespace common
}  // namespace nntask