#include "nntask/source/common/pre_process/opencv/common.h"

namespace nntask {
namespace common {

int OpencvConvert::convertFromCvtColorType(nndeploy::base::CvtColorType src) {
  int ret = -1;
  switch (src) {
    case nndeploy::base::kCvtColorTypeRGB2GRAY:
      ret = cv::COLOR_RGB2GRAY;
      break;
    case nndeploy::base::kCvtColorTypeBGR2GRAY:
      ret = cv::COLOR_BGR2GRAY;
      break;
    case nndeploy::base::kCvtColorTypeRGBA2GRAY:
      ret = cv::COLOR_RGBA2GRAY;
      break;
    case nndeploy::base::kCvtColorTypeBGRA2GRAY:
      ret = cv::COLOR_BGRA2GRAY;
      break;
    case nndeploy::base::kCvtColorTypeGRAY2RGB:
      ret = cv::COLOR_GRAY2RGB;
      break;
    case nndeploy::base::kCvtColorTypeBGR2RGB:
      ret = cv::COLOR_BGR2RGB;
      break;
    case nndeploy::base::kCvtColorTypeRGBA2RGB:
      ret = cv::COLOR_RGBA2RGB;
      break;
    case nndeploy::base::kCvtColorTypeBGRA2RGB:
      ret = cv::COLOR_BGRA2RGB;
      break;
    case nndeploy::base::kCvtColorTypeGRAY2BGR:
      ret = cv::COLOR_GRAY2BGR;
      break;
    case nndeploy::base::kCvtColorTypeRGB2BGR:
      ret = cv::COLOR_RGB2BGR;
      break;
    case nndeploy::base::kCvtColorTypeRGBA2BGR:
      ret = cv::COLOR_RGBA2BGR;
      break;
    case nndeploy::base::kCvtColorTypeBGRA2BGR:
      ret = cv::COLOR_BGRA2BGR;
      break;
    case nndeploy::base::kCvtColorTypeGRAY2RGBA:
      ret = cv::COLOR_GRAY2RGBA;
      break;
    case nndeploy::base::kCvtColorTypeRGB2RGBA:
      ret = cv::COLOR_RGB2RGBA;
      break;
    case nndeploy::base::kCvtColorTypeBGR2RGBA:
      ret = cv::COLOR_RGB2GRAY;
      break;
    case nndeploy::base::kCvtColorTypeBGRA2RGBA:
      ret = cv::COLOR_BGRA2RGBA;
      break;
    case nndeploy::base::kCvtColorTypeGRAY2BGRA:
      ret = cv::COLOR_GRAY2BGRA;
      break;
    case nndeploy::base::kCvtColorTypeRGB2BGRA:
      ret = cv::COLOR_RGB2BGRA;
      break;
    case nndeploy::base::kCvtColorTypeBGR2BGRA:
      ret = cv::COLOR_BGR2BGRA;
      break;
    case nndeploy::base::kCvtColorTypeRGBA2BGRA:
      ret = cv::COLOR_RGBA2BGRA;
      break;
    default:
      ret = -1;
      break;
  }

  return ret;
}

int OpencvConvert::convertFromInterpType(nndeploy::base::InterpType src) {
  int ret = -1;
  switch (src) {
    case nndeploy::base::kInterpTypeNearst:
      ret = cv::INTER_NEAREST;
      break;
    case nndeploy::base::kInterpTypeLinear:
      ret = cv::INTER_LINEAR;
      break;
    case nndeploy::base::kInterpTypeCubic:
      ret = cv::INTER_CUBIC;
      break;
    case nndeploy::base::kInterpTypeArer:
      ret = cv::INTER_AREA;
      break;
    case nndeploy::base::kInterpTypeNotSupport:
      ret = -1;
      break;
    default:
      ret = -1;
      break;
  }

  return ret;
}

/**
 * @brief cast + normalize + premute
 *
 * @return true
 * @return false
 */
bool OpencvConvert::convertToTensor(const cv::Mat& src,
                                    nndeploy::device::Tensor* dst, float* mean,
                                    float* std) {
  bool ret = false;

  int c = dst->getShapeIndex(1);
  int h = dst->getShapeIndex(2);
  int w = dst->getShapeIndex(3);

  if (dst->getDataFormat() == nndeploy::base::kDataFormatNCHW) {
    cv::Mat tmp;
    int cv_type = CV_MAKETYPE(CV_32F, c);
    src.convertTo(tmp, cv_type);
    std::vector<cv::Mat> tmp_vec;
    for (int i = 0; i < c; ++i) {
      float* data = (float*)dst->getPtr() + w * h * i;
      tmp_vec.emplace_back(cv::Mat(cv::Size(w, h), CV_32FC1, data));
    }
    cv::split(tmp, tmp_vec);
    for (int i = 0; i < c; ++i) {
      float scale = 1.0f / std[i];
      float bias = -mean[i] / std[i];
      tmp_vec[i] = tmp_vec[i] * scale;
      tmp_vec[i] = tmp_vec[i] + bias;
    }
    ret = true;
  } else {
    ret = false;
  }

  return ret;
}

}  // namespace common
}  // namespace nntask