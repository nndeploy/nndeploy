#include "nndeploy/model/preprocess/opencv_convert.h"

namespace nndeploy {
namespace model {

int OpenCvConvert::convertFromCvtColorType(base::CvtColorType src) {
  int ret = -1;
  switch (src) {
    case base::kCvtColorTypeRGB2GRAY:
      ret = cv::COLOR_RGB2GRAY;
      break;
    case base::kCvtColorTypeBGR2GRAY:
      ret = cv::COLOR_BGR2GRAY;
      break;
    case base::kCvtColorTypeRGBA2GRAY:
      ret = cv::COLOR_RGBA2GRAY;
      break;
    case base::kCvtColorTypeBGRA2GRAY:
      ret = cv::COLOR_BGRA2GRAY;
      break;
    case base::kCvtColorTypeGRAY2RGB:
      ret = cv::COLOR_GRAY2RGB;
      break;
    case base::kCvtColorTypeBGR2RGB:
      ret = cv::COLOR_BGR2RGB;
      break;
    case base::kCvtColorTypeRGBA2RGB:
      ret = cv::COLOR_RGBA2RGB;
      break;
    case base::kCvtColorTypeBGRA2RGB:
      ret = cv::COLOR_BGRA2RGB;
      break;
    case base::kCvtColorTypeGRAY2BGR:
      ret = cv::COLOR_GRAY2BGR;
      break;
    case base::kCvtColorTypeRGB2BGR:
      ret = cv::COLOR_RGB2BGR;
      break;
    case base::kCvtColorTypeRGBA2BGR:
      ret = cv::COLOR_RGBA2BGR;
      break;
    case base::kCvtColorTypeBGRA2BGR:
      ret = cv::COLOR_BGRA2BGR;
      break;
    case base::kCvtColorTypeGRAY2RGBA:
      ret = cv::COLOR_GRAY2RGBA;
      break;
    case base::kCvtColorTypeRGB2RGBA:
      ret = cv::COLOR_RGB2RGBA;
      break;
    case base::kCvtColorTypeBGR2RGBA:
      ret = cv::COLOR_RGB2GRAY;
      break;
    case base::kCvtColorTypeBGRA2RGBA:
      ret = cv::COLOR_BGRA2RGBA;
      break;
    case base::kCvtColorTypeGRAY2BGRA:
      ret = cv::COLOR_GRAY2BGRA;
      break;
    case base::kCvtColorTypeRGB2BGRA:
      ret = cv::COLOR_RGB2BGRA;
      break;
    case base::kCvtColorTypeBGR2BGRA:
      ret = cv::COLOR_BGR2BGRA;
      break;
    case base::kCvtColorTypeRGBA2BGRA:
      ret = cv::COLOR_RGBA2BGRA;
      break;
    default:
      ret = -1;
      break;
  }

  return ret;
}

int OpenCvConvert::convertFromInterpType(base::InterpType src) {
  int ret = -1;
  switch (src) {
    case base::kInterpTypeNearst:
      ret = cv::INTER_NEAREST;
      break;
    case base::kInterpTypeLinear:
      ret = cv::INTER_LINEAR;
      break;
    case base::kInterpTypeCubic:
      ret = cv::INTER_CUBIC;
      break;
    case base::kInterpTypeArer:
      ret = cv::INTER_AREA;
      break;
    case base::kInterpTypeNotSupport:
      ret = -1;
      break;
    default:
      ret = -1;
      break;
  }

  return ret;
}

int OpenCvConvert::convertFromBorderType(base::BorderType src) {
  int ret = -1;
  switch (src) {
    case base::kBorderTypeConstant:
      ret = cv::BORDER_CONSTANT;
      break;
    case base::kBorderTypeReflect:
      ret = cv::BORDER_REFLECT;
      break;
    case base::kBorderTypeEdge:
      ret = cv::BORDER_REPLICATE;
      break;
    case base::kInterpTypeNotSupport:
      ret = -1;
      break;
    default:
      ret = -1;
      break;
  }

  return ret;
}

cv::Scalar OpenCvConvert::convertFromScalar(const base::Scalar2d& src) {
  cv::Scalar ret(src.val_[0], src.val_[1], src.val_[2], src.val_[3]);
  return ret;
}

/**
 * @brief cast + normalize + premute
 *
 * @return true
 * @return false
 */
bool OpenCvConvert::convertToTensor(const cv::Mat& src, device::Tensor* dst,
                                    float* scale, float* mean, float* std) {
  bool ret = false;

  int c = dst->getShapeIndex(1);
  int h = dst->getShapeIndex(2);
  int w = dst->getShapeIndex(3);

  if (dst->getDataFormat() == base::kDataFormatNCHW) {
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
      float mul_scale = scale[i] / std[i];
      float add_bias = -mean[i] / std[i];
      tmp_vec[i] = tmp_vec[i] * mul_scale;
      tmp_vec[i] = tmp_vec[i] + add_bias;
    }
    ret = true;
  } else {
    NNDEPLOY_LOGE("data format not support!\n");
    ret = false;
  }

  return ret;
}

}  // namespace model
}  // namespace nndeploy