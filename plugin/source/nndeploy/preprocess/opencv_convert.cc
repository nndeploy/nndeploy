#include "nndeploy/preprocess/opencv_convert.h"

#include "nndeploy/preprocess/util.h"

namespace nndeploy {
namespace preprocess {

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
    case base::kBorderTypeNotSupport:
      ret = -1;
      break;
    default:
      ret = -1;
      break;
  }

  return ret;
}

cv::Scalar OpenCvConvert::convertFromScalar(const base::Scalar2d &src) {
  cv::Scalar ret(src.val_[0], src.val_[1], src.val_[2], src.val_[3]);
  return ret;
}

int OpenCvConvert::convertFromDataType(base::DataType src) {
  int dst = -1;
  if (src.code_ == base::kDataTypeCodeInt && src.bits_ == 8 &&
      src.lanes_ == 1) {
    dst = CV_8S;
  } else if (src.code_ == base::kDataTypeCodeInt && src.bits_ == 16 &&
             src.lanes_ == 1) {
    dst = CV_16S;
  } else if (src.code_ == base::kDataTypeCodeInt && src.bits_ == 32 &&
             src.lanes_ == 1) {
    dst = CV_32S;
  } else if (src.code_ == base::kDataTypeCodeUint && src.bits_ == 8 &&
             src.lanes_ == 1) {
    dst = CV_8U;
  } else if (src.code_ == base::kDataTypeCodeUint && src.bits_ == 16 &&
             src.lanes_ == 1) {
    dst = CV_16U;
  } else if (src.code_ == base::kDataTypeCodeFp && src.bits_ == 16 &&
             src.lanes_ == 1) {
#if CV_VERSION_MAJOR >= 4
    dst = CV_16F;
#else
    dst = CV_32F;
#endif
  } else if (src.code_ == base::kDataTypeCodeFp && src.bits_ == 32 &&
             src.lanes_ == 1) {
    dst = CV_32F;
  } else if (src.code_ == base::kDataTypeCodeFp && src.bits_ == 64 &&
             src.lanes_ == 1) {
    dst = CV_64F;
  } else {
    NNDEPLOY_LOGE("convertFromDataType failed\n");
    dst = -1;
  }
  return dst;
}

bool OpenCvConvert::normalize(const cv::Mat &src, cv::Mat &dst,
                              base::DataType data_type, float *scale,
                              float *mean, float *std) {
  int src_channels = src.channels();
  uint8_t *src_data = (uint8_t *)src.data;
  int dst_channels = dst.channels();
  void *dst_data = (void *)dst.data;
  int size = dst.rows * dst.cols;
  if (src_channels != dst_channels) {
    NNDEPLOY_LOGE("src_channels[%d] != dst_channels[%d].\n", src_channels,
                  dst_channels);
  }
  if (data_type.code_ == base::kDataTypeCodeFp && data_type.bits_ == 16 &&
      data_type.lanes_ == 1) {
    if (src_channels == 1) {
      normalizeFp16C1(src_data, dst_data, size, scale, mean, std);
    } else if (src_channels == 2) {
      normalizeFp16C2(src_data, dst_data, size, scale, mean, std);
    } else if (src_channels == 3) {
      normalizeFp16C3(src_data, dst_data, size, scale, mean, std);
    } else if (src_channels == 4) {
      normalizeFp16C4(src_data, dst_data, size, scale, mean, std);
    } else {
      normalizeFp16CN(src_data, dst_data, src_channels, size, scale, mean, std);
    }
  } else if (data_type.code_ == base::kDataTypeCodeFp &&
             data_type.bits_ == 32 && data_type.lanes_ == 1) {
    if (src_channels == 1) {
      normalizeFp32C1(src_data, (float *)dst_data, size, scale, mean, std);
    } else if (src_channels == 2) {
      normalizeFp32C2(src_data, (float *)dst_data, size, scale, mean, std);
    } else if (src_channels == 3) {
      normalizeFp32C3(src_data, (float *)dst_data, size, scale, mean, std);
    } else if (src_channels == 4) {
      normalizeFp32C4(src_data, (float *)dst_data, size, scale, mean, std);
    } else {
      normalizeFp32CN(src_data, (float *)dst_data, src_channels, size, scale,
                      mean, std);
    }
  } else if (data_type.code_ == base::kDataTypeCodeFp &&
             data_type.bits_ == 64 && data_type.lanes_ == 1) {
    if (src_channels == 1) {
      normalizeC1(src_data, (double *)dst_data, size, scale, mean, std);
    } else if (src_channels == 2) {
      normalizeC2(src_data, (double *)dst_data, size, scale, mean, std);
    } else if (src_channels == 3) {
      normalizeC3(src_data, (double *)dst_data, size, scale, mean, std);
    } else if (src_channels == 4) {
      normalizeC4(src_data, (double *)dst_data, size, scale, mean, std);
    } else {
      normalizeCN(src_data, (double *)dst_data, src_channels, size, scale, mean,
                  std);
    }
  } else if (data_type.code_ == base::kDataTypeCodeBFp &&
             data_type.bits_ == 16 && data_type.lanes_ == 1) {
    if (src_channels == 1) {
      normalizeBfp16C1(src_data, dst_data, size, scale, mean, std);
    } else if (src_channels == 2) {
      normalizeBfp16C2(src_data, dst_data, size, scale, mean, std);
    } else if (src_channels == 3) {
      normalizeBfp16C3(src_data, dst_data, size, scale, mean, std);
    } else if (src_channels == 4) {
      normalizeBfp16C4(src_data, dst_data, size, scale, mean, std);
    } else {
      normalizeBfp16CN(src_data, dst_data, src_channels, size, scale, mean,
                       std);
    }
  } else {
    NNDEPLOY_LOGE("data type not support!\n");
    return false;
  }
  return true;
}

/**
 * @brief cast + normalize + premute
 *
 * @return true
 * @return false
 */
bool OpenCvConvert::convertToTensor(const cv::Mat &src, device::Tensor *dst,
                                    bool normalize, float *scale, float *mean,
                                    float *std) {
  bool ret = true;

  int c = dst->getChannel();
  int h = dst->getHeight();
  int w = dst->getWidth();
  base::DataType data_type = dst->getDataType();
  size_t data_type_size = data_type.size();
  int cv_type = OpenCvConvert::convertFromDataType(data_type);
  int cv_single_type = CV_MAKETYPE(cv_type, 1);
  int cv_mix_type = CV_MAKETYPE(cv_type, c);

  if (dst->getDataFormat() == base::kDataFormatNCHW) {
    cv::Mat tmp(cv::Size(w, h), cv_mix_type);
    if (normalize) {
      ret = OpenCvConvert::normalize(src, tmp, data_type, scale, mean, std);
      if (!ret) {
        return ret;
      }
    } else {
      src.convertTo(tmp, cv_mix_type);
    }
    std::vector<cv::Mat> tmp_vec;
    for (int i = 0; i < c; ++i) {
      int8_t *data = ((int8_t *)dst->getData()) + w * h * i * data_type_size;
      tmp_vec.emplace_back(
          cv::Mat(cv::Size(w, h), cv_single_type, (void *)data));
    }
    cv::split(tmp, tmp_vec);
  } else if (dst->getDataFormat() == base::kDataFormatNHWC) {
    int8_t *data = (int8_t *)dst->getData();
    cv::Mat tmp(cv::Size(w, h), cv_mix_type, (void *)data);
    if (normalize) {
      ret = OpenCvConvert::normalize(src, tmp, data_type, scale, mean, std);
      if (!ret) {
        return ret;
      }
    } else {
      src.convertTo(tmp, cv_mix_type);
    }
  } else {
    NNDEPLOY_LOGE("data format not support!\n");
    ret = false;
  }

  return ret;
}

bool OpenCvConvert::convertToBatchTensor(const cv::Mat &src, device::Tensor *dst,
                                    bool normalize, float *scale, float *mean,
                                    float *std, int batch_index) {
  bool ret = true;

  int n = dst->getBatch();   // batch size
  int c = dst->getChannel();
  int h = dst->getHeight();
  int w = dst->getWidth();
  base::DataType data_type = dst->getDataType();
  size_t data_type_size = data_type.size();
  int cv_type = OpenCvConvert::convertFromDataType(data_type);
  int cv_single_type = CV_MAKETYPE(cv_type, 1);
  int cv_mix_type = CV_MAKETYPE(cv_type, c);

  // 每个 batch 的偏移量
  size_t batch_offset = (size_t)batch_index * c * h * w * data_type_size;

  if (dst->getDataFormat() == base::kDataFormatNCHW) {
    cv::Mat tmp(cv::Size(w, h), cv_mix_type);
    if (normalize) {
      ret = OpenCvConvert::normalize(src, tmp, data_type, scale, mean, std);
      if (!ret) {
        return ret;
      }
    } else {
      src.convertTo(tmp, cv_mix_type);
    }
    std::vector<cv::Mat> tmp_vec;
    for (int i = 0; i < c; ++i) {
      int8_t *data = ((int8_t *)dst->getData()) + batch_offset + w * h * i * data_type_size;
      tmp_vec.emplace_back(
          cv::Mat(cv::Size(w, h), cv_single_type, (void *)data));
    }
    cv::split(tmp, tmp_vec);
  } else if (dst->getDataFormat() == base::kDataFormatNHWC) {
    int8_t *data = (int8_t *)dst->getData() + batch_offset;
    cv::Mat tmp(cv::Size(w, h), cv_mix_type, (void *)data);
    if (normalize) {
      ret = OpenCvConvert::normalize(src, tmp, data_type, scale, mean, std);
      if (!ret) {
        return ret;
      }
    } else {
      src.convertTo(tmp, cv_mix_type);
    }
  } else {
    NNDEPLOY_LOGE("data format not support!\n");
    ret = false;
  }

  return ret;
}

}  // namespace preprocess
}  // namespace nndeploy