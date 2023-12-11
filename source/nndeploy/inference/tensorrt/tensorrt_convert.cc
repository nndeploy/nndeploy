
#include "nndeploy/inference/tensorrt/tensorrt_convert.h"

#include "nndeploy/inference/tensorrt/tensorrt_inference.h"

namespace nndeploy {
namespace inference {

base::DataType TensorRtConvert::convertToDataType(
    const nvinfer1::DataType &src) {
  base::DataType dst;
  switch (src) {
    case nvinfer1::DataType::kFLOAT:
      dst.code_ = base::kDataTypeCodeFp;
      dst.bits_ = 32;
      dst.lanes_ = 1;
      break;
    case nvinfer1::DataType::kHALF:
      dst.code_ = base::kDataTypeCodeBFp;
      dst.bits_ = 16;
      dst.lanes_ = 1;
      break;
    case nvinfer1::DataType::kINT32:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 32;
      dst.lanes_ = 1;
      break;
    case nvinfer1::DataType::kINT8:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 8;
      dst.lanes_ = 1;
      break;
#ifdef TENSORRT_MAJOR_8_MINOR_5
    case nvinfer1::DataType::kUINT8:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 8;
      dst.lanes_ = 1;
      break;
#endif
    case nvinfer1::DataType::kBOOL:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 8;
      dst.lanes_ = 1;
      break;
    default:
      dst.code_ = base::kDataTypeCodeFp;
      dst.bits_ = 32;
      dst.lanes_ = 1;
      break;
  }
  return dst;
}

nvinfer1::DataType TensorRtConvert::convertFromDataType(
    const base::DataType &src) {
  nvinfer1::DataType dst;
  if (src.code_ == base::kDataTypeCodeFp && src.bits_ == 32 &&
      src.lanes_ == 1) {
    dst = nvinfer1::DataType::kFLOAT;
  } else if (src.code_ == base::kDataTypeCodeBFp && src.bits_ == 16 &&
             src.lanes_ == 1) {
    dst = nvinfer1::DataType::kHALF;
  } else if (src.code_ == base::kDataTypeCodeInt && src.bits_ == 32 &&
             src.lanes_ == 1) {
    dst = nvinfer1::DataType::kINT32;
  } else if (src.code_ == base::kDataTypeCodeInt && src.bits_ == 8 &&
             src.lanes_ == 1) {
    dst = nvinfer1::DataType::kINT8;
#ifdef TENSORRT_MAJOR_8_MINOR_5
  } else if (src.code_ == base::kDataTypeCodeUint && src.bits_ == 8 &&
             src.lanes_ == 1) {
    dst = nvinfer1::DataType::kUINT8;
#endif
  } else {
    dst = nvinfer1::DataType::kFLOAT;
  }
  return dst;
}

base::DataFormat TensorRtConvert::convertToDataFormat(
    const nvinfer1::TensorFormat &src) {
  base::DataFormat dst;
  switch (src) {
    case nvinfer1::TensorFormat::kLINEAR:
      dst = base::kDataFormatNCHW;
      break;
    case nvinfer1::TensorFormat::kCHW4:
      dst = base::kDataFormatNC4HW;
      break;
    case nvinfer1::TensorFormat::kHWC:
      dst = base::kDataFormatNHWC;
      break;
    default:
      dst = base::kDataFormatNCHW;
      break;
  }
  return dst;
}

base::IntVector TensorRtConvert::convertToShape(const nvinfer1::Dims &src) {
  base::IntVector dst;
  int src_size = src.nbDims;
  for (int i = 0; i < src_size; ++i) {
    dst.emplace_back(src.d[i]);
  }
  return dst;
}

nvinfer1::Dims TensorRtConvert::convertFromShape(const base::IntVector &src) {
  int src_size = src.size();
  nvinfer1::Dims dst;
  dst.nbDims = src_size;
  for (int i = 0; i < src_size; ++i) {
    dst.d[i] = src[i];
  }
  return dst;
}

}  // namespace inference
}  // namespace nndeploy
