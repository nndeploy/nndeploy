
#include "nndeploy/inference/paddlelite/paddlelite_convert.h"

#include "nndeploy/inference/paddlelite/paddlelite_inference.h"

namespace nndeploy {
namespace inference {

base::DataType PaddleLiteConvert::convertToDataType(
    const paddle::lite_api::PrecisionType &src) {
  base::DataType dst;
  switch (src) {
    case paddle::lite_api::PrecisionType::kFP64:
      dst.code_ = base::kDataTypeCodeFp;
      dst.bits_ = 64;
      dst.lanes_ = 1;
      break;
    case paddle::lite_api::PrecisionType::kFloat:
      dst.code_ = base::kDataTypeCodeFp;
      dst.bits_ = 32;
      dst.lanes_ = 1;
      break;
    case paddle::lite_api::PrecisionType::kFP16:
      dst.code_ = base::kDataTypeCodeBFp;
      dst.bits_ = 16;
      dst.lanes_ = 1;
    case paddle::lite_api::PrecisionType::kInt64:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 64;
      dst.lanes_ = 1;
      break;
    case paddle::lite_api::PrecisionType::kInt32:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 32;
      dst.lanes_ = 1;
      break;
    case paddle::lite_api::PrecisionType::kInt16:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 16;
      dst.lanes_ = 1;
      break;
    case paddle::lite_api::PrecisionType::kInt8:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 8;
      dst.lanes_ = 1;
      break;
    case paddle::lite_api::PrecisionType::kUInt8:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 8;
      dst.lanes_ = 1;
      break;
    case paddle::lite_api::PrecisionType::kBool:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 8;
      dst.lanes_ = 1;
      break;
    case paddle::lite_api::PrecisionType::kAny: 
      dst.code_ = base::kDataTypeCodeOpaqueHandle;
      dst.bits_ = sizeof(size_t) * 8;
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

paddle::lite_api::PrecisionType PaddleLiteConvert::convertFromDataType(
    const base::DataType &src) {
  paddle::lite_api::PrecisionType dst;
  if (src.code_ == base::kDataTypeCodeFp && src.lanes_ == 1) {
    if (src.bits_ == 32) {
      dst = paddle::lite_api::PrecisionType::kFloat;
    } else if (src.bits_ == 64) {
      dst = paddle::lite_api::PrecisionType::kFP64;
    } else {
      dst = paddle::lite_api::PrecisionType::kUnk;
    }
  } else if (src.code_ == base::kDataTypeCodeBFp && src.bits_ == 16 &&
             src.lanes_ == 1) {
    dst = paddle::lite_api::PrecisionType::kFP16;
  } else if (src.code_ == base::kDataTypeCodeInt && src.lanes_ == 1) {
    if (src.bits_ == 8) {
      dst = paddle::lite_api::PrecisionType::kInt8;
    } else if (src.bits_ == 16) {
      dst = paddle::lite_api::PrecisionType::kInt16;
    } else if (src.bits_ == 32) {
      dst = paddle::lite_api::PrecisionType::kInt32;
    } else if (src.bits_ == 64) {
      dst = paddle::lite_api::PrecisionType::kInt64;
    } else {
      dst = paddle::lite_api::PrecisionType::kUnk;
    }
  } else if (src.code_ == base::kDataTypeCodeUint && src.lanes_ == 1) {
    if (src.bits_ == 8) {
      dst = paddle::lite_api::PrecisionType::kUInt8;
    } else {
      dst = paddle::lite_api::PrecisionType::kUnk;
    }
  } else {
    dst = paddle::lite_api::PrecisionType::kUnk;
  }
  return dst;
}

paddle::lite::TargetType PaddleLiteConvert::convertFromDeviceType(const base::DeviceType &src) {
  paddle::lite::TargetType type = paddle::lite::TargetType::kHost;
  switch (src.code_) {
    case base::kDeviceTypeCodeCpu:
      type = paddle::lite::TargetType::kHost;
      break;
    case base::kDeviceTypeCodeX86:
      type = paddle::lite::TargetType::kX86;
      break;
    case base::kDeviceTypeCodeArm:
      type = paddle::lite::TargetType::kARM;
      break;
    case base::kDeviceTypeCodeOpenCL:
      type = paddle::lite::TargetType::kOpenCL;
      break;
    case base::kDeviceTypeCodeMetal:
      type = paddle::lite::TargetType::kMetal;
      break;
    case base::kDeviceTypeCodeCuda:
      type = paddle::lite::TargetType::kCUDA;
      break;
    case base::kDeviceTypeCodeNotSupport:
      type = paddle::lite::TargetType::kUnk;
      break;   
  }
  return type;
}

base::DeviceType PaddleLiteConvert::convertToDeviceType(const paddle::lite::TargetType &src) {
  base::DeviceType type = base::kDeviceTypeCodeNotSupport;
  switch (src) {
    case paddle::lite::TargetType::kHost:
      type = base::kDeviceTypeCodeCpu;
      break;
    case paddle::lite::TargetType::kX86:
      type = base::kDeviceTypeCodeX86;
      break;
    case paddle::lite::TargetType::kARM:
      type = base::kDeviceTypeCodeArm;
      break;
    case paddle::lite::TargetType::kOpenCL:
      type = base::kDeviceTypeCodeOpenCL;
      break;
    case paddle::lite::TargetType::kMetal:
      type = base::kDeviceTypeCodeMetal;
      break;
    case paddle::lite::TargetType::kCUDA:
      type = base::kDeviceTypeCodeCuda;
      break;
    default:
      type = base::kDeviceTypeCodeNotSupport;
      break;
  }
  return type;
}

base::DataFormat PaddleLiteConvert::convertToDataFormat(
    const paddle::lite_api::DataLayoutType &src) {
  base::DataFormat dst;
  switch (src) {
    case paddle::lite_api::DataLayoutType::kNCHW:
      dst = base::kDataFormatNCHW;
      break;
    case paddle::lite_api::DataLayoutType::kImageDefault:
      dst = base::kDataFormatNC4HW;
      break;
    case paddle::lite_api::DataLayoutType::kNHWC:
      dst = base::kDataFormatNHWC;
      break;
    default:
      dst = base::kDataFormatNCHW;
      break;
  }
  return dst;      
}

base::IntVector PaddleLiteConvert::convertToShape(const paddle::lite::DDim &src) {
  base::IntVector dst;
  size_t src_size = src.size();
  for (int i = 0; i < src_size; ++i) {
    dst.push_back(src[i]);
  }
  return dst;
}

paddle::lite::DDim PaddleLiteConvert::convertFromShape(const base::IntVector &src) {
  size_t src_size = src.size();
  paddle::lite::DDim dst;
  for (int i = 0; i < src_size; ++i) {
    dst[i] = src[i];
  }
  return dst;
}

}  // namespace inference
}  // namespace nndeploy
