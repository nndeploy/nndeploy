
#include "nndeploy/inference/mdc/mdc_convert.h"

namespace nndeploy {
namespace inference {

base::DataType MdcConvert::convertToDataType(const aclDataType &src) {
  base::DataType dst;
  switch (src) {
    case ACL_FLOAT:
      dst.code_ = base::kDataTypeCodeFp;
      dst.bits_ = 32;
      dst.lanes_ = 1;
      break;
    case ACL_FLOAT16:
      dst.code_ = base::kDataTypeCodeFp;
      dst.bits_ = 16;
      dst.lanes_ = 1;
    case ACL_INT64:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 64;
      dst.lanes_ = 1;
      break;
    case ACL_INT32:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 32;
      dst.lanes_ = 1;
      break;
    case ACL_INT16:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 16;
      dst.lanes_ = 1;
      break;
    case ACL_INT8:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 8;
      dst.lanes_ = 1;
      break;

    case ACL_UINT64:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 64;
      dst.lanes_ = 1;
      break;
    case ACL_UINT32:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 32;
      dst.lanes_ = 1;
      break;
    case ACL_UINT16:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 16;
      dst.lanes_ = 1;
      break;
    case ACL_UINT8:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 8;
      dst.lanes_ = 1;
      break;
    case ACL_BOOL:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 8;
      dst.lanes_ = 1;
      break;
    case ACL_DT_UNDEFINED:  // 未知数据类型，默认值
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

aclDataType MdcConvert::convertFromDataType(const base::DataType &src) {
  aclDataType dst;
  if (src.code_ == base::kDataTypeCodeFp && src.lanes_ == 1) {
    if (src.bits_ == 32) {
      dst = ACL_FLOAT;
    } else {
      dst = ACL_DT_UNDEFINED;
    }
  } else if (src.code_ == base::kDataTypeCodeFp && src.bits_ == 16 && src.lanes_ == 1) {
    dst = ACL_FLOAT16;
  } else if (src.code_ == base::kDataTypeCodeInt && src.lanes_ == 1) {
    if (src.bits_ == 8) {
      dst = ACL_INT8;
    } else if (src.bits_ == 16) {
      dst = ACL_INT16;
    } else if (src.bits_ == 32) {
      dst = ACL_INT32;
    } else if (src.bits_ == 64) {
      dst = ACL_INT64;
    } else {
      dst = ACL_DT_UNDEFINED;
    }
  } else if (src.code_ == base::kDataTypeCodeUint && src.lanes_ == 1) {
    if (src.bits_ == 8) {
      dst = ACL_UINT8;
    } else if (src.bits_ == 16) {
      dst = ACL_UINT16;
    } else if (src.bits_ == 32) {
      dst = ACL_UINT32;
    } else if (src.bits_ == 64) {
      dst = ACL_UINT64;
    } else {
      dst = ACL_DT_UNDEFINED;
    }
  } else {
    dst = ACL_DT_UNDEFINED;
  }
  return dst;
}

base::DataFormat MdcConvert::getDataFormatByShape(const base::IntVector &src) {
  base::DataFormat dst = base::kDataFormatNotSupport;
  if (src.size() == 5) {
    dst = base::kDataFormatNCDHW;
  } else if (src.size() == 4) {
    dst = base::kDataFormatNCHW;
  } else if (src.size() == 3) {
    dst = base::kDataFormatNHW;
  } else if (src.size() == 2) {
    dst = base::kDataFormatNC;
  } else if (src.size() == 1) {
    dst = base::kDataFormatN;
  } else {
    dst = base::kDataFormatNotSupport;
  }
  return dst;
}

base::IntVector MdcConvert::convertToShape(std::vector<int64_t> &src, base::IntVector max_shape) {
  base::IntVector dst;
  if (!max_shape.empty()) {
    dst = max_shape;
  } else {
    int src_size = src.size();
    for (int i = 0; i < src_size; ++i) {
      dst.emplace_back(static_cast<int>(src[i]));
    }
  }
  return dst;
}

std::vector<int64_t> MdcConvert::convertFromShape(const base::IntVector &src) {
  int src_size = src.size();
  std::vector<int64_t> dst;
  for (int i = 0; i < src_size; ++i) {
    dst.emplace_back(static_cast<int64_t>(src[i]));
  }
  return dst;
}

}  // namespace inference
}  // namespace nndeploy
