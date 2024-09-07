#include "nndeploy/inference/snpe/snpe_convert.h"

#include "nndeploy/inference/snpe/snpe_include.h"


namespace nndeploy {
namespace inference {

base::DataType SnpeConvert::convertToDataType(SnpeBuffer_Type_t &src) {
  base::DataType dst;

  switch (src) {
    case USERBUFFER_FLOAT:
      dst.code_ = base::kDataTypeCodeFp;
      dst.bits_ = 32;
      dst.lanes_ = 1;
      break;
    case USERBUFFER_TF8:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 8;
      dst.lanes_ = 1;
      break;
    case ITENSOR:
      dst.code_ = base::kDataTypeCodeFp;
      dst.bits_ = 32;  // 32 or 8
      dst.lanes_ = 1;
      break;
    case USERBUFFER_TF16:
      dst.code_ = base::kDataTypeCodeBFp;
      dst.bits_ = 16;
      dst.lanes_ = 1;
      break;
    default:
      break;
  }

  return dst;
}

base::DataFormat SnpeConvert::convertToDataFormat() {
  base::DataFormat dst;
  dst = base::kDataFormatNHWC;
  return dst;
}

base::IntVector SnpeConvert::convertToShape(
    const zdl::DlSystem::Dimension *dims, size_t rank) {
  base::IntVector dst;

  while (rank--) {
    dst.emplace_back(*dims);
    dims++;
  }

  return dst;
}

}  // namespace inference
}  // namespace nndeploy