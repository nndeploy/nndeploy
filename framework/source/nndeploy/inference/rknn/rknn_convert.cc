
#include "nndeploy/inference/rknn/rknn_convert.h"

namespace nndeploy {
namespace inference {

base::DataType RknnConvert::convertToDataType(const rknn_tensor_type &src) {
  base::DataType dst;
  switch (src) {
    case RKNN_TENSOR_FLOAT32:
      dst.code_ = base::kDataTypeCodeFp;
      dst.bits_ = 32;
      dst.lanes_ = 1;
      break;
    case RKNN_TENSOR_FLOAT16:
      dst.code_ = base::kDataTypeCodeBFp;
      dst.bits_ = 16;
      dst.lanes_ = 1;
      break;
    case RKNN_TENSOR_INT8:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 8;
      dst.lanes_ = 1;
      break;
    case RKNN_TENSOR_UINT8:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 8;
      dst.lanes_ = 1;
      break;
    case RKNN_TENSOR_INT16:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 16;
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

base::DataFormat RknnConvert::convertToDataFormat(
    const rknn_tensor_format &src) {
  base::DataFormat dst;
  switch (src) {
    case RKNN_TENSOR_NCHW:
      dst = base::kDataFormatNCHW;
      break;
    case RKNN_TENSOR_NHWC:
      dst = base::kDataFormatNHWC;
      break;
    default:
      dst = base::kDataFormatNCHW;
      break;
  }
  return dst;
}

base::IntVector RknnConvert::convertToShape(const rknn_tensor_attr &src,
                                            const rknn_tensor_format &dst_fmt) {
  base::IntVector dst;
  for (int i = 0; i < src.n_dims; i++) {
    dst.push_back(int(src.dims[i]));
  }
#ifdef RKNN_TOOLKIT_1
  std::reverse(dst.begin(), dst.end());
#endif
  switch (dst_fmt) {
    case RKNN_TENSOR_FORMAT_MAX:
      break;
    case RKNN_TENSOR_NHWC:
      if (src.fmt == RKNN_TENSOR_NCHW) {
        // nchw -> nhwc
        if (src.n_dims > 3) {
          std::swap(dst[1], dst[3]);
          std::swap(dst[1], dst[2]);
        }
      }
      break;
    case RKNN_TENSOR_NCHW:
      if (src.fmt == RKNN_TENSOR_NHWC) {
        // nhwc -> nchw
        if (src.n_dims > 3) {
          std::swap(dst[1], dst[3]);
          std::swap(dst[2], dst[3]);
        }
      }
      break;
  }
  return dst;
}

}  // namespace inference
}  // namespace nndeploy