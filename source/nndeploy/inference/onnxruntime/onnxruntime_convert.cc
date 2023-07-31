
#include "nndeploy/inference/onnxruntime/onnxruntime_convert.h"

#include "nndeploy/inference/onnxruntime/onnxruntime_inference.h"

namespace nndeploy {
namespace inference {

base::DataType OnnxRuntimeConvert::convertToDataType(
    const ONNXTensorElementDataType &src) {
  base::DataType dst;
  switch (src) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      dst.code_ = base::kDataTypeCodeFp;
      dst.bits_ = 32;
      dst.lanes_ = 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 8;
      dst.lanes_ = 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 8;
      dst.lanes_ = 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 16;
      dst.lanes_ = 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 16;
      dst.lanes_ = 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 32;
      dst.lanes_ = 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 64;
      dst.lanes_ = 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 8;
      dst.lanes_ = 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      dst.code_ = base::kDataTypeCodeFp;
      dst.bits_ = 16;
      dst.lanes_ = 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      dst.code_ = base::kDataTypeCodeFp;
      dst.bits_ = 64;
      dst.lanes_ = 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 32;
      dst.lanes_ = 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 64;
      dst.lanes_ = 1;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      dst.code_ = base::kDataTypeCodeBFp;
      dst.bits_ = 16;
      dst.lanes_ = 1;
    default:
      dst.code_ = base::kDataTypeCodeOpaqueHandle;
      dst.bits_ = sizeof(size_t) * 8;
      dst.lanes_ = 1;
  }
  return dst;
}

ONNXTensorElementDataType OnnxRuntimeConvert::convertFromDataType(
    base::DataType &src) {
  ONNXTensorElementDataType dst;
  if (src.code_ == base::kDataTypeCodeFp && src.lanes_ == 1) {
    if (src.bits_ == 16) {
      dst = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    } else if (src.bits_ == 32) {
      dst = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    } else if (src.bits_ == 64) {
      dst = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    } else {
      dst = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
  } else if (src.code_ == base::kDataTypeCodeBFp && src.bits_ == 16 &&
             src.lanes_ == 1) {
    dst = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
  } else if (src.code_ == base::kDataTypeCodeInt && src.lanes_ == 1) {
    if (src.bits_ == 8) {
      dst = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    } else if (src.bits_ == 16) {
      dst = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    } else if (src.bits_ == 32) {
      dst = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    } else if (src.bits_ == 64) {
      dst = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    } else {
      dst = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
  } else if (src.code_ == base::kDataTypeCodeUint && src.lanes_ == 1) {
    if (src.bits_ == 8) {
      dst = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    } else if (src.bits_ == 16) {
      dst = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    } else if (src.bits_ == 32) {
      dst = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    } else if (src.bits_ == 64) {
      dst = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    } else {
      dst = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
  } else {
    dst = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
  return dst;
}

base::DataFormat OnnxRuntimeConvert::getDataFormatByShape(
    const base::IntVector &src) {
  base::DataFormat dst = base::kDataFormatNotSupport;
  if (src.size() == 5) {
    dst = base::kDataFormatNCDHW;
  } else if (src.size() == 4) {
    dst = base::kDataFormatNCHW;
  } else if (src.size() == 3) {
    dst = base::kDataFormatNCHW;
  } else if (src.size() == 2) {
    dst = base::kDataFormatNC;
  } else if (src.size() == 1) {
    dst = base::kDataFormatScalar;
  } else {
    dst = base::kDataFormatNotSupport;
  }
  return dst;
}

base::IntVector OnnxRuntimeConvert::convertToShape(std::vector<int64_t> &src,
                                                   base::IntVector max_shape) {
  base::IntVector dst;
  if (max_shape.empty()) {
    dst = max_shape;
  } else {
    std::vector<int64_t> src_shape = src;
    int src_size = src_shape.size();
    for (int i = 0; i < src_size; ++i) {
      dst[i] = (int)src_shape[i];
      if (dst[i] == -1) {
        dst[i] = 1;
      }
    }
  }
  return dst;
}

std::vector<int64_t> OnnxRuntimeConvert::convertFromShape(
    const base::IntVector &src) {
  int src_size = src.size();
  std::vector<int64_t> dst;
  for (int i = 0; i < src_size; ++i) {
    dst[i] = (int64_t)src[i];
  }
  return dst;
}

base::Status OnnxRuntimeConvert::convertFromInferenceParam(
    OnnxRuntimeInferenceParam *src, Ort::SessionOptions *dst) {
  base::Status status = base::kStatusCodeOk;
  return status;
}

base::Status OnnxRuntimeConvert::convertToTensor(Ort::Value &src,
                                                 std::string name,
                                                 device::Device *device,
                                                 device::Tensor *dst,
                                                 bool flag) {
  base::Status status = base::kStatusCodeOk;
  return status;
}

Ort::Value OnnxRuntimeConvert::convertFromTensor(device::Tensor *src) {
  Ort::Value value(nullptr);
  return value;
}

}  // namespace inference
}  // namespace nndeploy
