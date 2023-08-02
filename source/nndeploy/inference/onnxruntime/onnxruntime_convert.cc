
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
    int src_size = src.size();
    for (int i = 0; i < src_size; ++i) {
      dst[i] = (int)src[i];
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
    const OnnxRuntimeInferenceParam &src, Ort::SessionOptions &dst) {
  base::Status status = base::kStatusCodeOk;
  if (src.graph_optimization_level_ >= 0) {
    dst.SetGraphOptimizationLevel(
        GraphOptimizationLevel(src.graph_optimization_level_));
  }
  if (src.inter_op_num_threads_ > 0) {
    dst.SetInterOpNumThreads(src.inter_op_num_threads_);
  }
  if (src.execution_mode_ >= 0) {
    dst.SetExecutionMode(ExecutionMode(src.execution_mode_));
  }

  if (src.device_type_ == base::kDeviceTypeCodeCuda) {
    auto all_providers = Ort::GetAvailableProviders();
    bool support_cuda = false;
    std::string providers_msg = "";
    for (size_t i = 0; i < all_providers.size(); ++i) {
      providers_msg = providers_msg + all_providers[i] + ", ";
      if (all_providers[i] == "CUDAExecutionProvider") {
        support_cuda = true;
      }
    }
    if (!support_cuda) {
      NNDEPLOY_LOGE(
          "Compiled fastdeploy with onnxruntime doesn't "
          "support GPU, the available providers are %s, "
          "will fallback to CPUExecutionProvider.\n",
          providers_msg);
      src.device_type_ = device::getHostDeviceType();
    } else {
      OrtCUDAProviderOptions cuda_srcs;
      cuda_srcs.device_id = src.device_type_.device_id_;
      device::Device *device = device::getDevice(src.device_type_);
      if (device) {
        cuda_srcs.has_user_compute_stream = 1;
        cuda_srcs.user_compute_stream = src.external_stream_;
      }
      session_srcs_.AppendExecutionProvider_CUDA(cuda_srcs);
    }
  }

  if (device::isHostDeviceType(src.device_type_)) {
    if (src.num_thread_ > 0) {
      dst.SetIntraOpNumThreads(src.num_thread_);
    }
  }

  return status;
}

base::Status OnnxRuntimeConvert::convertToTensor(Ort::Value &src,
                                                 std::string name,
                                                 device::Device *device,
                                                 device::Tensor *dst)

{
  base::Status status = base::kStatusCodeOk;

  const auto info = src.GetTensorTypeAndShapeInfo();
  const auto data_type = info.GetElementType();
  size_t numel = info.GetElementCount();
  auto shape = info.GetShape();
  const void *value_ptr = value.GetTensorData<void *>();
  if (dst->empty()) {
  } else {
  }

  return status;
}

Ort::Value OnnxRuntimeConvert::convertFromTensor(device::Tensor *src) {
  Ort::Value value(nullptr);
  return value;
}

}  // namespace inference
}  // namespace nndeploy
