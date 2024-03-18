
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
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 8;
      dst.lanes_ = 1;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 8;
      dst.lanes_ = 1;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 16;
      dst.lanes_ = 1;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 16;
      dst.lanes_ = 1;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 32;
      dst.lanes_ = 1;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 64;
      dst.lanes_ = 1;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 8;
      dst.lanes_ = 1;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      dst.code_ = base::kDataTypeCodeFp;
      dst.bits_ = 16;
      dst.lanes_ = 1;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      dst.code_ = base::kDataTypeCodeFp;
      dst.bits_ = 64;
      dst.lanes_ = 1;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 32;
      dst.lanes_ = 1;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      dst.code_ = base::kDataTypeCodeUint;
      dst.bits_ = 64;
      dst.lanes_ = 1;
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      dst.code_ = base::kDataTypeCodeBFp;
      dst.bits_ = 16;
      dst.lanes_ = 1;
      break;
    default:
      dst.code_ = base::kDataTypeCodeOpaqueHandle;
      dst.bits_ = sizeof(size_t) * 8;
      dst.lanes_ = 1;
      break;
  }
  return dst;
}

ONNXTensorElementDataType OnnxRuntimeConvert::convertFromDataType(
    const base::DataType &src) {
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

base::IntVector OnnxRuntimeConvert::convertToShape(std::vector<int64_t> &src,
                                                   base::IntVector max_shape) {
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

std::vector<int64_t> OnnxRuntimeConvert::convertFromShape(
    const base::IntVector &src) {
  int src_size = src.size();
  std::vector<int64_t> dst;
  for (int i = 0; i < src_size; ++i) {
    dst.emplace_back(static_cast<int64_t>(src[i]));
  }
  return dst;
}

base::Status OnnxRuntimeConvert::convertFromInferenceParam(
    OnnxRuntimeInferenceParam &src, Ort::SessionOptions &dst) {
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
  // dst.DisableCpuMemArena();

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
      NNDEPLOY_LOGE("doesn't support GPU.\n");
      src.device_type_ = device::getDefaultHostDeviceType();
    } else {
      OrtCUDAProviderOptions cuda_srcs;
      cuda_srcs.device_id = src.device_type_.device_id_;
      device::Device *device = device::getDevice(src.device_type_);
      if (device) {
        cuda_srcs.has_user_compute_stream = 1;
        cuda_srcs.user_compute_stream = device->getCommandQueue();
      }
      dst.AppendExecutionProvider_CUDA(cuda_srcs);
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
                                                 const std::string &name,
                                                 device::Device *device,
                                                 device::Tensor *dst) {
  base::Status status = base::kStatusCodeOk;

  bool copy_flag = false;
  if (!dst->empty() &&
      dst->getBufferSourceType() == device::kBufferSourceTypeAllocate) {
    copy_flag = true;
  }

  const auto src_info = src.GetTensorTypeAndShapeInfo();
  const auto src_data_type = src_info.GetElementType();
  size_t src_numel = src_info.GetElementCount();
  auto src_shape = src_info.GetShape();

  const auto dst_data_type =
      OnnxRuntimeConvert::convertToDataType(src_data_type);
  const auto elesize = dst_data_type.size();

  size_t src_size = src_numel * elesize;
  const void *value_ptr = src.GetTensorData<void *>();
  if (copy_flag) {
    device::Buffer *src_buffer = device->create(src_size, (void *)value_ptr);
    device::Buffer *dst_buffer = dst->getBuffer();
    device->copy(src_buffer, dst_buffer);
    device->deallocate(src_buffer);
  } else {
    dst->destory();
    device::TensorDesc desc;
    desc.shape_ = OnnxRuntimeConvert::convertToShape(src_shape);
    desc.data_type_ = dst_data_type;
    desc.data_format_ = OnnxRuntimeConvert::getDataFormatByShape(desc.shape_);
    dst->create(device, desc, (void *)value_ptr, name);
  }

  return status;
}

Ort::Value OnnxRuntimeConvert::convertFromTensor(device::Tensor *src) {
  base::DeviceType device_type = src->getDeviceType();
  auto src_data_type = src->getDataType();
  ONNXTensorElementDataType dst_data_type =
      OnnxRuntimeConvert::convertFromDataType(src_data_type);
  if (device_type == base::kDeviceTypeCodeCuda) {
    Ort::MemoryInfo memory_info("Cuda", OrtDeviceAllocator, 0,
                                OrtMemTypeDefault);
    auto dst_shape = convertFromShape(src->getShape());
    auto ort_value = Ort::Value::CreateTensor(
        memory_info, src->getPtr(), src->getSize(), dst_shape.data(),
        src->getShape().size(), dst_data_type);
    return ort_value;
  } else if (device::isHostDeviceType(device_type)) {
    Ort::MemoryInfo memory_info("Cpu", OrtDeviceAllocator, 0,
                                OrtMemTypeDefault);
    auto dst_shape = convertFromShape(src->getShape());
    auto ort_value = Ort::Value::CreateTensor(
        memory_info, src->getPtr(), src->getSize(), dst_shape.data(),
        src->getShape().size(), dst_data_type);
    return ort_value;
  } else {
    return Ort::Value(nullptr);
  }
}

}  // namespace inference
}  // namespace nndeploy
