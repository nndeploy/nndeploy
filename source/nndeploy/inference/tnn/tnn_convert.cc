
#include "nndeploy/inference/tnn/tnn_convert.h"

namespace nndeploy {
namespace inference {

base::DataType TnnConvert::convertToDataType(const TNN_NS::DataType &src) {
  base::DataType dst;
  switch (src) {
    case TNN_NS::DATA_TYPE_INT8:
    case TNN_NS::DATA_TYPE_INT32:
    case TNN_NS::DATA_TYPE_INT64:
      dst.code_ = base::kDataTypeCodeInt;
      break;
    case TNN_NS::DATA_TYPE_UINT32:
      dst.code_ = base::kDataTypeCodeUint;
      break;
    case TNN_NS::DATA_TYPE_FLOAT:
      dst.code_ = base::kDataTypeCodeFp;
      break;
    case TNN_NS::DATA_TYPE_BFP16:
      dst.code_ = base::kDataTypeCodeBFp;
      break;
    default:
      dst.code_ = base::kDataTypeCodeOpaqueHandle;
      break;
  }
  dst.bits_ = DataTypeUtils::GetBytesSize(src) * 8;
  dst.lanes_ = 1;
  return dst;
}

halide_type_t TnnConvert::convertFromDataType(base::DataType &src) {
  TNN_NS::DataType dst;
  if (src.code_ == base::kDataTypeCodeInt && src.bits_ == 8 &&
      src.lanes_ == 1) {
    dst = TNN_NS::DATA_TYPE_INT8;
  } else if (src.code_ == base::kDataTypeCodeInt && src.bits_ == 32 &&
             src.lanes_ == 1) {
    dst = TNN_NS::DATA_TYPE_INT32;
  } else if (src.code_ == base::kDataTypeCodeInt && src.bits_ == 64 &&
             src.lanes_ == 1) {
    dst = TNN_NS::DATA_TYPE_INT64;
  } else if (src.code_ == base::kDataTypeCodeUInt && src.bits_ == 32 &&
             src.lanes_ == 1) {
    dst = TNN_NS::DATA_TYPE_UINT32;
  } else if (src.code_ == base::kDataTypeCodeFp && src.bits_ == 32 &&
             src.lanes_ == 1) {
    dst = TNN_NS::DATA_TYPE_FLOAT;
  } else if (src.code_ == base::kDataTypeCodeBFp && src.bits_ == 16 &&
             src.lanes_ == 1) {
    dst = TNN_NS::DATA_TYPE_BFP16;
  } else if (src.code_ == base::kDataTypeCodeFp && src.bits_ == 16 &&
             src.lanes_ == 1) {
    dst = TNN_NS::DATA_TYPE_HALF;
  } else {
    dst = TNN_NS::DATA_TYPE_FLOAT;
  }
  return dst;
}

base::DataFormat TnnConvert::convertToDataFormat(
    const TNN_NS::DataFormat &src) {
  base::DataFormat dst;
  switch (src) {
    case TNN::Tensor::TENSORFLOW:
      dst = base::kDataFormatNHWC;
      break;
    case TNN::Tensor::CAFFE:
      dst = base::kDataFormatNCHW;
      break;
    case TNN::Tensor::CAFFE_C4:
      dst = base::kDataFormatNC4HW;
      break;
    default:
      dst = base::kDataFormatNotSupport;
      break;
  }
  return dst;
}

TNN::Tensor::DimensionType TnnConvert::convertFromDataFormat(
    const base::DataFormat &src) {
  TNN::Tensor::DimensionType dst = TNN::Tensor::CAFFE;
  switch (src) {
    case base::kDataFormatNCHW:
      dst = TNN::Tensor::CAFFE;
      break;
    case base::kDataFormatNHWC:
      dst = TNN::Tensor::TENSORFLOW;
      break;
    case base::kDataFormatNC4HW:
      dst = TNN::Tensor::CAFFE_C4;
      break;
    default:
      dst = TNN::Tensor::CAFFE;
      break;
  }
  return dst;
}

TNNForwardType TnnConvert::convertFromDeviceType(const base::DeviceType &src) {
  TNNForwardType type = TNN_FORWARD_CPU;
  switch (src.code_) {
    case base::kDeviceTypeCodeCpu:
      type = TNN_FORWARD_CPU;
      break;
    case base::kDeviceTypeCodeX86:
      type = TNN_FORWARD_CPU;
      break;
    case base::kDeviceTypeCodeArm:
      type = TNN_FORWARD_CPU;
      break;
    case base::kDeviceTypeCodeOpenCL:
      type = TNN_FORWARD_OPENCL;
      break;
    case base::kDeviceTypeCodeOpenGL:
      type = TNN_FORWARD_OPENGL;
      break;
    case base::kDeviceTypeCodeMetal:
      type = TNN_FORWARD_METAL;
      break;
    case base::kDeviceTypeCodeCuda:
      type = TNN_FORWARD_CUDA;
      break;
    default:
      type = TNN_FORWARD_CPU;
      break;
  }
  return type;
}

TNN::BackendConfig::PowerMode TnnConvert::convertFromPowerType(
    const base::PowerType &src) {
  switch (src) {
    case base::kPowerTypeLow:
      return TNN::BackendConfig::PowerMode::Power_Low;
    case base::kPowerTypeNormal:
      return TNN::BackendConfig::PowerMode::Power_Normal;
    case base::kPowerTypeHigh:
      return TNN::BackendConfig::PowerMode::Power_High;
    default:
      return TNN::BackendConfig::PowerMode::Power_Normal;
  }
}

TNN::BackendConfig::PrecisionMode TnnConvert::convertFromPrecisionType(
    const base::PrecisionType &src) {
  switch (src) {
    case base::kPrecisionTypeFp16:
      return TNN::BackendConfig::PrecisionMode::Precision_Low;
    case base::kPrecisionTypeBFp16:
      return TNN::BackendConfig::PrecisionMode::Precision_Low;
    case base::kPrecisionTypeFp32:
      return TNN::BackendConfig::PrecisionMode::Precision_Normal;
    case base::kPrecisionTypeFp64:
      return TNN::BackendConfig::PrecisionMode::Precision_High;
    default:
      return TNN::BackendConfig::PrecisionMode::Precision_Normal;
  }
}

base::Status TnnConvert::convertFromInferenceParam(TnnInferenceParam *src,
                                                   TNN::ScheduleConfig *dst) {
  if (src == nullptr || dst == nullptr) {
    return base::kStatusCodeErrorInvalidParam;
  }

  dst->saveTensors = src->save_tensors_;
  dst->type = convertFromDeviceType(src->device_type_);
  if (dst->type == TNN_FORWARD_CPU) {
    dst->numThread = src->num_thread_;
  } else if (dst->type == TNN_FORWARD_OPENCL ||
             dst->type == TNN_FORWARD_OPENGL ||
             dst->type == TNN_FORWARD_METAL || dst->type == TNN_FORWARD_CUDA) {
    dst->mode = src->gpu_tune_kernel_;
  }
  dst->path = src->path_;
  dst->backupType = convertFromDeviceType(src->backup_device_type_);

  dst->backendConfig = new TNN::BackendConfig();
  dst->backendConfig->power = convertFromPowerType(src->power_type_);
  dst->backendConfig->precision =
      convertFromPrecisionType(src->precision_type_);
  dst->backendConfig->memory = src->memory_mode_;

  return base::kStatusCodeOk;
}

device::Tensor *TnnConvert::convertToTensor(TNN::Tensor *src, std::string name,
                                            device::Device *device) {
  halide_type_t src_data_type = src->getType();
  base::DataType data_type = TnnConvert::convertToDataType(src_data_type);
  TNN::Tensor::DimensionType src_data_format = src->getDimensionType();
  base::DataFormat format = TnnConvert::convertToDataFormat(src_data_format);
  base::IntVector shape = src->shape();
  base::SizeVector stride = base::SizeVector();
  device::TensorDesc desc(data_type, format, shape, stride);
  device::Tensor *dst = nullptr;
  if (device == nullptr) {
    dst = new device::Tensor(desc, name);
  } else {
    auto src_buffer = src->buffer();
    void *data_ptr = (void *)src_buffer.host;
    base::IntVector memory_config = base::IntVector();
    dst = new device::Tensor(device, desc, data_ptr, name, memory_config);
  }
  return dst;
}

TNN::Tensor *TnnConvert::convertFromTensor(device::Tensor *src) {
  if (!device::isHostDeviceType(src->getDeviceType())) {
    return nullptr;
  }
  device::TensorDesc desc = src->getDesc();
  std::vector<int> shape = desc.shape_;
  halide_type_t type = TnnConvert::convertFromDataType(desc.data_type_);
  void *data = src->getPtr();
  TNN::Tensor::DimensionType dimType =
      TnnConvert::convertFromDataFormat(desc.format_);
  TNN::Tensor *dst = TNN::Tensor::create(shape, type, data, dimType);
  return dst;
}

}  // namespace inference
}  // namespace nndeploy
