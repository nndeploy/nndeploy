
#include "nndeploy/inference/mnn/mnn_convert.h"

namespace nndeploy {
namespace inference {

base::DataType MnnConvert::convertToDataType(const halide_type_t &src) {
  base::DataType dst;
  switch (src.code) {
    case halide_type_int:
      dst.code_ = base::kDataTypeCodeInt;
      break;
    case halide_type_uint:
      dst.code_ = base::kDataTypeCodeUint;
      break;
    case halide_type_float:
      dst.code_ = base::kDataTypeCodeFp;
      break;
    case halide_type_handle:
      dst.code_ = base::kDataTypeCodeOpaqueHandle;
      break;
    default:
      dst.code_ = base::kDataTypeCodeOpaqueHandle;
      break;
  }
  dst.bits_ = src.bits;
  dst.lanes_ = src.lanes;
  return dst;
}
halide_type_t MnnConvert::convertFromDataType(const base::DataType &src) {
  halide_type_t dst;
  switch (src.code_) {
    case base::kDataTypeCodeInt:
      dst.code = halide_type_int;
      break;
    case base::kDataTypeCodeUint:
      dst.code = halide_type_uint;
      break;
    case base::kDataTypeCodeFp:
      dst.code = halide_type_float;
      break;
    default:
      dst.code = halide_type_handle;
      break;
  }
  dst.bits = src.bits_;
  dst.lanes = src.lanes_;
  return dst;
}

base::DataFormat MnnConvert::convertToDataFormat(
    const MNN::Tensor::DimensionType &src) {
  base::DataFormat dst;
  switch (src) {
    case MNN::Tensor::TENSORFLOW:
      dst = base::kDataFormatNHWC;
      break;
    case MNN::Tensor::CAFFE:
      dst = base::kDataFormatNCHW;
      break;
    case MNN::Tensor::CAFFE_C4:
      dst = base::kDataFormatNC4HW;
      break;
    default:
      dst = base::kDataFormatNotSupport;
      break;
  }
  return dst;
}
base::DataFormat MnnConvert::convertToDataFormat(
  const MNN::Express::Dimensionformat &src) {
base::DataFormat dst;
switch (src) {
  case MNN::Express::Dimensionformat::NHWC:
    dst = base::kDataFormatNHWC;
    break;
  case MNN::Express::Dimensionformat::NCHW:
    dst = base::kDataFormatNCHW;
    break;
  case MNN::Express::Dimensionformat::NC4HW4:
    dst = base::kDataFormatNC4HW;
    break;
  default:
    dst = base::kDataFormatNotSupport;
    break;
}
return dst;
}
MNN::Tensor::DimensionType MnnConvert::convertFromDataFormat(
    const base::DataFormat &src) {
  MNN::Tensor::DimensionType dst = MNN::Tensor::CAFFE;
  switch (src) {
    case base::kDataFormatNCHW:
      dst = MNN::Tensor::CAFFE;
      break;
    case base::kDataFormatNHWC:
      dst = MNN::Tensor::TENSORFLOW;
      break;
    case base::kDataFormatNC4HW:
      dst = MNN::Tensor::CAFFE_C4;
      break;
    default:
      dst = MNN::Tensor::CAFFE;
      break;
  }
  return dst;
}

MNNForwardType MnnConvert::convertFromDeviceType(const base::DeviceType &src) {
  MNNForwardType type = MNN_FORWARD_CPU;
  switch (src.code_) {
    case base::kDeviceTypeCodeCpu:
      type = MNN_FORWARD_CPU;
      break;
    case base::kDeviceTypeCodeX86:
      type = MNN_FORWARD_CPU;
      break;
    case base::kDeviceTypeCodeArm:
      type = MNN_FORWARD_CPU;
      break;
    case base::kDeviceTypeCodeOpenCL:
      type = MNN_FORWARD_OPENCL;
      break;
    case base::kDeviceTypeCodeOpenGL:
      type = MNN_FORWARD_OPENGL;
      break;
    case base::kDeviceTypeCodeMetal:
      type = MNN_FORWARD_METAL;
      break;
    case base::kDeviceTypeCodeCuda:
      type = MNN_FORWARD_CUDA;
      break;
    default:
      type = MNN_FORWARD_CPU;
      break;
  }
  return type;
}

MNN::BackendConfig::PowerMode MnnConvert::convertFromPowerType(
    const base::PowerType &src) {
  switch (src) {
    case base::kPowerTypeLow:
      return MNN::BackendConfig::PowerMode::Power_Low;
    case base::kPowerTypeNormal:
      return MNN::BackendConfig::PowerMode::Power_Normal;
    case base::kPowerTypeHigh:
      return MNN::BackendConfig::PowerMode::Power_High;
    default:
      return MNN::BackendConfig::PowerMode::Power_Normal;
  }
}

MNN::BackendConfig::PrecisionMode MnnConvert::convertFromPrecisionType(
    const base::PrecisionType &src) {
  switch (src) {
    case base::kPrecisionTypeFp16:
      return MNN::BackendConfig::PrecisionMode::Precision_Low;
    case base::kPrecisionTypeBFp16:
      return MNN::BackendConfig::PrecisionMode::Precision_Low;
    case base::kPrecisionTypeFp32:
      return MNN::BackendConfig::PrecisionMode::Precision_Normal;
    case base::kPrecisionTypeFp64:
      return MNN::BackendConfig::PrecisionMode::Precision_High;
    default:
      return MNN::BackendConfig::PrecisionMode::Precision_Normal;
  }
}

base::Status MnnConvert::convertFromInferenceParam(MnnInferenceParam *src,
                                                   MNN::ScheduleConfig *dst) {
  if (src == nullptr || dst == nullptr) {
    return base::kStatusCodeErrorInvalidParam;
  }
  dst->saveTensors = src->save_tensors_;
  dst->type = convertFromDeviceType(src->device_type_);
  if (dst->type == MNN_FORWARD_CPU) {
    dst->numThread = src->num_thread_;
  } else if (dst->type == MNN_FORWARD_OPENCL ||
             dst->type == MNN_FORWARD_OPENGL ||
             dst->type == MNN_FORWARD_METAL || dst->type == MNN_FORWARD_CUDA) {
    dst->mode = src->gpu_tune_kernel_;
  }
  dst->path = src->path_;
  dst->backupType = convertFromDeviceType(src->backup_device_type_);

  dst->backendConfig = new MNN::BackendConfig();
  dst->backendConfig->power = convertFromPowerType(src->power_type_);
  dst->backendConfig->precision =
      convertFromPrecisionType(src->precision_type_);
  dst->backendConfig->memory = src->memory_mode_;

  return base::kStatusCodeOk;
}

device::Tensor *MnnConvert::convertToTensor(MNN::Tensor *src, std::string name,
                                            device::Device *device) {
  halide_type_t src_data_type = src->getType();
  base::DataType data_type = MnnConvert::convertToDataType(src_data_type);
  MNN::Tensor::DimensionType src_data_format = src->getDimensionType();
  base::DataFormat format = MnnConvert::convertToDataFormat(src_data_format);
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

MNN::Tensor *MnnConvert::convertFromTensor(device::Tensor *src) {
  if (!device::isHostDeviceType(src->getDeviceType())) {
    return nullptr;
  }
  device::TensorDesc desc = src->getDesc();
  std::vector<int> shape = desc.shape_;
  halide_type_t type = MnnConvert::convertFromDataType(desc.data_type_);
  void *data = src->getData();
  MNN::Tensor::DimensionType dimType =
      MnnConvert::convertFromDataFormat(desc.data_format_);
  MNN::Tensor *dst = MNN::Tensor::create(shape, type, data, dimType);
  return dst;
}

// TODO: 内存拷贝，异构内存拷贝
device::Tensor *convertToTensor(const MNN::Express::VARP &var, std::string name,
                                device::Device *device, bool is_copy) {
  // auto src = var.get()->getTensor();
  // if (src == nullptr) {
  //   return nullptr;
  // }

  // halide_type_t src_data_type = src->getType();
  // base::DataType data_type = MnnConvert::convertToDataType(src_data_type);
  // MNN::Tensor::DimensionType src_data_format = src->getDimensionType();
  // base::DataFormat format = MnnConvert::convertToDataFormat(src_data_format);
  // base::IntVector shape = src->shape();
  // base::SizeVector stride = base::SizeVector();
  // device::TensorDesc desc(data_type, format, shape, stride);
  // device::Tensor *dst = nullptr;
  // if (device == nullptr) {
  //   dst = new device::Tensor(desc, name);
  // } else {
  //   auto src_buffer = src->buffer();
  //   void *data_ptr = (void *)src_buffer.host;
  //   base::IntVector memory_config = base::IntVector();
  //   dst = new device::Tensor(device, desc, data_ptr, name, memory_config);
  // }
  // return dst;

  auto info = var.get()->getInfo();

  halide_type_t src_data_type = info->type;
  base::DataType data_type = MnnConvert::convertToDataType(src_data_type);
  MNN::Express::Dimensionformat src_data_format = info->order;
  base::DataFormat format = MnnConvert::convertToDataFormat(src_data_format);
  base::IntVector shape = info->dim; 
  base::SizeVector stride = base::SizeVector();
  device::TensorDesc desc(data_type, format, shape, stride);
  device::Tensor *dst = nullptr;
  if (device == nullptr) {
    dst = new device::Tensor(desc, name);
  } else {
    void *data_ptr = (void *)(var.get()->writeMap<float>());
    base::IntVector memory_config = base::IntVector();
    dst = new device::Tensor(device, desc, data_ptr, name, memory_config);
  }
  return dst;
}

}  // namespace inference
}  // namespace nndeploy
