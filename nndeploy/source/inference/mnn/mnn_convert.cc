
#include "nndeploy/source/inference/mnn/mnn_convert.h"

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

halide_type_t MnnConvert::convertFromDataType(base::DataType &src) {
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

MNN::BackendConfig::PrecisionMode MnnConvert::convertFromPowerType(
    const base::PrecisionType &src) {
  switch (src) {
    case base::kPrecisionTypeFp16:
      return MNN::BackendConfig::PrecisionMode::Precision_Low;
    case base::kPrecisionTypeFpBFp16:
      return MNN::BackendConfig::PrecisionMode::Precision_Low;
    case base::kPrecisionTypeFp32:
      return MNN::BackendConfig::PrecisionMode::Precision_Normal;
    case base::kPrecisionTypeFp64:
      return MNN::BackendConfig::PrecisionMode::Precision_High;
    default:
      return MNN::BackendConfig::PrecisionMode::Precision_Normal;
  }
}

base::Status MnnConvert::convertFromConfig(
    MnnConfigImpl *config, MNN::ScheduleConfig *internal_config) {
  if (config == nullptr || internal_config == nullptr) {
    return base::kStatusCodeErrorInvalidParam;
  }

  internal_config->saveTensors = config->save_tensors_;
  internal_config->type = convertFromDeviceType(config->device_type_);
  if (internal_config->type == MNN_FORWARD_CPU) {
    internal_config->numThread = config->num_thread_;
  } else if (internal_config->type == MNN_FORWARD_OPENCL ||
             internal_config->type == MNN_FORWARD_OPENGL ||
             internal_config->type == MNN_FORWARD_METAL ||
             internal_config->type == MNN_FORWARD_CUDA) {
    internal_config->mode = config->gpu_tune_mode_;
  }
  internal_config->path = config->path_;
  internal_config->backupType =
      convertFromDeviceType(config->backup_device_type_);

  internal_config->backendConfig = new MNN::BackendConfig();
  internal_config->backendConfig->power =
      convertFromPowerType(config->power_type_);
  internal_config->backendConfig->precision =
      convertFromPowerType(config->precision_type_);
  internal_config->backendConfig->memory = config->memory_mode_;

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
  device::TensorImplDesc desc(data_type, format, shape, stride);
  base::TensorImplType type = base::kTensorImplTypeDefault;
  auto src_buffer = src->buffer();
  void *data_ptr = (void *)src_buffer.host;
  base::IntVector memory_config = base::IntVector();
  device::Tensor *dst =
      new device::Tensor(device, desc, data_ptr, name, memory_config, type);
  return dst;
}

MNN::Tensor *MnnConvert::convertFromTensor(device::Tensor *src) {
  device::TensorImplDesc desc = src->getDesc();
  std::vector<int> shape = desc.shape_;
  halide_type_t type = MnnConvert::convertFromDataType(desc.data_type_);
  void *data = src->getPtr();
  MNN::Tensor::DimensionType dimType =
      MnnConvert::convertFromDataFormat(desc.format_);
  MNN::Tensor *dst = MNN::Tensor::create(shape, type, data, dimType);
  return dst;
}

}  // namespace inference
}  // namespace nndeploy
