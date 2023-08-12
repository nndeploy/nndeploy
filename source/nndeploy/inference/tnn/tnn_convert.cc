

#include "nndeploy/inference/tnn/tnn_convert.h"

namespace nndeploy {
namespace inference {

base::DataType TnnConvert::convertToDataType(const tnn::DataType &src) {
  base::DataType dst;
  switch (src) {
    case tnn::DataType::DATA_TYPE_INT8:
    case tnn::DataType::DATA_TYPE_INT32:
    case tnn::DataType::DATA_TYPE_INT64:
      dst.code_ = base::kDataTypeCodeInt;
      break;
    case tnn::DataType::DATA_TYPE_UINT32:
      dst.code_ = base::kDataTypeCodeUint;
      break;
    case tnn::DataType::DATA_TYPE_FLOAT:
      dst.code_ = base::kDataTypeCodeFp;
      break;
    case tnn::DataType::DATA_TYPE_BFP16:
      dst.code_ = base::kDataTypeCodeBFp;
      break;
    default:
      dst.code_ = base::kDataTypeCodeOpaqueHandle;
      break;
  }
  dst.bits_ = tnn::DataTypeUtils::GetBytesSize(src) * 8;
  dst.lanes_ = 1;
  return dst;
}

tnn::DataType TnnConvert::convertFromDataType(const base::DataType &src) {
  tnn::DataType dst;
  if (src.code_ == base::kDataTypeCodeInt && src.bits_ == 8 &&
      src.lanes_ == 1) {
    dst = tnn::DataType::DATA_TYPE_INT8;
  } else if (src.code_ == base::kDataTypeCodeInt && src.bits_ == 32 &&
             src.lanes_ == 1) {
    dst = tnn::DataType::DATA_TYPE_INT32;
  } else if (src.code_ == base::kDataTypeCodeInt && src.bits_ == 64 &&
             src.lanes_ == 1) {
    dst = tnn::DataType::DATA_TYPE_INT64;
  } else if (src.code_ == base::kDataTypeCodeUint && src.bits_ == 32 &&
             src.lanes_ == 1) {
    dst = tnn::DataType::DATA_TYPE_UINT32;
  } else if (src.code_ == base::kDataTypeCodeFp && src.bits_ == 32 &&
             src.lanes_ == 1) {
    dst = tnn::DataType::DATA_TYPE_FLOAT;
  } else if (src.code_ == base::kDataTypeCodeBFp && src.bits_ == 16 &&
             src.lanes_ == 1) {
    dst = tnn::DataType::DATA_TYPE_BFP16;
  } else if (src.code_ == base::kDataTypeCodeFp && src.bits_ == 16 &&
             src.lanes_ == 1) {
    dst = tnn::DataType::DATA_TYPE_HALF;
  } else {
    dst = tnn::DataType::DATA_TYPE_FLOAT;
  }
  return dst;
}

base::DataFormat TnnConvert::convertToDataFormat(const tnn::DataFormat &src) {
  base::DataFormat dst;
  switch (src) {
    case tnn::DataFormat::DATA_FORMAT_AUTO:
      dst = base::kDataFormatAuto;
    case tnn::DataFormat::DATA_FORMAT_NHWC:
      dst = base::kDataFormatNHWC;
      break;
    case tnn::DataFormat::DATA_FORMAT_NCHW:
      dst = base::kDataFormatNCHW;
      break;
    case tnn::DataFormat::DATA_FORMAT_NC4HW4:
      dst = base::kDataFormatNC4HW;
      break;
    case tnn::DataFormat::DATA_FORMAT_NC8HW8:
      dst = base::kDataFormatNC8HW;
    case tnn::DataFormat::DATA_FORMAT_NCDHW:
      dst = base::kDataFormatNCDHW;
    default:
      dst = base::kDataFormatNotSupport;
      break;
  }
  return dst;
}

tnn::DataFormat TnnConvert::convertFromDataFormat(const base::DataFormat &src) {
  tnn::DataFormat dst = tnn::DataFormat::DATA_FORMAT_AUTO;
  switch (src) {
    case base::kDataFormatAuto:
      dst = tnn::DataFormat::DATA_FORMAT_AUTO;
    case base::kDataFormatNCHW:
      dst = tnn::DataFormat::DATA_FORMAT_NHWC;
      break;
    case base::kDataFormatNHWC:
      dst = tnn::DataFormat::DATA_FORMAT_NHWC;
      break;
    case base::kDataFormatNC4HW:
      dst = tnn::DataFormat::DATA_FORMAT_NC4HW4;
      break;
    case base::kDataFormatNC8HW:
      dst = tnn::DataFormat::DATA_FORMAT_NC8HW8;
      break;
    case base::kDataFormatNCDHW:
      dst = tnn::DataFormat::DATA_FORMAT_NCDHW;
      break;
    default:
      dst = tnn::DataFormat::DATA_FORMAT_AUTO;
      break;
  }
  return dst;
}

tnn::DeviceType TnnConvert::convertFromDeviceType(const base::DeviceType &src) {
  tnn::DeviceType type = tnn::DeviceType::DEVICE_NAIVE;
  switch (src.code_) {
    case base::kDeviceTypeCodeCpu:
      type = tnn::DeviceType::DEVICE_NAIVE;
      break;
    case base::kDeviceTypeCodeX86:
      type = tnn::DeviceType::DEVICE_X86;
      break;
    case base::kDeviceTypeCodeArm:
      type = tnn::DeviceType::DEVICE_ARM;
      break;
    case base::kDeviceTypeCodeOpenCL:
      type = tnn::DeviceType::DEVICE_OPENCL;
      break;
    case base::kDeviceTypeCodeMetal:
      type = tnn::DeviceType::DEVICE_METAL;
      break;
    case base::kDeviceTypeCodeCuda:
      type = tnn::DeviceType::DEVICE_CUDA;
      break;
    default:
      type = tnn::DeviceType::DEVICE_NAIVE;
      break;
  }
  return type;
}

base::DeviceType TnnConvert::convertToDeviceType(const tnn::DeviceType &src) {
  base::DeviceType type = base::kDeviceTypeCodeNotSupport;
  switch (src) {
    case tnn::DeviceType::DEVICE_NAIVE:
      type = base::kDeviceTypeCodeCpu;
      break;
    case tnn::DeviceType::DEVICE_X86:
      type = base::kDeviceTypeCodeX86;
      break;
    case tnn::DeviceType::DEVICE_ARM:
      type = base::kDeviceTypeCodeArm;
      break;
    case tnn::DeviceType::DEVICE_OPENCL:
      type = base::kDeviceTypeCodeOpenCL;
      break;
    case tnn::DeviceType::DEVICE_METAL:
      type = base::kDeviceTypeCodeMetal;
      break;
    case tnn::DeviceType::DEVICE_CUDA:
      type = base::kDeviceTypeCodeCuda;
      break;
    default:
      type = base::kDeviceTypeCodeNotSupport;
      break;
  }
  return type;
}

base::ModelType TnnConvert::convertToModelType(const tnn::ModelType &src) {
  base::ModelType type = base::kModelTypeDefault;
  switch (src) {
    case tnn::ModelType::MODEL_TYPE_TNN:
      type = base::kModelTypeTnn;
      break;
    case tnn::ModelType::MODEL_TYPE_NCNN:
      type = base::kModelTypeNcnn;
      break;
    case tnn::ModelType::MODEL_TYPE_OPENVINO:
      type = base::kModelTypeOpenVino;
      break;
    case tnn::ModelType::MODEL_TYPE_COREML:
      type = base::kModelTypeCoreML;
      break;
    default:
      type = base::kModelTypeNotSupport;
      break;
  }
  return type;
}

tnn::ModelType TnnConvert::convertFromModelType(const base::ModelType &src) {
  tnn::ModelType type = tnn::ModelType::MODEL_TYPE_TNN;
  switch (src) {
    case base::kModelTypeTnn:
      type = tnn::ModelType::MODEL_TYPE_TNN;
      break;
    case base::kModelTypeNcnn:
      type = tnn::ModelType::MODEL_TYPE_NCNN;
      break;
    case base::kModelTypeOpenVino:
      type = tnn::ModelType::MODEL_TYPE_OPENVINO;
      break;
    case base::kModelTypeCoreML:
      type = tnn::ModelType::MODEL_TYPE_COREML;
      break;
    default:
      type = tnn::ModelType::MODEL_TYPE_TNN;
      break;
  }
  return type;
}

base::ShareMemoryType TnnConvert::convertToShareMemoryMode(
    const tnn::ShareMemoryMode &src) {
  base::ShareMemoryType type = base::kShareMemoryTypeNoShare;
  switch (src) {
    case tnn::ShareMemoryMode::SHARE_MEMORY_MODE_SET_FROM_EXTERNAL:
      type = base::kShareMemoryTypeShareFromExternal;
      break;
    case tnn::ShareMemoryMode::SHARE_MEMORY_MODE_DEFAULT:
      type = base::kShareMemoryTypeNoShare;
      break;
    default:
      type = base::kShareMemoryTypeNotSupport;
      break;
  }
  return type;
}

tnn::ShareMemoryMode TnnConvert::convertFromShareMemoryMode(
    const base::ShareMemoryType &src) {
  tnn::ShareMemoryMode type = tnn::ShareMemoryMode::SHARE_MEMORY_MODE_DEFAULT;
  switch (src) {
    case base::kShareMemoryTypeShareFromExternal:
      type = tnn::ShareMemoryMode::SHARE_MEMORY_MODE_SET_FROM_EXTERNAL;
      break;
    default:
      type = tnn::ShareMemoryMode::SHARE_MEMORY_MODE_DEFAULT;
      break;
  }
  return type;
}

tnn::Precision TnnConvert::convertFromPrecisionType(
    const base::PrecisionType &src) {
  switch (src) {
    case base::kPrecisionTypeFp16:
      return tnn::Precision::PRECISION_LOW;
    case base::kPrecisionTypeBFp16:
      return tnn::Precision::PRECISION_LOW;
    case base::kPrecisionTypeFp32:
      return tnn::Precision::PRECISION_NORMAL;
    case base::kPrecisionTypeFp64:
      return tnn::Precision::PRECISION_HIGH;
    default:
      return tnn::Precision::PRECISION_AUTO;
  }
}

base::Status TnnConvert::convertFromInferenceParam(
    inference::TnnInferenceParam *src, tnn::ModelConfig &model_config_,
    tnn::NetworkConfig &network_config_) {
  model_config_.model_type = convertFromModelType(src->model_type_);
  model_config_.params = src->model_value_;

  network_config_.device_type = convertFromDeviceType(src->device_type_);
  network_config_.device_id = src->device_type_.device_id_;
  network_config_.data_format = convertFromDataFormat(src->data_format_);
  network_config_.precision = convertFromPrecisionType(src->precision_type_);
  network_config_.cache_path = src->cache_path_[0];
  network_config_.share_memory_mode =
      convertFromShareMemoryMode(src->share_memory_mode_);
  network_config_.library_path = src->library_path;

  // enable_tune_kernel暂定大于-1为存在gpu使用情况
  if (src->gpu_tune_kernel_ > -1) {
    network_config_.enable_tune_kernel = true;
  }

  return base::kStatusCodeOk;
}

device::Tensor *TnnConvert::matConvertToTensor(tnn::Mat *src,
                                               std::string name) {
  base::DataType data_type_;
  base::DataFormat data_format_;
  if (src->GetMatType() == tnn::NCHW_FLOAT) {
    data_type_ = base::dataTypeOf<float>();
    data_format_ = base::kDataFormatNCHW;
  } else if (src->GetMatType() == tnn::NC_INT32) {
    data_type_ = base::dataTypeOf<int>();
    data_format_ = base::kDataFormatNCHW;
  } else {
    NNDEPLOY_LOGE("TNN matConvertToTensor failed!\n");
    return nullptr;
  }
  device::Device *device =
      device::getDevice(convertToDeviceType(src->GetDeviceType()));
  device::TensorDesc tensor_desc_(data_type_, data_format_, src->GetDims(),
                                  base::SizeVector());
  void *data_ptr = src->GetData();
  base::IntVector memory_config = base::IntVector();
  device::Tensor *dst =
      new device::Tensor(device, tensor_desc_, data_ptr, name, memory_config);

  return dst;
}

tnn::Mat *TnnConvert::matConvertFromTensor(device::Tensor *src) {
  device::TensorDesc desc = src->getDesc();
  std::vector<int> shape_dims = desc.shape_;

  tnn::MatType mat_type;
  if (src->getDataType().code_ == base::kDataTypeCodeFp ||
      src->getDataFormat() == base::kDataFormatNCHW) {
    mat_type = tnn::NCHW_FLOAT;
  } else if (src->getDataType().code_ == base::kDataTypeCodeInt ||
             src->getDataFormat() == base::kDataFormatNCHW) {
    mat_type = tnn::NC_INT32;
  } else {
    NNDEPLOY_LOGE("TNN matConvertFromTensor failed!\n");
    return nullptr;
  }
  void *data = src->getPtr();
  tnn::DeviceType device_type = convertFromDeviceType(src->getDeviceType());
  tnn::Mat *dst = new tnn::Mat(device_type, mat_type, shape_dims, data);
  return dst;
}

// 假如src是CPU/ARM/X86，就共享内存，否则建立一个空Tensor
device::Tensor *TnnConvert::blobConvertToTensor(tnn::Blob *src) {
  tnn::BlobDesc blob_desc_ = src->GetBlobDesc();
  base::DeviceType device_type =
      convertToDeviceType(blob_desc_.device_type);  //
  base::DataType data_type = convertToDataType(blob_desc_.data_type);
  base::DataFormat data_format = convertToDataFormat(blob_desc_.data_format);
  base::IntVector dims = blob_desc_.dims;
  std::string name = blob_desc_.name;
  device::TensorDesc tensor_desc_(data_type, data_format, dims,
                                  base::SizeVector());

  device::Tensor *dst = nullptr;
  if (device::isHostDeviceType(device_type)) {
    device::Device *device = device::getDevice(device_type);
    void *data_ptr = (src->GetHandle()).base;
    base::IntVector memory_config = base::IntVector();
    dst =
        new device::Tensor(device, tensor_desc_, data_ptr, name, memory_config);
  } else {
    dst = new device::Tensor(tensor_desc_, name);
  }

  return dst;
}

}  // namespace inference
}  // namespace nndeploy
