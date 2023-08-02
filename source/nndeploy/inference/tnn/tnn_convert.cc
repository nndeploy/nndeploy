

#include "nndeploy/inference/tnn/tnn_convert.h"

/*
#TODO:
1.data_format中默认为auto是否得当 -可以之后再看有没有错误
2.precision_type是简写还是要仔细判断？-等下仔细写
3.convertFromInferenceParam有待办
    -gpu_tune_kernel_不等于0的话就设置为true？是0还是-1？-这个之后再临时看
4.default情况如果TNN不支持就打印log并exit
-------------------------------------------------------------------------
5.MatToTensor：
    -时device == nullptr情况为什么不创建？
    -错误情况下return类型不正确 -返回空并检查如果是空就报错
6.所有输入是不是最好都加const？
*/

namespace nndeploy {
namespace inference {

base::DataType TnnConvert::convertToDataType(const TNN_NS::DataType &src) {
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
  dst.bits_ = DataTypeUtils::GetBytesSize(src) * 8;
  dst.lanes_ = 1;
  return dst;
}

tnn::DataType TnnConvert::convertFromDataType(base::DataType &src) {
  TNN_NS::DataType dst;
  if (src.code_ == base::kDataTypeCodeInt && src.bits_ == 8 &&
      src.lanes_ == 1) {
    dst = tnn::DataType::DATA_TYPE_INT8;
  } else if (src.code_ == base::kDataTypeCodeInt && src.bits_ == 32 &&
             src.lanes_ == 1) {
    dst = tnn::DataType::DATA_TYPE_INT32;
  } else if (src.code_ == base::kDataTypeCodeInt && src.bits_ == 64 &&
             src.lanes_ == 1) {
    dst = tnn::DataType::DATA_TYPE_INT64;
  } else if (src.code_ == base::kDataTypeCodeUInt && src.bits_ == 32 &&
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

base::DataFormat TnnConvert::convertToDataFormat(
    const TNN_NS::DataFormat &src) {
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
    case base::kDeviceTypeCodeOpenGL:
      type = NULL;
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
  switch (src.code_) {
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
  switch (src.code_) {
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

tnn::ModelType TnnConvert::convertToModelType(const base::ModelType &src) {
  tnn::ModelType type = tnn::ModelType::MODEL_TYPE_TNN;
  switch (src.code_) {
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

base::ShareMemoryType TnnConvert::convertFromShareMemoryMode(
    const tnn::ShareMemoryMode &src) {
  base::ShareMemoryType type = base::kShareMemoryTypeNoShare;
  switch (src.code_) {
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

tnn::ShareMemoryMode TnnConvert::convertToShareMemoryMode(
    const base::ShareMemoryType &src) {  // 这里的逻辑问题很大
  tnn::ShareMemoryMode type = tnn::ShareMemoryMode::SHARE_MEMORY_MODE_DEFAULT;
  switch (src.code_) {
    case base::kShareMemoryTypeShareFromExternal:
      type = tnn::ShareMemoryMode::SHARE_MEMORY_MODE_SET_FROM_EXTERNAL;
      break;
    case base::kShareMemoryTypeNotSupport:
      NNDEPLOY_LOGE("base::ShareMemoryType = kShareMemoryTypeNotSupport!\n");
      type = tnn::ShareMemoryMode::SHARE_MEMORY_MODE_DEFAULT;
      break;
    default:
      type = base::kShareMemoryTypeNoShare;
      break;
  }
  return type;
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

base::Status TnnConvert::convertFromInferenceParam(
    inference::TnnInferenceParam &src, TNN_NS::ModelConfig &model_config_,
    TNN_NS::NetworkConfig &network_config_) {
  if (src == nullptr || model_config_ == nullptr ||
      network_config_ == nullptr) {
    return base::kStatusCodeErrorInvalidParam;
  }

  // enable_tune_kernel是否有特殊情况 -
  // 这个还没有定论，暂定大于-1为存在gpu使用情况
  network_config_.device_id = src.device_type_.device_id_;
  network_config_.device_type = convertFromDeviceType(src.device_type_);
  network_config_.data_format = convertFromDataFormat(src.inputs_data_format_);
  network_config_.precision = convertFromPrecisionType(src.precision_type_);
  network_config_.cache_path = src.cache_path_;
  network_config_.share_memory_mode =
      convertFromDataFormat(src.share_memory_mode_);
  network_config_.library_path = src.library_path;

  if (src.gpu_tune_kernel_ > -1) {
    network_config_.enable_tune_kernel = true;
  }

  model_config_.model_type = convertFromModelType(src.model_type_);
  model_config_.params = src.model_value_;  // 注意这里可能会出问题

  return base::kStatusCodeOk;
}

device::Tensor *TnnConvert::MatConvertToTensor(TNN::Mat *src,
                                               std::string name) {
  if (src->GetMatType() == tnn::NCHW_FLOAT) {
    DataType data_type_(base::kDataTypeCodeFp, 32);
    data_format_ = base::kDataFormatNCHW;
  } else if (src->GetMatType() == tnn::NC_INT32) {
    DataType data_type_(base::kDataTypeCodeInt, 32);
    data_format_ = base::kDataFormatNCHW;
  } else {
    NNDEPLOY_LOGE("TNN MatConvertToTensor failed!\n");
    return base::kStatusCodeErrorInferenceTnn;  // 这里return的结果不对
  }
  device::Device *device = getDevice(convertToDeviceType(src->GetDeviceType()));
  TensorDesc tensor_desc_(data_type_, data_format_, src->GetDims());

  device::Tensor *dst = nullptr;

  void *data_ptr = src.GetData();
  base::IntVector memory_config = base::IntVector();
  dst = new device::Tensor(device, tensor_desc_, data_ptr, name, memory_config);

  return dst;
}

TNN::Mat *TnnConvert::MatConvertFromTensor(device::Tensor *src) {
  device::TensorDesc desc = src->getDesc();
  std::vector<int> shape_dims = desc.shape_;

  if (src->getDataType() == base::kDataTypeCodeFp ||
      src->getDataFormat() == kDataFormatNCHW) {  // 是不是要bits=32呀
    tnn::MatType mat_type = tnn::NCHW_FLOAT;
  } else if (src->getDataType() == base::kDataTypeCodeInt ||
             src->getDataFormat() == kDataFormatNCHW) {
    tnn::MatType mat_type = tnn::NC_INT32;
  } else {
    NNDEPLOY_LOGE("TNN MatConvertFromTensor failed!\n");
    return base::kStatusCodeErrorInferenceTnn;
  }
  void *data = src->getPtr();
  TNN::DeviceType device_type =
      TnnConvert::convertToDeviceType(src->getDeviceType());
  TNN::Mat *dst = new TNN::Mat(device_type, MatType mat_type, shape_dims, data);
  return dst;
}

tnn::Blob *TnnConvert::BlobConvertFromTensor(device::Tensor *src) {
  tnn::DeviceType device_type = convertToDeviceType(src->getDeviceType());
  tnn::DataType data_type = convertToDataType(src->getDataType());
  tnn::DataFormat data_format = convertToDataFormat(src->getDataFormat());
  tnn::DimsVector dims = src->getShape();
  std::string name = src->getName();

  tnn::BlobDesc blob_desc(device_type, data_type, data_format, dims, name);
  tnn::Blob *dst = new tnn::Blob(blob_desc);

  return dst;
}

device::Tensor *TnnConvert::BlobConvertToTensor(tnn::Blob *src) {
  BlobDesc blob_desc_ = src->GetBlobDesc();
  base::DeviceType device_type =
      convertFromDeviceType(blob_desc_.device_type);  //
  base::DataType data_type = convertFromDataType(blob_desc_.data_type);
  base::DataFormat data_format = convertFromDataFormat(blob_desc_.data_format);
  base::IntVector dims = blob_desc_.dims;
  std::string name = blob_desc_.name;
  device::Device *device = getDevice(convertToDeviceType(src->GetDeviceType()));

  TensorDesc tensor_desc_(data_type, data_format, dims);

  device::Tensor *dst = nullptr;

  void *data_ptr = (src.GetHandle())->base;
  base::IntVector memory_config = base::IntVector();
  dst = new device::Tensor(device, tensor_desc_, data_ptr, name, memory_config);
}
return dst;
}
}  // namespace inference
}  // namespace nndeploy
