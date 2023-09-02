

#include "nndeploy/inference/ncnn/ncnn_convert.h"

namespace nndeploy {
namespace inference {

base::DataType NcnnConvert::convertToDataType(const int &src) {
  base::DataType dst;
  if (src == 4) {
    dst.code_ = base::kDataTypeCodeFp;
  }
  dst.bits_ = src * 8;
  dst.lanes_ = 1;
  return dst;
}

ncnn::DataType NcnnConvert::convertFromDataType(const base::DataType &src) {
  ncnn::DataType dst;
  if (src.code_ == base::kDataTypeCodeInt && src.bits_ == 8 &&
      src.lanes_ == 1) {
    dst = ncnn::DataType::DATA_TYPE_INT8;
  } else if (src.code_ == base::kDataTypeCodeInt && src.bits_ == 32 &&
             src.lanes_ == 1) {
    dst = ncnn::DataType::DATA_TYPE_INT32;
  } else if (src.code_ == base::kDataTypeCodeInt && src.bits_ == 64 &&
             src.lanes_ == 1) {
    dst = ncnn::DataType::DATA_TYPE_INT64;
  } else if (src.code_ == base::kDataTypeCodeUint && src.bits_ == 32 &&
             src.lanes_ == 1) {
    dst = ncnn::DataType::DATA_TYPE_UINT32;
  } else if (src.code_ == base::kDataTypeCodeFp && src.bits_ == 32 &&
             src.lanes_ == 1) {
    dst = ncnn::DataType::DATA_TYPE_FLOAT;
  } else if (src.code_ == base::kDataTypeCodeBFp && src.bits_ == 16 &&
             src.lanes_ == 1) {
    dst = ncnn::DataType::DATA_TYPE_BFP16;
  } else if (src.code_ == base::kDataTypeCodeFp && src.bits_ == 16 &&
             src.lanes_ == 1) {
    dst = ncnn::DataType::DATA_TYPE_HALF;
  } else {
    dst = ncnn::DataType::DATA_TYPE_FLOAT;
  }
  return dst;
}

base::DataFormat NcnnConvert::convertToDataFormat(const ncnn::DataFormat &src) {
  base::DataFormat dst;
  switch (src) {
    case ncnn::DataFormat::DATA_FORMAT_AUTO:
      dst = base::kDataFormatAuto;
      break;
    case ncnn::DataFormat::DATA_FORMAT_NHWC:
      dst = base::kDataFormatNHWC;
      break;
    case ncnn::DataFormat::DATA_FORMAT_NCHW:
      dst = base::kDataFormatNCHW;
      break;
    case ncnn::DataFormat::DATA_FORMAT_NC4HW4:
      dst = base::kDataFormatNC4HW;
      break;
    case ncnn::DataFormat::DATA_FORMAT_NC8HW8:
      dst = base::kDataFormatNC8HW;
    case ncnn::DataFormat::DATA_FORMAT_NCDHW:
      dst = base::kDataFormatNCDHW;
    default:
      dst = base::kDataFormatNotSupport;
      break;
  }
  return dst;
}

ncnn::DataFormat NcnnConvert::convertFromDataFormat(
    const base::DataFormat &src) {
  ncnn::DataFormat dst = ncnn::DataFormat::DATA_FORMAT_AUTO;
  switch (src) {
    case base::kDataFormatAuto:
      dst = ncnn::DataFormat::DATA_FORMAT_AUTO;
      break;
    case base::kDataFormatNCHW:
      dst = ncnn::DataFormat::DATA_FORMAT_NHWC;
      break;
    case base::kDataFormatNHWC:
      dst = ncnn::DataFormat::DATA_FORMAT_NHWC;
      break;
    case base::kDataFormatNC4HW:
      dst = ncnn::DataFormat::DATA_FORMAT_NC4HW4;
      break;
    case base::kDataFormatNC8HW:
      dst = ncnn::DataFormat::DATA_FORMAT_NC8HW8;
      break;
    case base::kDataFormatNCDHW:
      dst = ncnn::DataFormat::DATA_FORMAT_NCDHW;
      break;
    default:
      dst = ncnn::DataFormat::DATA_FORMAT_AUTO;
      break;
  }
  return dst;
}

ncnn::DeviceType NcnnConvert::convertFromDeviceType(
    const base::DeviceType &src) {
  ncnn::DeviceType type = ncnn::DeviceType::DEVICE_NAIVE;
  switch (src.code_) {
    case base::kDeviceTypeCodeCpu:
      type = ncnn::DeviceType::DEVICE_NAIVE;
      break;
    case base::kDeviceTypeCodeX86:
      type = ncnn::DeviceType::DEVICE_X86;
      break;
    case base::kDeviceTypeCodeArm:
      type = ncnn::DeviceType::DEVICE_ARM;
      break;
    case base::kDeviceTypeCodeOpenCL:
      type = ncnn::DeviceType::DEVICE_OPENCL;
      break;
    case base::kDeviceTypeCodeMetal:
      type = ncnn::DeviceType::DEVICE_METAL;
      break;
    case base::kDeviceTypeCodeCuda:
      type = ncnn::DeviceType::DEVICE_CUDA;
      break;
    default:
      type = ncnn::DeviceType::DEVICE_NAIVE;
      break;
  }
  return type;
}

base::DeviceType NcnnConvert::convertToDeviceType(const ncnn::DeviceType &src) {
  base::DeviceType type = base::kDeviceTypeCodeNotSupport;
  switch (src) {
    case ncnn::DeviceType::DEVICE_NAIVE:
      type = base::kDeviceTypeCodeCpu;
      break;
    case ncnn::DeviceType::DEVICE_X86:
      type = base::kDeviceTypeCodeX86;
      break;
    case ncnn::DeviceType::DEVICE_ARM:
      type = base::kDeviceTypeCodeArm;
      break;
    case ncnn::DeviceType::DEVICE_OPENCL:
      type = base::kDeviceTypeCodeOpenCL;
      break;
    case ncnn::DeviceType::DEVICE_METAL:
      type = base::kDeviceTypeCodeMetal;
      break;
    case ncnn::DeviceType::DEVICE_CUDA:
      type = base::kDeviceTypeCodeCuda;
      break;
    default:
      type = base::kDeviceTypeCodeNotSupport;
      break;
  }
  return type;
}

base::ModelType NcnnConvert::convertToModelType(const ncnn::ModelType &src) {
  base::ModelType type = base::kModelTypeDefault;
  switch (src) {
    case ncnn::ModelType::MODEL_TYPE_NCNN:
      type = base::kModelTypeNcnn;
      break;
    case ncnn::ModelType::MODEL_TYPE_NCNN:
      type = base::kModelTypeNcnn;
      break;
    case ncnn::ModelType::MODEL_TYPE_OPENVINO:
      type = base::kModelTypeOpenVino;
      break;
    case ncnn::ModelType::MODEL_TYPE_COREML:
      type = base::kModelTypeCoreML;
      break;
    default:
      type = base::kModelTypeNotSupport;
      break;
  }
  return type;
}

ncnn::ModelType NcnnConvert::convertFromModelType(const base::ModelType &src) {
  ncnn::ModelType type = ncnn::ModelType::MODEL_TYPE_NCNN;
  switch (src) {
    case base::kModelTypeNcnn:
      type = ncnn::ModelType::MODEL_TYPE_NCNN;
      break;
    case base::kModelTypeNcnn:
      type = ncnn::ModelType::MODEL_TYPE_NCNN;
      break;
    case base::kModelTypeOpenVino:
      type = ncnn::ModelType::MODEL_TYPE_OPENVINO;
      break;
    case base::kModelTypeCoreML:
      type = ncnn::ModelType::MODEL_TYPE_COREML;
      break;
    default:
      type = ncnn::ModelType::MODEL_TYPE_NCNN;
      break;
  }
  return type;
}

base::ShareMemoryType NcnnConvert::convertToShareMemoryMode(
    const ncnn::ShareMemoryMode &src) {
  base::ShareMemoryType type = base::kShareMemoryTypeNoShare;
  switch (src) {
    case ncnn::ShareMemoryMode::SHARE_MEMORY_MODE_SET_FROM_EXTERNAL:
      type = base::kShareMemoryTypeShareFromExternal;
      break;
    case ncnn::ShareMemoryMode::SHARE_MEMORY_MODE_DEFAULT:
      type = base::kShareMemoryTypeNoShare;
      break;
    default:
      type = base::kShareMemoryTypeNotSupport;
      break;
  }
  return type;
}

ncnn::ShareMemoryMode NcnnConvert::convertFromShareMemoryMode(
    const base::ShareMemoryType &src) {
  ncnn::ShareMemoryMode type = ncnn::ShareMemoryMode::SHARE_MEMORY_MODE_DEFAULT;
  switch (src) {
    case base::kShareMemoryTypeShareFromExternal:
      type = ncnn::ShareMemoryMode::SHARE_MEMORY_MODE_SET_FROM_EXTERNAL;
      break;
    default:
      type = ncnn::ShareMemoryMode::SHARE_MEMORY_MODE_DEFAULT;
      break;
  }
  return type;
}

ncnn::Precision NcnnConvert::convertFromPrecisionType(
    const base::PrecisionType &src) {
  switch (src) {
    case base::kPrecisionTypeFp16:
      return ncnn::Precision::PRECISION_LOW;
    case base::kPrecisionTypeBFp16:
      return ncnn::Precision::PRECISION_LOW;
    case base::kPrecisionTypeFp32:
      return ncnn::Precision::PRECISION_NORMAL;
    case base::kPrecisionTypeFp64:
      return ncnn::Precision::PRECISION_HIGH;
    default:
      return ncnn::Precision::PRECISION_AUTO;
  }
}

base::Status NcnnConvert::convertFromInferenceParam(
    inference::NcnnInferenceParam *src, ncnn::Option &dst) {
  base::Status status = base::kStatusCodeOk;
  dst.lightmode = src->lightmode_;

  dst.num_threads = src->num_thread_;

  dst.openmp_blocktime = src->openmp_blocktime_;

  dst.use_winograd_convolution = src->use_winograd_convolution_;

  dst.use_sgemm_convolution = src->use_sgemm_convolution_;

  dst.use_int8_inference = src->use_int8_inference_;

  dst.use_vulkan_compute =
      src->device_type_.code_ == base::kDeviceTypeCodeVulkan;

  dst.use_bf16_storage = src->use_bf16_storage_;

  dst.use_fp16_packed = src->use_fp16_packed_;
  dst.use_fp16_storage = src->use_fp16_storage_;
  dst.use_fp16_arithmetic = src->use_fp16_arithmetic_;
  dst.use_int8_packed = src->use_int8_packed_;
  dst.use_int8_storage = src->use_int8_storage_;
  dst.use_int8_arithmetic = src->use_int8_arithmetic_;

  dst.use_packing_layout_ = src->use_packing_layout_;

  dst.use_shader_pack8 = src->use_shader_pack8_;

  dst.use_subgroup_basic = src->use_subgroup_basic_;
  dst.use_subgroup_vote = src->use_subgroup_vote_;
  dst.use_subgroup_ballot = src->use_subgroup_ballot_;
  dst.use_subgroup_shuffle = src->use_subgroup_shuffle_;

  dst.use_image_storage = src->use_image_storage_;
  dst.use_tensor_storage = src->use_tensor_storage_;

  dst.use_reserved_0 = src->use_reserved_0_;

  dst.flush_denormals = src->flush_denormals_;

  dst.use_local_pool_allocator = src->use_local_pool_allocator_;

  // enable local memory optimization for gpu inference
  dst.use_shader_local_memory = src->use_local_pool_allocator_;

  // enable cooperative matrix optimization for gpu inference
  dst.use_cooperative_matrix = src->use_cooperative_matrix_;

  // more fine-grained control of winograd convolution
  dst.use_winograd23_convolution = src->use_winograd23_convolution_;
  dst.use_winograd43_convolution = src->use_winograd43_convolution_;
  dst.use_winograd63_convolution = src->use_winograd63_convolution_;

  // this option is turned on for A53/A55 automatically
  // but you can force this on/off if you wish
  dst.use_a53_a55_optimized_kernel = src->use_a53_a55_optimized_kernel_;

  return base::kStatusCodeOk;
}

base::Status NcnnConvert::matConvertToTensor(ncnn::Mat &src,
                                             const std::string &name,
                                             device::Tensor *dst) {
  dst->destory();
  device::Device *device = device::getDefaultHostDevice();
  base::DataType data_type = convertToDataType(src.elemsize);
  base::DataFormat data_format =
      convertToDataFormat(src.elempack, src.dims, src.w, src.h, src.d, src.c,
                          src.cstep);  // 目前只使用了shape.dims
  base::IntVector dims =
      convertToShape(src.dims, src.w, src.h, src.d, src.c, src.cstep);
  device::TensorDesc tensor_desc_(data_type, data_format, dims,
                                  base::SizeVector());
  void *data_ptr = src.data;
  dst->create(device, tensor_desc_, data_ptr, name);
  return base::kStatusCodeOk;
}

ncnn::Mat NcnnConvert::matConvertFromTensor(device::Tensor *src) {
  if (!device::isHostDeviceType(src->getDeviceType())) {
    Mat dst();
    return dst;
  }
  void *data = src->getPtr();
  int elemsize = src->getDataType().size();
  base::IntVector shape = src->getShape();
  if (shape.size() == 2) {
    Mat dst(shape[1], data, elemsize);
    return dst;
  } else if (shape.size() == 3) {
    Mat dst(shape[2], shape[1], data, elemsize);
    return dst;
  } else if (shape.size() == 4) {
    Mat dst(shape[3], shape[2], shape[1], data, elemsize);
    return dst;
  } else if (shape.size() == 5) {
    Mat dst(shape[4], shape[3], shape[2], shape[1], data, elemsize);
    return dst;
  } else {
    Mat dst();
    return dst;
  }
}

device::Tensor *NcnnConvert::blobConvertToTensor(ncnn::Blob &src) {
  device::Device *device = device::getDefaultHostDevice();
  base::DataType data_type = convertToDataType(src.shape.elemsize);
  base::DataFormat data_format = convertToDataFormat(
      src.shape.elempack, src.shape.dims, src.shape.w, src.shape.h, src.shape.d,
      src.shape.c, src.shape.cstep);  // 目前只使用了shape.dims
  base::IntVector dims =
      convertToShape(src.shape.dims, src.shape.w, src.shape.h, src.shape.d,
                     src.shape.c, src.shape.cstep);
  device::TensorDesc tensor_desc_(data_type, data_format, dims,
                                  base::SizeVector());
  std::string name = src.name;
  device::Tensor *dst = nullptr;
  if (src.shape.data != nullptr) {
    device::Device *device = device::getDefaultHostDevice();
    void *data_ptr = src.shape.data;
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
