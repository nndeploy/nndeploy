#include "nndeploy/inference/tvm/tvm_convert.h"

namespace nndeploy {
namespace inference {

base::DataType TvmConvert::convertToDataType(
    const tvm::runtime::DataType &src) {
  base::DataType dst;
  switch (src.code()) {
    case tvm::runtime::DataType::kInt:
      dst.code_ = base::kDataTypeCodeInt;
      break;
    case tvm::runtime::DataType::kUInt:
      dst.code_ = base::kDataTypeCodeUint;
      break;
    case tvm::runtime::DataType::kFloat:
      dst.code_ = base::kDataTypeCodeFp;
      break;
    case tvm::runtime::DataType::kBFloat:
      dst.code_ = base::kDataTypeCodeBFp;
      break;
    default:
      dst.code_ = base::kDataTypeCodeOpaqueHandle;
      break;
  }
  dst.bits_ = src.bits();
  dst.lanes_ = src.lanes();
  return dst;
}

tvm::runtime::DataType TvmConvert::convertFromDataType(
    const base::DataType &src) {
  tvm::runtime::DataType dst;

  switch (src.code_) {
    case base::kDataTypeCodeInt:
      dst = tvm::runtime::DataType::Int(src.bits_, src.lanes_);
      break;
    case base::kDataTypeCodeUint:
      dst = tvm::runtime::DataType::UInt(src.bits_, src.lanes_);
      break;
    case base::kDataTypeCodeFp:
      dst = tvm::runtime::DataType::Float(src.bits_, src.lanes_);
      break;
    case base::kDataTypeCodeBFp:
      dst = tvm::runtime::DataType::BFloat(src.bits_, src.lanes_);
      break;
    default:
      dst = tvm::runtime::DataType::Void();
      break;
  }
  return dst;
}

base::DataFormat TvmConvert::convertToDataFormat(const std::string &src) {
  base::DataFormat dst;
  if (src == "NCHW") {
    dst = base::kDataFormatNCHW;
  } else if (src == "NHWC") {
    dst = base::kDataFormatNHWC;
  } else {
    dst = base::kDataFormatNotSupport;
  }

  return dst;
}

std::string TvmConvert::convertFromDataFormat(const base::DataFormat &src) {
  std::string dst;
  switch (src) {
    case base::kDataFormatNCHW:
      dst = "NCHW";
      break;
    case base::kDataFormatNHWC:
      dst = "NHWC";
      break;
    default:
      NNDEPLOY_LOGE("TVM convertFromDataFormat failed!\n");
  }
  return dst;
}

base::DeviceType TvmConvert::convertToDeviceType(const DLDeviceType &src) {
  base::DeviceType dst;
  switch (src) {
    case kDLCPU:
      dst = base::kDeviceTypeCodeCpu;
      break;
    case kDLCUDA:
      dst = base::kDeviceTypeCodeCuda;
      break;
    case kDLOpenCL:
      dst = base::kDeviceTypeCodeOpenCL;
      break;
    case kDLVulkan:
      dst = base::kDeviceTypeCodeVulkan;
      break;
    case kDLMetal:
      dst = base::kDeviceTypeCodeMetal;
    default:
      dst = base::kDeviceTypeCodeNotSupport;
      break;
  }
  return dst;
}

DLDeviceType TvmConvert::convertFromDeviceType(const base::DeviceType &src) {
  DLDeviceType dst;
  switch (src.code_) {
    case base::kDeviceTypeCodeCpu:
      dst = kDLCPU;
      break;
    case base::kDeviceTypeCodeCuda:
      dst = kDLCUDA;
      break;
    case base::kDeviceTypeCodeOpenCL:
      dst = kDLOpenCL;
      break;
    case base::kDeviceTypeCodeVulkan:
      dst = kDLVulkan;
      break;
    case base::kDeviceTypeCodeMetal:
      dst = kDLMetal;
      break;
    default:
      NNDEPLOY_LOGE("Tvm unsupport device type: %d", src.code_);
      break;
  }
  return dst;
}

base::IntVector TvmConvert::convertToShape(
    const tvm::runtime::ShapeTuple &src) {
  base::IntVector dst(src->size);
  for (size_t i = 0; i < src->size; ++i) {
    dst[i] = src->data[i];
  }
  return dst;
}

base::SizeVector TvmConvert::convertToStride(const int64_t *strides,
                                             size_t size) {
  // strides of the tensor (in number of elements, not bytes) can be NULL,
  // indicating tensor is compact and row-majored.
  if (strides == nullptr) {
    return {};
  }
  base::SizeVector dst(size);
  for (size_t i = 0; i < size; ++i) {
    dst[i] = strides[i];
  }
  return dst;
}

device::Tensor *TvmConvert::convertToTensor(const tvm::runtime::NDArray &src,
                                            std::string name) {
  auto data_type = convertToDataType(src.DataType());
  auto data_shape = convertToShape(src.Shape());

  base::DataFormat data_format;  // TVM的NDArray中缺失data_format信息,
                                 // 根据shape大小给一个默认的format
  switch (data_shape.size()) {
    case 1:
      data_format = base::kDataFormatN;
      break;
    case 2:
      data_format = base::kDataFormatNC;
      break;
    case 3:
      data_format = base::kDataFormatNCL;
      break;
    case 4:
      data_format = base::kDataFormatNCHW;
      break;
    case 5:
      data_format = base::kDataFormatNCDHW;
      break;
    default:
      data_format = base::kDataFormatNotSupport;
      break;
  }

  auto device_type = convertToDeviceType(src->device.device_type);
  auto device = device::getDevice(device_type);
  base::SizeVector strides = convertToStride(src->strides, src->ndim);
  base::IntVector memory_config = base::IntVector();

  device::TensorDesc desc(data_type, data_format, data_shape, strides);

  device::Tensor *dst =
      new device::Tensor(device, desc, src->data, name, memory_config);

  return dst;
}

}  // namespace inference
}  // namespace nndeploy