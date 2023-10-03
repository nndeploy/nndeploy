
#include "nndeploy/inference/coreml/coreml_convert.h"

namespace nndeploy {
namespace inference {

base::DataType CoremlConvert::convertToDataType(const OSType &src) {
  base::DataType dst;
  switch (src) {
    case kCVPixelFormatType_OneComponent8:
    case kCVPixelFormatType_32BGRA:
    case kCVPixelFormatType_32RGBA:
      dst.code_ = base::kDataTypeCodeInt;
      dst.bits_ = 8;
      break;
    default:
      break;
  }
  return dst;
}

NSObject *CoremlConvert::convertFromDeviceType(const base::DeviceType &src) {
  NSObject *type = nil;
  switch (src.code_) {
    case base::kDeviceTypeCodeCpu:
    case base::kDeviceTypeCodeX86:
    case base::kDeviceTypeCodeArm:
    case base::kDeviceTypeCodeOpenCL:
      type = reinterpret_cast<NSObject *>(new MLCPUComputeDevice());
      break;
    case base::kDeviceTypeCodeOpenGL:
    case base::kDeviceTypeCodeMetal:
    case base::kDeviceTypeCodeCuda:
    case base::kDeviceTypeCodeVulkan:
      type = reinterpret_cast<NSObject *>(new MLGPUComputeDevice());
      break;
    case base::kDeviceTypeCodeNpu:
      type = reinterpret_cast<NSObject *>(new MLNeuralEngineComputeDevice());
      break;
    default:
      type = reinterpret_cast<NSObject *>(new MLCPUComputeDevice());
  }
  return type;
}

base::Status CoremlConvert::convertFromInferenceParam(
    CoremlInferenceParam *src, MLModelConfiguration *dst) {
  dst.allowLowPrecisionAccumulationOnGPU = src->low_precision_acceleration_;
  switch (src->inference_units_) {
    case CoremlInferenceParam::inferenceUnits::ALL_UNITS:
      dst.computeUnits = MLComputeUnitsAll;
      break;
    case CoremlInferenceParam::inferenceUnits::CPU_ONLY:
      dst.computeUnits = MLComputeUnitsCPUOnly;
      break;
    case CoremlInferenceParam::inferenceUnits::CPU_AND_GPU:
      dst.computeUnits = MLComputeUnitsCPUAndGPU;
      break;
    case CoremlInferenceParam::inferenceUnits::CPU_AND_NPU:
      dst.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
      break;
    default:
      dst.computeUnits = MLComputeUnitsCPUOnly;
      break;
  }
  return base::kStatusCodeOk;
}

device::Tensor *CoremlConvert::convertToTensor(MLFeatureDescription *src, NSString *name,
                                device::Device *device) {
  MLFeatureType tensor_type = [src type];
  device::Tensor *dst = nullptr;
  device::TensorDesc desc;
  switch (tensor_type) {
    case MLFeatureTypeImage:
    {
      MLImageConstraint *image_attr = [src imageConstraint];
      base::DataType data_type = CoremlConvert::convertToDataType([image_attr pixelFormatType]);
      base::DataFormat format = base::kDataFormatNHWC;
      base::IntVector shape = {1, int([image_attr pixelsHigh]), int ([image_attr pixelsWide]), 3};
      base::SizeVector stride = base::SizeVector();
      desc = device::TensorDesc(data_type, format, shape, stride);
      break;
    }
    case MLFeatureTypeDouble:
    {
      base::DataType data_type = base::DataType();
      base::DataFormat format = base::kDataFormatN;
      base::IntVector shape = {1};
      base::SizeVector stride = base::SizeVector();
      desc = device::TensorDesc(data_type, format, shape, stride);
      break;
    }
    default:
      break;
  }
  dst = new device::Tensor(desc, std::string([name cStringUsingEncoding:NSASCIIStringEncoding]));
  return dst;
}

}  // namespace inference
}  // namespace nndeploy
