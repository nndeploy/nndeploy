
#include "nndeploy/inference/coreml/coreml_convert.h"

#import <CoreML/CoreML.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

namespace nndeploy {
namespace inference {

/**
 * @brief Convert from base::DataType to MLFeatureType
 * @review：只有这一种类型吗？
 */
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

/**
 * @review：
 */
NSObject *CoremlConvert::convertFromDeviceType(const base::DeviceType &src) {
  NSObject *type = nil;
  // if (@available(iOS 17.0, macOS 14.0)){}
  // switch (src.code_) {
  //   case base::kDeviceTypeCodeCpu:
  //   case base::kDeviceTypeCodeX86:
  //   case base::kDeviceTypeCodeArm:
  //   case base::kDeviceTypeCodeOpenCL:  // review： 这个应该属于GPU吧？或者属于default
  //     type = reinterpret_cast<NSObject *>(new MLCPUComputeDevice());
  //     break;
  //   case base::kDeviceTypeCodeOpenGL:  // review： 属于default会不会更好呀
  //   case base::kDeviceTypeCodeMetal:
  //   case base::kDeviceTypeCodeCuda:    // review： 属于default会不会更好呀
  //   case base::kDeviceTypeCodeVulkan:  // review： 属于default会不会更好呀
  //     type = reinterpret_cast<NSObject *>(new MLGPUComputeDevice());
  //     break;
  //   case base::kDeviceTypeCodeNpu:
  //     type = reinterpret_cast<NSObject *>(new MLNeuralEngineComputeDevice());
  //     break;
  //   default:
  //     type = reinterpret_cast<NSObject *>(new MLCPUComputeDevice());
  // }
  return type;
}

base::Status CoremlConvert::convertFromInferenceParam(CoremlInferenceParam *src,
                                                      MLModelConfiguration *dst) {
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
      if (@available(iOS 16.0, macOS 13.0, *)) {
        //dst.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
      }
      break;
    default:
      dst.computeUnits = MLComputeUnitsCPUOnly;
      break;
  }
  return base::kStatusCodeOk;
}

device::Tensor *CoremlConvert::convertToTensor(MLFeatureDescription *src, NSString *name,
                                               device::Device *device) {
  //这里有需要进一步确认
  
  MLFeatureType tensor_type = [src type];
  device::Tensor *dst = nullptr;
  device::TensorDesc desc;
  switch (tensor_type) {
    case MLFeatureTypeImage: {
      MLImageConstraint *image_attr = [src imageConstraint];
      base::DataType data_type = CoremlConvert::convertToDataType([image_attr pixelFormatType]);
      base::DataFormat format = base::kDataFormatNHWC;
      base::IntVector shape = {1, int([image_attr pixelsHigh]), int([image_attr pixelsWide]), 3};
      base::SizeVector stride = base::SizeVector();
      desc = device::TensorDesc(data_type, format, shape, stride);
      break;
    }
    case MLFeatureTypeDouble: {
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


// void CoremmlConvert::convertFromTensor(NSMutableDictionary) {}

}  // namespace inference
}  // namespace nndeploy
