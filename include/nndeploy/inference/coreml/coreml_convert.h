
#ifndef _NNDEPLOY_INFERENCE_COREML_COREML_CONVERT_H_
#define _NNDEPLOY_INFERENCE_COREML_COREML_CONVERT_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/file.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/inference/coreml/coreml_include.h"
#include "nndeploy/inference/coreml/coreml_inference_param.h"
#include "nndeploy/inference/inference_param.h"

#import <CoreML/CoreML.h>
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>


namespace nndeploy {
namespace inference {

class CoremlConvert {
 public:
  // TODO: these two functions are for buffer type kind data
  static base::DataType convertToDataType(const OSType &src);
  static OSType convertFromDataType(const base::DataType &src);

  static base::DataFormat convertToDataFormat(const MLFeatureDescription &src);

  static MLFeatureDescription *convertFromDataFormat(const base::DataFormat &src);
  // You need to free it manually
  static NSObject *convertFromDeviceType(const base::DeviceType &src);

  static device::Tensor *convertToTensor(MLFeatureDescription *src, NSString *name,
                                         device::Device *device);
  static MLFeatureDescription *convertFromTensor(device::Tensor *src);

  static base::Status convertFromInferenceParam(CoremlInferenceParam *src,
                                                MLModelConfiguration *dst);
};

}  // namespace inference
}  // namespace nndeploy

#endif
