
#ifndef _NNDEPLOY_INFERENCE_PADDLELITE_PADDLELITE_CONVERT_H_
#define _NNDEPLOY_INFERENCE_PADDLELITE_PADDLELITE_CONVERT_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/paddlelite/paddlelite_include.h"


namespace nndeploy {
namespace inference {

class PaddleLiteConvert {
 public:
  static base::DataType convertToDataType(const paddle::lite_api::PrecisionType &src);
  static paddle::lite_api::PrecisionType convertFromDataType(const base::DataType &src);

  static paddle::lite::TargetType convertFromDeviceType(const base::DeviceType &src);
  static base::DeviceType convertToDeviceType(const paddle::lite::TargetType &src);
  

  static base::DataFormat convertToDataFormat(
      const paddle::lite_api::DataLayoutType &src);

  static base::IntVector convertToShape(const paddle::lite::DDim &src);
  static paddle::lite::DDim convertFromShape(const base::IntVector &src);

};

}  // namespace inference
}  // namespace nndeploy

#endif