
#ifndef _NNDEPLOY_SOURCE_INFERENCE_TENSOR_RT_TENSOR_RT_CONVERT_H_
#define _NNDEPLOY_SOURCE_INFERENCE_TENSOR_RT_TENSOR_RT_CONVERT_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/device/device.h"
#include "nndeploy/source/device/tensor.h"
#include "nndeploy/source/inference/config.h"
#include "nndeploy/source/inference/tensor_rt/tensor_rt_include.h"
#include "nndeploy/source/inference/tensor_rt/tensor_rt_config.h"
#include "nndeploy/source/inference/tensor_rt/tensor_rt_inference_impl.h"

namespace nndeploy {
namespace inference {

class TensorRtConvert {
 public:
  static base::DataType convertToDataType(const halide_type_t &src);
  static halide_type_t convertFromDataType(base::DataType &src);

  static base::DataFormat convertToDataFormat(
      const TENSOR_RT::Tensor::DimensionType &src);
  static TENSOR_RT::Tensor::DimensionType convertFromDataFormat(
      const base::DataFormat &src);

  static TENSOR_RTForwardType convertFromDeviceType(const base::DeviceType &src);

  static TENSOR_RT::BackendConfig::PowerMode convertFromPowerType(
      const base::PowerType &src);

  static TENSOR_RT::BackendConfig::PrecisionMode convertFromPowerType(
      const base::PrecisionType &src);

  static base::Status convertFromConfig(TensorRtConfigImpl *src,
                                        TENSOR_RT::ScheduleConfig *dst);

  static device::Tensor *convertToTensor(TENSOR_RT::Tensor *src, std::string name,
                                         device::Device *device);
  static TENSOR_RT::Tensor *convertFromTensor(device::Tensor *src);
};

}  // namespace inference
}  // namespace nndeploy

#endif
