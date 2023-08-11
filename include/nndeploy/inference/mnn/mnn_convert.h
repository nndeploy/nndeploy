
#ifndef _NNDEPLOY_INFERENCE_MNN_MNN_CONVERT_H_
#define _NNDEPLOY_INFERENCE_MNN_MNN_CONVERT_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/file.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/mnn/mnn_include.h"
#include "nndeploy/inference/mnn/mnn_inference_param.h"

namespace nndeploy {
namespace inference {

class MnnConvert {
 public:
  static base::DataType convertToDataType(const halide_type_t &src);
  static halide_type_t convertFromDataType(const base::DataType &src);

  static base::DataFormat convertToDataFormat(
      const MNN::Tensor::DimensionType &src);
  static MNN::Tensor::DimensionType convertFromDataFormat(
      const base::DataFormat &src);

  static MNNForwardType convertFromDeviceType(const base::DeviceType &src);

  static MNN::BackendConfig::PowerMode convertFromPowerType(
      const base::PowerType &src);

  static MNN::BackendConfig::PrecisionMode convertFromPrecisionType(
      const base::PrecisionType &src);

  static base::Status convertFromInferenceParam(MnnInferenceParam *src,
                                                MNN::ScheduleConfig *dst);

  static device::Tensor *convertToTensor(MNN::Tensor *src, std::string name,
                                         device::Device *device);
  static MNN::Tensor *convertFromTensor(device::Tensor *src);
};

}  // namespace inference
}  // namespace nndeploy

#endif
