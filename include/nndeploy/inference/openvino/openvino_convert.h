
#ifndef _NNDEPLOY_INFERENCE_OPENVINO_OPENVINO_CONVERT_H_
#define _NNDEPLOY_INFERENCE_OPENVINO_OPENVINO_CONVERT_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/openvino/openvino_include.h"
#include "nndeploy/inference/openvino/openvino_inference_param.h"

namespace nndeploy {
namespace inference {

class OpenVinoConvert {
 public:
  static std::string OpenVinoConvert::convertFromDeviceType(
      const base::device &src, const base::device &srcs);
  static base::DataType convertToDataType(const ov::element::Type &src);
  static ov::element::Type convertFromDataType(base::DataType &src);

  static base::DataFormat getDataFormatByShape(const base::IntVector &src);

  static base::IntVector convertToShape(const ov::PartialShape &src);
  static ov::PartialShape convertFromShape(const base::IntVector &src);

  static base::SizeVector convertToStride(const ov::Strides &src);
  static ov::Strides convertFromStride(const base::SizeVector &src);

  static base::Status convertFromInferenceParam(OpenVinoInferenceParam *src,
                                                std::string &dst_device_type,
                                                ov::AnyMap &dst_properties);
  static ov::Tensor convertFromTensor(device::Tensor *src);
  static device::Tensor *convertToTensor(ov::Tensor &src);
};

}  // namespace inference
}  // namespace nndeploy

#endif
