
#ifndef _NNDEPLOY_INFERENCE_RKNN_RKNN_CONVERT_H_
#define _NNDEPLOY_INFERENCE_RKNN_RKNN_CONVERT_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/rknn/rknn_include.h"
#include "nndeploy/inference/rknn/rknn_inference_param.h"

namespace nndeploy {
namespace inference {

class RknnConvert {
 public:
  static base::DataType convertToDataType(const rknn_tensor_type &src);

  static base::DataFormat convertToDataFormat(const rknn_tensor_format &src);

  static base::IntVector convertToShape(
      const rknn_tensor_attr &src,
      const rknn_tensor_format &dst_fmt = RKNN_TENSOR_FORMAT_MAX);
};

}  // namespace inference
}  // namespace nndeploy

#endif
