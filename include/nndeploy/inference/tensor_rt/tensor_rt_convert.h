
#ifndef _NNDEPLOY_INFERENCE_TENSOR_RT_TENSOR_RT_CONVERT_H_
#define _NNDEPLOY_INFERENCE_TENSOR_RT_TENSOR_RT_CONVERT_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/tensor_rt/tensor_rt_include.h"
#include "nndeploy/inference/tensor_rt/tensor_rt_inference_param.h"

namespace nndeploy {
namespace inference {

class TensorRtConvert {
 public:
  static base::DataType convertToDataType(const nvinfer1::DataType &src);
  static nvinfer1::DataType convertFromDataType(base::DataType &src);

  static base::DataFormat convertToDataFormat(
      const nvinfer1::TensorFormat &src);

  static base::IntVector convertToShape(const nvinfer1::Dims &src);
  static nvinfer1::Dims convertFromShape(const base::IntVector &src);
};

}  // namespace inference
}  // namespace nndeploy

#endif
