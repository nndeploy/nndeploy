
#ifndef _NNDEPLOY_SOURCE_INFERENCE_TENSOR_RT_TENSOR_RT_CONVERT_H_
#define _NNDEPLOY_SOURCE_INFERENCE_TENSOR_RT_TENSOR_RT_CONVERT_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/device/device.h"
#include "nndeploy/source/device/tensor.h"
#include "nndeploy/source/inference/inference_param.h"
#include "nndeploy/source/inference/tensor_rt/tensor_rt_include.h"
#include "nndeploy/source/inference/tensor_rt/tensor_rt_inference.h"
#include "nndeploy/source/inference/tensor_rt/tensor_rt_inference_param.h"

namespace nndeploy {
namespace inference {

class TensorRtConvert {
 public:
  static base::DataType convertToDataType(const nvinfer1::DataType &src);
  static nvinfer1::DataType convertFromDataType(base::DataType &src);

  static base::DataFormat convertToDataFormat(const nvinfer1::TensorFormat &src)

      static base::IntVector convertToShape(const nvinfer1::Dims &src);
  static nvinfer1::Dims convertFromShape(const base::IntVector &src);
};

}  // namespace inference
}  // namespace nndeploy

#endif
