
#ifndef _NNDEPLOY_INFERENCE_ONNXRUNTIME_ONNXRUNTIME_CONVERT_H_
#define _NNDEPLOY_INFERENCE_ONNXRUNTIME_ONNXRUNTIME_CONVERT_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/onnxruntime/onnxruntime_include.h"
#include "nndeploy/inference/onnxruntime/onnxruntime_inference_param.h"

namespace nndeploy {
namespace inference {

class OnnxRuntimeConvert {
 public:
  static base::DataType convertToDataType(const nvinfer1::DataType &src);
  static nvinfer1::DataType convertFromDataType(base::DataType &src);

  static base::DataFormat convertToDataFormat(
      const nvinfer1::TensorFormat &src);

  static base::IntVector convertToShape(TensorTypeAndShapeInfo &src,
                                        bool is_dynamic_shape = false,
                                        IntVector &max_shape = IntVector());
  static nvinfer1::Dims convertFromShape(const base::IntVector &src);

  // 会衍生出其他需要进行转换的类型
  static base::Status convertFromInferenceParam(OnnxRuntimeInferenceParam *src,
                                                Ort::SessionOptions *dst);

  static device::Tensor *convertToTensor(Ort::Value &src, std::string name,
                                         device::Device *device);
  static Ort::Value convertFromTensor(device::Tensor *src);
};

}  // namespace inference
}  // namespace nndeploy

#endif
