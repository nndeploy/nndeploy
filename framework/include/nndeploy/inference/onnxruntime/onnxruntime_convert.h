
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
  static base::DataType convertToDataType(const ONNXTensorElementDataType &src);
  static ONNXTensorElementDataType convertFromDataType(
      const base::DataType &src);

  static base::DataFormat getDataFormatByShape(const base::IntVector &src);

  static base::IntVector convertToShape(
      std::vector<int64_t> &src, base::IntVector max_shape = base::IntVector());
  static std::vector<int64_t> convertFromShape(const base::IntVector &src);

  // 会衍生出其他需要进行转换的类型
  static base::Status convertFromInferenceParam(
      OnnxRuntimeInferenceParam &src, Ort::SessionOptions &dst,
      device::Stream *stream = nullptr);

  static base::Status convertToTensor(Ort::Value &src, const std::string &name,
                                      device::Device *device,
                                      device::Tensor *dst);
  static Ort::Value convertFromTensor(device::Tensor *src);
};

}  // namespace inference
}  // namespace nndeploy

#endif
