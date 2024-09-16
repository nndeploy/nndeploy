#ifndef _NNDEPLOY_INFERENCE_TVM_TVM_CONVERT_H_
#define _NNDEPLOY_INFERENCE_TVM_TVM_CONVERT_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/status.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/inference/inference_param.h"
#include "nndeploy/inference/tvm/tvm_include.h"
#include "nndeploy/inference/tvm/tvm_inference_param.h"

namespace nndeploy {
namespace inference {

class TvmConvert {
 public:
  static base::DataType convertToDataType(const tvm::runtime::DataType &src);
  static tvm::runtime::DataType convertFromDataType(const base::DataType &src);

  // tvm中使用string存储dataformat
  static base::DataFormat convertToDataFormat(const std::string &src);
  static std::string convertFromDataFormat(const base::DataFormat &src);

  static base::DeviceType convertToDeviceType(const DLDeviceType &src);
  static DLDeviceType convertFromDeviceType(const base::DeviceType &src);

  static device::Tensor *convertToTensor(const tvm::runtime::NDArray &src,
                                         std::string name);

  static base::IntVector convertToShape(const tvm::runtime::ShapeTuple &src);
  static base::SizeVector convertToStride(const int64_t *strides, size_t size);
};
}  // namespace inference
}  // namespace nndeploy

#endif