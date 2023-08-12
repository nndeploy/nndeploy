/**
 * @brief tnn的数据类型与nndeploy的数据类型互相转换
 * tnn_convert主要作用于nndeploy到TNN和反之的各种枚举类型转换，包括DataType,DataFormat,DeviceType,ModelType,ShareMemoryMode,PrecisionType的转换
 * 主要目的是对tnn_inference进行辅助，所以也包括从inferen_param到TNN的instence需要的modle_config和instence_config的转换函数
 * 以及tensor到TNN的blob和mat类型的转换函数
 * 具体可以参考TNN的include路径下的所有.h文件
 */
#ifndef _NNDEPLOY_INFERENCE_TNN_TNN_CONVERT_H_
#define _NNDEPLOY_INFERENCE_TNN_TNN_CONVERT_H_

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
#include "nndeploy/inference/tnn/tnn_include.h"
#include "nndeploy/inference/tnn/tnn_inference_param.h"

namespace nndeploy {
namespace inference {

/**
 * @brief tnn的数据类型与nndeploy的数据类型互相转换
 * @数据类型
 * 1、推理配置的数据类型，例如 推理设备、几线程等、推理精度等
 * 2、推理数据，例如 推理输入Blob\Mat、推理输出Blob\Mat 到 nndeploy的Tensor
 * @步骤
 * 1、熟悉tnn的数据类型
 * 2、熟悉nndeploy的数据类型
 * 3、转换
 */
class TnnConvert {
 public:
  static base::DataType convertToDataType(const tnn::DataType &src);
  static tnn::DataType convertFromDataType(const base::DataType &src);

  static base::DataFormat convertToDataFormat(const tnn::DataFormat &src);
  static tnn::DataFormat convertFromDataFormat(const base::DataFormat &src);

  static base::DeviceType convertToDeviceType(const tnn::DeviceType &src);
  static tnn::DeviceType convertFromDeviceType(const base::DeviceType &src);

  static base::ShareMemoryType convertToShareMemoryMode(
      const tnn::ShareMemoryMode &src);
  static tnn::ShareMemoryMode convertFromShareMemoryMode(
      const base::ShareMemoryType &src);

  static base::ModelType convertToModelType(const tnn::ModelType &src);
  static tnn::ModelType convertFromModelType(const base::ModelType &src);

  static tnn::Precision convertFromPrecisionType(
      const base::PrecisionType &src);

  static base::Status convertFromInferenceParam(
      inference::TnnInferenceParam *src, tnn::ModelConfig &model_config_,
      tnn::NetworkConfig &network_config_);

  static device::Tensor *matConvertToTensor(tnn::Mat *src, std::string name);
  static tnn::Mat *matConvertFromTensor(device::Tensor *src);
  static device::Tensor *blobConvertToTensor(tnn::Blob *src);
};

}  // namespace inference
}  // namespace nndeploy

#endif
