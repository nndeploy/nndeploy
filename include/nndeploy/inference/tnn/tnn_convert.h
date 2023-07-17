/**
 * @brief tnn的数据类型与nndeploy的数据类型互相转换
 *
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
  static base::DataType convertToDataType(const TNN_NS::DataType &src);
  static TNN_NS::DataType convertFromDataType(base::DataType &src);

  static base::DataFormat convertToDataFormat(const TNN_NS::DataFormat &src);
  static TNN_NS::DataFormat convertFromDataFormat(const base::DataFormat &src);

  static TNN_NS::DeviceType convertFromDeviceType(const base::DeviceType &src);

  static TNN_NS::ShareMemoryMode convertFromShareMemoryMode(
      const base::ShareMemoryType &src);

  static TNN_NS::Precision convertFromPrecisionType(
      const base::PrecisionType &src);

  static base::Status convertFromInferenceParam(TnnInferenceParam *src,
                                                TNN_NS::ModelConfig *model_dst,
                                                TNN_NS::NetworkConfig *net_dst);

  static device::Tensor *convertToTensor(TNN_NS::Blob *src);
  static TNN_NS::Blob *convertFromTensor(device::Tensor *src);

  static device::Tensor *convertToTensor(TNN_NS::Mat *src);
  static TNN_NS::Mat *convertFromTensor(device::Tensor *src);
};

}  // namespace inference
}  // namespace nndeploy

#endif
