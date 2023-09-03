
#ifndef _NNDEPLOY_INFERENCE_NCNN_NCNN_CONVERT_H_
#define _NNDEPLOY_INFERENCE_NCNN_NCNN_CONVERT_H_

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
#include "nndeploy/inference/ncnn/ncnn_include.h"
#include "nndeploy/inference/ncnn/ncnn_inference_param.h"

namespace nndeploy {
namespace inference {

class NcnnConvert {
 public:
  static base::DataType convertToDataType(const int &src);
  static base::DataFormat convertToDataFormat(const int &elempack,
                                              const int &dims, const int &w,
                                              const int &h, const int &d,
                                              const int &c,
                                              const size_t &cstep);
  static base::IntVector convertToShape(const int &dims, const int &w,
                                        const int &h, const int &d,
                                        const int &c, const size_t &cstep);

  static base::Status convertFromInferenceParam(
      inference::NcnnInferenceParam *src, ncnn::Option &dst);

  /**
   * @brief 浅拷贝
   *
   * @param src
   * @param dst
   * @return base::Status
   */
  static base::Status matConvertToTensor(ncnn::Mat &src,
                                         const std::string &name,
                                         device::Tensor *dst);
  /**
   * @brief 浅拷贝
   *
   * @param src
   * @return ncnn::Mat
   */
  static ncnn::Mat matConvertFromTensor(device::Tensor *src);
  /**
   * @brief 浅拷贝
   *
   * @param src
   * @return device::Tensor*
   */
  static device::Tensor *blobConvertToTensor(ncnn::Blob &src);
};

}  // namespace inference
}  // namespace nndeploy

#endif
