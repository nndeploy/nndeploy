#ifndef _NNTASK_SOURCE_COMMON_PROCESS_0PENCV_PRE_PROCESS_H_
#define _NNTASK_SOURCE_COMMON_PROCESS_0PENCV_PRE_PROCESS_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/base/string.h"
#include "nndeploy/source/base/value.h"
#include "nndeploy/source/device/buffer.h"
#include "nndeploy/source/device/buffer_pool.h"
#include "nndeploy/source/device/device.h"
#include "nndeploy/source/device/tensor.h"
#include "nntask/source/common/execution.h"
#include "nntask/source/common/opencv_include.h"
#include "nntask/source/common/packet.h"
#include "nntask/source/common/params.h"
#include "nntask/source/common/task.h"

namespace nntask {
namespace common {

/**
 * @brief cast + normalize + premute
 *
 * @return true
 * @return false
 */
bool matToTensor(const cv::Mat& src, nndeploy::device::Tensor* dst,
                 cv::Scalar& scale, cv::Scalar& bias);

class OpenCVResizeNorm : public Execution {
 public:
  OpenCVResizeNorm(nndeploy::base::DeviceType device_type,
                   const std::string& name = "")
      : Execution(device_type, name) {}
  virtual ~OpenCVResizeNorm() {}

  virtual nndeploy::base::Status run() {
    cv::Mat* src = input_->getCvMat();
    nndeploy::device::Tensor* dst = output_->getTensor();

    int c = dst->getShapeIndex(1);
    int h = dst->getShapeIndex(2);
    int w = dst->getShapeIndex(3);

    cv::Mat tmp;
    cv::resize(*src, tmp, cv::Size(w, h));
    cv::Scalar scale;
    scale[0] = 1.0f / 255;
    scale[1] = 1.0f / 255;
    scale[2] = 1.0f / 255;
    scale[3] = 1.0f / 255;
    cv::Scalar bias;
    bias[0] = 0.0f;
    bias[1] = 0.0f;
    bias[2] = 0.0f;
    bias[3] = 0.0f;
    matToTensor(tmp, dst, scale, bias);

    return nndeploy::base::kStatusCodeOk;
  }
};

}  // namespace common
}  // namespace nntask

#endif /* _NNTASK_SOURCE_COMMON_PROCESS_0PENCV_PRE_PROCESS_H_ */
