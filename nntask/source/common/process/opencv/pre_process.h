#ifndef _NNTASK_SOURCE_COMMON_PROCESS_GENERAL_H_
#define _NNTASK_SOURCE_COMMON_PROCESS_GENERAL_H_

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
#include "nntask/source/common/task.h"

namespace nntask {
namespace common {

/**
 * @brief
 * cvtcolor
 * resize
 * padding
 * warp_affine
 * crop
 */

/**
 * @brief cast + normalize + premute
 *
 * @return true
 * @return false
 */
bool matToTensor(const cv::Mat& src, nndeploy::device::Tensor* dst,
                 std::vector<float> scale, std::vector<float> bias);

class CvtclorResizeParam : public nndeploy::base::Param {
 public:
  nndeploy::base::PixelType src_pix_type_code_;
  nndeploy::base::PixelType dst_pix_type_code_;
};

class OpenCVCvtclorResizeNorm : public Execution {
 public:
  ResizeBn(nndeploy::base::DeviceType device_type, const std::string& name = "")
      : Execution(device_type, name) {
    param_ = new OpenCVResizeBnParam();
  };
  virtual ~ResizeBn(){};

  virtual nndeploy::base::Status run() {
    cv::mat* src = input_->getCvMat();
    nndeploy::device::Tensor* dst = output_->getTensor();

    int c = dst->getShapeIndex[1];
    int h = dst->getShapeIndex[2];
    int w = dst->getShapeIndex[3];

    cv::mat tmp;
    cv::resize(*src, tmp, cv::Size(w, h));

    tmp.convertTo(tmp, CV_32FC3);
    tmp = tmp / 255.0f;

    std::vector<cv::mat> tmp_vec;
    for (int i = 0; i < c; ++i) {
      float* data = (float*)dst->getPtr() + w * h * i;
      cv::mat tmp(cv::Size(w, h), CV_32FC1, data);
    }

    cv::split(tmp, tmp_vec);
  }
};

}  // namespace common
}  // namespace nntask

#endif /* _NNTASK_SOURCE_COMMON_PROCESS_GENERAL_H_ */
