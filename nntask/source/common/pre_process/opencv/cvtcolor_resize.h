#ifndef _NNTASK_SOURCE_COMMON_PRE_PROCESS_0PENCV_CVTCOLOR_RESIZE_H_
#define _NNTASK_SOURCE_COMMON_PRE_PROCESS_0PENCV_CVTCOLOR_RESIZE_H_

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
namespace opencv {

class CvtColrResize : public Execution {
 public:
  OpenCVResizeNorm(nndeploy::base::DeviceType device_type,
                   const std::string& name = "")
      : Execution(device_type, name) {
    param_ = std::make_shared<CvtclorResizeParam>();
  }
  virtual ~OpenCVResizeNorm() {}

  virtual nndeploy::base::Status run() {
    CvtclorResizeParam* tmp_param =
        dynamic_cast<CvtclorResizeParam*>(param_.get());
    cv::Mat* src = input_->getCvMat();
    nndeploy::device::Tensor* dst = output_->getTensor();

    int c = dst->getShapeIndex(1);
    int h = dst->getShapeIndex(2);
    int w = dst->getShapeIndex(3);

    cv::Mat tmp = *src;

    int cv_cvt_type = cv::CV;
    if () {
      cv::cvtcolor(tmp, tmp, cv_cvt_type);
    }

    if () {
      cv::resize(*src, tmp, cv::Size(w, h));
    }

    cv::Scalar mean;
    cv::Scalar std;
    for (int i = 0; i < 4; ++i) {
      mean[i] = tmp_param->mean_[i];
      std = tmp_param->std_[i];
    }
    OpencvConvert::convertToTensor(tmp, dst, mean, std);

    return nndeploy::base::kStatusCodeOk;
  }
};

}  // namespace opencv
}  // namespace common
}  // namespace nntask

#endif /* _NNTASK_SOURCE_COMMON_PRE_PROCESS_0PENCV_CVTCOLOR_RESIZE_H_ */
