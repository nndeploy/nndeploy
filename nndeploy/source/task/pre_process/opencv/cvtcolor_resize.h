#ifndef _NNDEPLOY_SOURCE_TASK_PRE_PROCESS_0PENCV_CVTCOLOR_RESIZE_H_
#define _NNDEPLOY_SOURCE_TASK_PRE_PROCESS_0PENCV_CVTCOLOR_RESIZE_H_

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
#include "nndeploy/source/task/execution.h"
#include "nndeploy/source/task/opencv_include.h"
#include "nndeploy/source/task/packet.h"
#include "nndeploy/source/task/pre_process/opencv/opencv_convert.h"
#include "nndeploy/source/task/pre_process/params.h"
#include "nndeploy/source/task/task.h"

namespace nndeploy {
namespace task {

class OpencvCvtColrResize : public Execution {
 public:
  OpencvCvtColrResize(const std::string& name = "") : Execution(name) {
    param_ = std::make_shared<CvtclorResizeParam>();
  }
  virtual ~OpencvCvtColrResize() {}

  virtual base::Status run() {
    CvtclorResizeParam* tmp_param =
        dynamic_cast<CvtclorResizeParam*>(param_.get());
    cv::Mat* src = input_->getCvMat();
    device::Tensor* dst = output_->getTensor();

    int c = dst->getShapeIndex(1);
    int h = dst->getShapeIndex(2);
    int w = dst->getShapeIndex(3);

    cv::Mat tmp = *src;

    if (tmp_param->src_pixel_type_ != tmp_param->dst_pixel_type_) {
      base::CvtColorType cvt_type = base::calCvtColorType(
          tmp_param->src_pixel_type_, tmp_param->dst_pixel_type_);
      if (cvt_type == base::kCvtColorTypeNotSupport) {
        NNDEPLOY_LOGE("cvtColor type not support");
        return base::kStatusCodeErrorNotSupport;
      }
      int cv_cvt_type = OpencvConvert::convertFromCvtColorType(cvt_type);
      cv::cvtColor(tmp, tmp, cv_cvt_type);
    }

    if (tmp_param->interp_type_ != base::kInterpTypeNotSupport) {
      int interp_type =
          OpencvConvert::convertFromInterpType(tmp_param->interp_type_);
      cv::resize(*src, tmp, cv::Size(w, h), 0.0, 0.0, interp_type);
    }

    OpencvConvert::convertToTensor(tmp, dst, tmp_param->scale_,
                                   tmp_param->mean_, tmp_param->std_);

    return base::kStatusCodeOk;
  }
};

}  // namespace task
}  // namespace nndeploy

#endif /* _NNDEPLOY_SOURCE_TASK_PRE_PROCESS_0PENCV_CVTCOLOR_RESIZE_H_ */
