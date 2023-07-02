#ifndef _NNDEPLOY_TASK_PRE_PROCESS_0PENCV_CVTCOLOR_RESIZE_H_
#define _NNDEPLOY_TASK_PRE_PROCESS_0PENCV_CVTCOLOR_RESIZE_H_

#include "nndeploy/base/basic.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/task/packet.h"
#include "nndeploy/task/pre_process/opencv/opencv_convert.h"
#include "nndeploy/task/pre_process/params.h"
#include "nndeploy/task/task.h"

namespace nndeploy {
namespace task {
namespace opencv {

class OpencvCvtColrResize : public Task {
 public:
  OpencvCvtColrResize(const std::string& name, Packet* input, Packet* output)
      : Task(name, input, output) {
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

}  // namespace opencv
}  // namespace task
}  // namespace nndeploy

#endif /* _NNDEPLOY_TASK_PRE_PROCESS_0PENCV_CVTCOLOR_RESIZE_H_ */
