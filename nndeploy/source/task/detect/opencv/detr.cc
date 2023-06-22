#include "nndeploy/source/task/detect/opencv/detr.h"

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
#include "nndeploy/source/task/pre_process/opencv/cvtcolor_resize.h"
#include "nndeploy/source/task/task.h"

namespace nndeploy {
namespace task {

template <typename T>
int softmax(const T* src, T* dst, int length) {
  T denominator{0};
  for (int i = 0; i < length; ++i) {
    dst[i] = std::exp(src[i]);
    denominator += dst[i];
  }
  for (int i = 0; i < length; ++i) {
    dst[i] /= denominator;
  }
  return 0;
}

base::Status DetrPostProcess::run() {
  results_.result_.clear();

  DetrPostParam* temp_param = (DetrPostParam*)param_.get();
  device::Tensor* tensor_logits = input_->getTensor(0);
  float* logits = (float*)tensor_logits->getPtr();
  device::Tensor* tensor_boxes = input_->getTensor(1);
  float* boxes = (float*)tensor_boxes->getPtr();

  for (int i = 0; i < temp_param->num_qurrey_; i++) {
    std::vector<float> scores(temp_param->num_class_);
    softmax(logits + i * temp_param->num_class_, scores.data(),
            temp_param->num_class_);

    auto maxPosition = std::max_element(scores.begin(), scores.end() - 1);
    if (*maxPosition < score_threshold_) {
      continue;
    } else {
      DetectResult result;
      result.score_ = *maxPosition;
      result.label_id_ = maxPosition - dst.begin();

      float cx = boxes[i * 4];
      float cy = boxes[i * 4 + 1];
      float cw = boxes[i * 4 + 2];
      float ch = boxes[i * 4 + 3];

      result.bbox_[0] = (cx - 0.5 * cw);
      result.bbox_[1] = (cy - 0.5 * ch);
      result.bbox_[2] = (cx + 0.5 * cw);
      result.bbox_[3] = (cy + 0.5 * ch);

      results_.emplace_back(result);
    }
  }
  output_->setParam(results_);

  return base::kStatusCodeOk;
}

task::Task* creatDetrTask(const std::string& name, base::InferenceType type) {
  task::Task* task = new task::Task(name, type);
  task->createPreprocess<OpencvCvtColrResize>();
  task->createPostprocess<DetrPostProcess>();

  CvtclorResizeParam* pre_param =
      dynamic_cast<CvtclorResizeParam*>(task->getPreProcessParam());
  pre_param->src_pixel_type_ = base::kPixelTypeBGR;
  pre_param->dst_pixel_type_ = base::kPixelTypeBGR;
  pre_param->interp_type_ = base::kInterpTypeLinear;

  pre_param->mean_[0] = 0.485f;
  pre_param->mean_[1] = 0.456f;
  pre_param->mean_[2] = 0.406f;

  pre_param->std_[0] = 0.229f;
  pre_param->std_[1] = 0.224f;
  pre_param->std_[2] = 0.225f;

  return task;
}

}  // namespace task
}  // namespace nndeploy
