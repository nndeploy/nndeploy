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

base::Status DetrPostProcess::run() {
  std::vector<Bbox> bboxes;
  Bbox bbox;
  int PROB_THRESH = 0.7;

  // TODO
  int NUM_QURREY = -1;
  int iw = -1;
  int ih = -1;

  device::Tensor* tensor_Logits = input_->getTensor(0);
  float* Logits = (float*)tensor_Logits->getPtr();
  device::Tensor* tensor_Boxes = input_->getTensor(1);
  float* Boxes = (float*)tensor_Boxes->getPtr();

  for (int i = 0; i < NUM_QURREY; i++) {
    std::vector<float> Probs;
    std::vector<float> Boxes_wh;
    for (int j = 0; j < 22; j++) {
      Probs.push_back(Logits[i * 22 + j]);
    }

    int length = Probs.size();
    std::vector<float> dst(length);

    // softmax(Probs.data(), dst.data(), length);

    auto maxPosition = std::max_element(dst.begin(), dst.end() - 1);
    // std::cout << maxPosition - dst.begin() << "  |  " << *maxPosition  <<
    // std::endl;

    if (*maxPosition < PROB_THRESH) {
      Probs.clear();
      Boxes_wh.clear();
      continue;
    } else {
      bbox.score = *maxPosition;
      bbox.cid = maxPosition - dst.begin();

      float cx = Boxes[i * 4];
      float cy = Boxes[i * 4 + 1];
      float cw = Boxes[i * 4 + 2];
      float ch = Boxes[i * 4 + 3];

      float x1 = (cx - 0.5 * cw) * iw;
      float y1 = (cy - 0.5 * ch) * ih;
      float x2 = (cx + 0.5 * cw) * iw;
      float y2 = (cy + 0.5 * ch) * ih;

      bbox.xmin = x1;
      bbox.ymin = y1;
      bbox.xmax = x2;
      bbox.ymax = y2;

      bboxes.push_back(bbox);

      Probs.clear();
      Boxes_wh.clear();
    }
  }
  DetectResult* result = (DetectResult*)output_->getParam();

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
  pre_param->mean_[0] = 0.0f;
  pre_param->mean_[1] = 0.0f;
  pre_param->mean_[2] = 0.0f;
  pre_param->mean_[3] = 0.0f;
  pre_param->std_[0] = 255.0f;
  pre_param->std_[1] = 255.0f;
  pre_param->std_[2] = 255.0f;
  pre_param->std_[3] = 255.0f;

  return task;
}

}  // namespace task
}  // namespace nndeploy
