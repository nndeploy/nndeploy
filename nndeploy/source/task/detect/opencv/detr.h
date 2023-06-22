/**
 * @model source: git@github.com:DataXujing/TensorRT-DETR.git
 */
#ifndef _NNDEPLOY_SOURCE_DETECT_0PENCV_DETR_H_
#define _NNDEPLOY_SOURCE_DETECT_0PENCV_DETR_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/param.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/base/string.h"
#include "nndeploy/source/base/value.h"
#include "nndeploy/source/device/buffer.h"
#include "nndeploy/source/device/buffer_pool.h"
#include "nndeploy/source/device/device.h"
#include "nndeploy/source/device/tensor.h"
#include "nndeploy/source/task/detect/result.h"
#include "nndeploy/source/task/execution.h"
#include "nndeploy/source/task/opencv_include.h"
#include "nndeploy/source/task/packet.h"
#include "nndeploy/source/task/task.h"

namespace nndeploy {
namespace task {

class DetrPostParam : public base::Param {
  int32_t NUM_CLASS = 22;
  int32_t NUM_QURREY = 100;  // detr默认是100
  float PROB_THRESH = 0.7f;
};

class DetrPostProcess : public Execution {
 public:
  DetrPostProcess(const std::string& name = "") : Execution(name) {}
  virtual ~DetrPostProcess() {}

  virtual base::Status run() {
    // TODO
    int NUM_QURREY = -1;
    int iw = -1;
    int ih = -1;

    device::Tensor* tensor_logits = input_->getTensor(0);
    float* logits = (float*)tensor_logits->getPtr();
    device::Tensor* tensor_boxes = input_->getTensor(1);
    float* boxes = (float*)tensor_boxes->getPtr();

    for (int i = 0; i < NUM_QURREY; i++) {
      std::vector<float> Probs;
      std::vector<float> boxes_wh;
      for (int j = 0; j < 22; j++) {
        Probs.push_back(logits[i * 22 + j]);
      }

      int length = Probs.size();
      std::vector<float> dst(length);

      // softmax(Probs.data(), dst.data(), length);

      auto maxPosition = std::max_element(dst.begin(), dst.end() - 1);
      // std::cout << maxPosition - dst.begin() << "  |  " << *maxPosition  <<
      // std::endl;

      if (*maxPosition < PROB_THRESH) {
        Probs.clear();
        boxes_wh.clear();
        continue;
      } else {
        bbox.score = *maxPosition;
        bbox.cid = maxPosition - dst.begin();

        float cx = boxes[i * 4];
        float cy = boxes[i * 4 + 1];
        float cw = boxes[i * 4 + 2];
        float ch = boxes[i * 4 + 3];

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
        boxes_wh.clear();
      }
    }
    DetectResult* result = (DetectResult*)output_->getParam();

    return base::kStatusCodeOk;
  }
};

task::Task* creatDetrTask(const std::string& name, base::InferenceType type);

}  // namespace task
}  // namespace nndeploy

#endif /* _NNDEPLOY_SOURCE_DETECT_0PENCV_POST_PROCESS_H_ */
