// Copyright (c) OpenMMLab. All rights reserved.

#ifndef _NNDEPLOY_SOURCE_TASK_DETECT_COMMON_H_
#define _NNDEPLOY_SOURCE_TASK_DETECT_COMMON_H_

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/param.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/base/string.h"
#include "nndeploy/source/base/type.h"
#include "nndeploy/source/base/value.h"
#include "nndeploy/source/device/buffer.h"
#include "nndeploy/source/device/buffer_pool.h"
#include "nndeploy/source/device/device.h"
#include "nndeploy/source/device/tensor.h"

namespace nndeploy {
namespace task {

/**
 * @brief NMS Param
 *
 */
class NMSParam : public base::Param {
 public:
  NMSParam() {}
  ~NMSParam() {}
  int32_t background_label_ = -1;
  float score_threshold_ = 0.3;
  int32_t keep_top_k_ = 100;
  float nms_eta_ = 1.0;
  float nms_threshold_ = 0.5;
  int32_t nms_top_k_ = 1000;
  bool normalized_ = true;
};

/**
 * @brief Detect Result
 *
 */
class NewDetectResult : public base::Param {
 public:
  int index_;
  int label_id_;
  float score_;
  std::array<float, 4> bbox_;  // left(x0), top(y0), right(x1), bottom(y1)
  device::Mat mask_;
};

class NewDetectResults : public base::Param {
 public:
  std::vector<DetectResult> result_;
};

/**
 * @brief 通用函数
 *
 */
std::array<float, 4> getOriginBox(float left, float top, float right,
                                  float bottom, const float* scale_factor,
                                  float x_offset, float y_offset, int ori_width,
                                  int ori_height);
// @brief Filter results using score threshold and topk candidates.
// scores (Tensor): The scores, shape (num_bboxes, K).
// probs: The scores after being filtered
// label_ids: The class labels
// anchor_idxs: The anchor indexes
void FilterScoresAndTopk(const mmdeploy::framework::Tensor& scores,
                         float score_thr, int topk, std::vector<float>& probs,
                         std::vector<int>& label_ids,
                         std::vector<int>& anchor_idxs);
float iOU(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1,
          float ymin1, float xmax1, float ymax1);

void NMS(const mmdeploy::framework::Tensor& dets, float iou_threshold,
         std::vector<int>& keep_idxs);

}  // namespace task
}  // namespace nndeploy

#endif /* _NNDEPLOY_SOURCE_TASK_DETECT_COMMON_H_ */
