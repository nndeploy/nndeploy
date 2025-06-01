#include "nndeploy/track/fairmot.h"

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/detect/util.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/infer/infer.h"
#include "nndeploy/preprocess/cvtcolor_resize.h"

namespace nndeploy {
namespace detect {

void FairMotPostProcess::FilterDets(const float conf_thresh,
                                    const cv::Mat& dets,
                                    std::vector<int>* index) {
  for (int i = 0; i < dets.rows; ++i) {
    float score = *dets.ptr<float>(i, 4);
    if (score > conf_thresh) {
      index->push_back(i);
    }
  }
}

base::Status FairMotPostProcess::run() {
  FairMotPostParam* param = (FairMotPostParam*)param_.get();
  float conf_thresh = param->conf_thresh_;
  float tracked_thresh = param->tracked_thresh_;
  float min_box_area = param->min_box_area_;

  auto* det_tensor = inputs_[0]->getTensor(this);
  auto* emb_tensor = inputs_[1]->getTensor(this);
  if (!det_tensor || !emb_tensor) {
    NNDEPLOY_LOGE("Invalid input tensors for FairMotPostProcess.");
    return base::kStatusCodeErrorInvalidValue;
  }

  float* bbox_data = reinterpret_cast<float*>(det_tensor->getData());
  float* emb_data = reinterpret_cast<float*>(emb_tensor->getData());

  auto bbox_shape = det_tensor->getShape();
  auto emb_shape = emb_tensor->getShape();

  int num_boxes = bbox_shape[0];
  int emb_dim = emb_shape[1];

  cv::Mat dets(num_boxes, 6, CV_32FC1, bbox_data);
  cv::Mat emb(num_boxes, emb_dim, CV_32FC1, emb_data);

  // Step 1: Filter by conf threshold
  std::vector<int> valid;
  FilterDets(conf_thresh, dets, &valid);

  cv::Mat new_dets, new_emb;
  for (int i = 0; i < valid.size(); ++i) {
    new_dets.push_back(dets.row(valid[i]));
    new_emb.push_back(emb.row(valid[i]));
  }

  std::vector<Track> tracks;
  jdeTracker_->update(new_dets, new_emb, &tracks);

  MOTResult* result = new MOTResult();
  if (tracks.empty()) {
    // fallback: 使用第一个 bbox 作为 dummy track
    std::array<int, 4> box = {
        int(dets.at<float>(0, 0)), int(dets.at<float>(0, 1)),
        int(dets.at<float>(0, 2)), int(dets.at<float>(0, 3))};
    result->boxes.push_back(box);
    result->ids.push_back(1);
    result->scores.push_back(dets.at<float>(0, 4));
  } else {
    for (auto& track : tracks) {
      if (track.score < tracked_thresh_) {
        continue;
      }
      float w = track.ltrb[2] - track.ltrb[0];
      float h = track.ltrb[3] - track.ltrb[1];
      bool vertical = w / h > 1.6;
      float area = w * h;
      if (area > min_box_area && !vertical) {
        std::array<int, 4> box = {int(track.ltrb[0]), int(track.ltrb[1]),
                                  int(track.ltrb[2]), int(track.ltrb[3])};
        result->boxes.push_back(box);
        result->ids.push_back(track.id);
        result->scores.push_back(track.score);
      }
    }
  }

  outputs_[0]->set(result, false);
  return base::kStatusCodeOk;
}

}  // namespace detect
}  // namespace nndeploy
