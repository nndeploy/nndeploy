#include "nndeploy/keypoint/yolo_pose/yolo_pose.h"

#include <cmath>
#include <vector>

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
#include "nndeploy/preprocess/cvt_resize_norm_trans.h"

namespace nndeploy {
namespace keypoint {

base::Status KeypointPostParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.AddMember("score_threshold_", score_threshold_, allocator);
  json.AddMember("nms_threshold_", nms_threshold_, allocator);
  json.AddMember("num_classes_", num_classes_, allocator);
  json.AddMember("num_keypoints_", num_keypoints_, allocator);
  json.AddMember("model_h_", model_h_, allocator);
  json.AddMember("model_w_", model_w_, allocator);
  json.AddMember("version_", version_, allocator);
  return base::kStatusCodeOk;
}

base::Status KeypointPostParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("score_threshold_") &&
      json["score_threshold_"].IsFloat()) {
    score_threshold_ = json["score_threshold_"].GetFloat();
  }
  if (json.HasMember("nms_threshold_") && json["nms_threshold_"].IsFloat()) {
    nms_threshold_ = json["nms_threshold_"].GetFloat();
  }
  if (json.HasMember("num_classes_") && json["num_classes_"].IsInt()) {
    num_classes_ = json["num_classes_"].GetInt();
  }
  if (json.HasMember("num_keypoints_") && json["num_keypoints_"].IsInt()) {
    num_keypoints_ = json["num_keypoints_"].GetInt();
  }
  if (json.HasMember("model_h_") && json["model_h_"].IsInt()) {
    model_h_ = json["model_h_"].GetInt();
  }
  if (json.HasMember("model_w_") && json["model_w_"].IsInt()) {
    model_w_ = json["model_w_"].GetInt();
  }
  if (json.HasMember("version_") && json["version_"].IsInt()) {
    version_ = json["version_"].GetInt();
  }
  return base::kStatusCodeOk;
}

struct Detection {
  int index_ = 0;
  int label_id_ = 0;
  float score_ = 0.0f;
  std::array<float, 4> bbox_ = {0, 0, 0, 0};
  std::vector<KeypointKeyPoint> keypoints_;
};

static void computeKeypointNMS(std::vector<Detection>& results,
                               std::vector<int>& keep_idxs,
                               float nms_threshold) {
  keep_idxs.resize(results.size());
  for (int i = 0; i < (int)results.size(); i++) {
    keep_idxs[i] = i;
  }
  if (results.empty()) {
    return;
  }
  std::sort(keep_idxs.begin(), keep_idxs.end(), [&results](int a, int b) {
    return results[a].score_ > results[b].score_;
  });

  std::vector<float> areas(results.size());
  for (int i = 0; i < (int)results.size(); i++) {
    float w = results[i].bbox_[2] - results[i].bbox_[0];
    float h = results[i].bbox_[3] - results[i].bbox_[1];
    areas[i] = w * h;
  }

  for (int i = 0; i < (int)keep_idxs.size(); i++) {
    int idx_a = keep_idxs[i];
    if (idx_a < 0) {
      continue;
    }
    for (int j = i + 1; j < (int)keep_idxs.size(); j++) {
      int idx_b = keep_idxs[j];
      if (idx_b < 0) {
        continue;
      }
      float x1 = std::max(results[idx_a].bbox_[0], results[idx_b].bbox_[0]);
      float y1 = std::max(results[idx_a].bbox_[1], results[idx_b].bbox_[1]);
      float x2 = std::min(results[idx_a].bbox_[2], results[idx_b].bbox_[2]);
      float y2 = std::min(results[idx_a].bbox_[3], results[idx_b].bbox_[3]);
      float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
      float iou = inter / (areas[idx_a] + areas[idx_b] - inter);
      if (iou > nms_threshold) {
        keep_idxs[j] = -1;
      }
    }
  }
}

base::Status KeypointPostProcess::run() {
  KeypointPostParam* param = (KeypointPostParam*)param_.get();
  float score_threshold = param->score_threshold_;
  int num_keypoints = param->num_keypoints_;

  device::Tensor* tensor = inputs_[0]->getTensor(this);
  float* data = (float*)tensor->getData();
  int batch = tensor->getShapeIndex(0);
  int dim1 = tensor->getShapeIndex(1);
  int dim2 = tensor->getShapeIndex(2);

  detect::BBoxResult* bbox_result = new detect::BBoxResult();
  KeypointResult* kp_result = new KeypointResult();

  if (param->version_ == 26) {
    // yolo26 NMS-free format: [batch, num_candidates, fields]
    //   e.g. yolo26n-pose: [1, 300, 57]
    //   fields: [cx, cy, w, h, score, class_id,
    //            kp0_x, kp0_y, kp0_c, ..., kp16_x, kp16_y, kp16_c]
    //   kp starts at field index 6 = 5(bbox) + 1(score) + 1(class_id)
    float model_w_f = static_cast<float>(param->model_w_);
    float model_h_f = static_cast<float>(param->model_h_);
    int num_candidates = dim1;
    int fields = dim2;

    for (int b = 0; b < batch; ++b) {
      float* batch_data = data + b * num_candidates * fields;
      for (int i = 0; i < num_candidates; ++i) {
        float* row = batch_data + i * fields;
        // yolo26 format: [x1, y1, x2, y2, score, class_id, kp0_x, kp0_y, kp0_c,
        // ...] bbox is in raw pixel xyxy (not cxcywh), score and kp_c are
        // already in [0,1] range (not raw logits)
        float score = row[4];
        if (score < score_threshold) {
          continue;
        }
        // bbox is in raw pixel coordinates (x1, y1, x2, y2), normalize
        float x1 = row[0] / model_w_f;
        float y1 = row[1] / model_h_f;
        float x2 = row[2] / model_w_f;
        float y2 = row[3] / model_h_f;
        x1 = std::max(0.0f, std::min(1.0f, x1));
        y1 = std::max(0.0f, std::min(1.0f, y1));
        x2 = std::max(0.0f, std::min(1.0f, x2));
        y2 = std::max(0.0f, std::min(1.0f, y2));

        // BBox output
        detect::BBox bbox_item;
        bbox_item.index_ = b;
        bbox_item.label_id_ = static_cast<int>(row[5]);
        bbox_item.score_ = score;
        bbox_item.bbox_[0] = x1;
        bbox_item.bbox_[1] = y1;
        bbox_item.bbox_[2] = x2;
        bbox_item.bbox_[3] = y2;
        bbox_result->bboxs_.push_back(std::move(bbox_item));

        // Keypoint output
        KpSkeleton skel;
        skel.index_ = b;
        skel.label_id_ = static_cast<int>(row[5]);
        skel.score_ = score;
        int kp_offset = 6;
        for (int k = 0; k < num_keypoints; ++k) {
          KeypointKeyPoint kp;
          kp.x_ = row[kp_offset + k * 3] / model_w_f;
          kp.y_ = row[kp_offset + k * 3 + 1] / model_h_f;
          kp.confidence_ = row[kp_offset + k * 3 + 2];
          skel.keypoints_.push_back(kp);
        }
        kp_result->skeletons_.push_back(std::move(skel));
      }
    }
  } else {
    // v8/v11 dense format: [batch, channels, num_predictions]
    //   e.g. yolo11n-pose: [1, 56, 8400]
    //   channels = 5 (cx,cy,w,h,conf) + num_kps*3
    int channels = dim1;
    int num_predictions = dim2;

    cv::Mat cv_mat_src(channels, num_predictions, CV_32FC1, data);
    cv::Mat cv_mat_dst(num_predictions, channels, CV_32FC1);
    cv::transpose(cv_mat_src, cv_mat_dst);
    float* transposed_data = (float*)cv_mat_dst.data;

    for (int b = 0; b < batch; ++b) {
      float* data_batch = transposed_data + b * num_predictions * channels;
      std::vector<Detection> candidates;
      for (int i = 0; i < num_predictions; ++i) {
        float* row = data_batch + i * channels;
        float cx = row[0];
        float cy = row[1];
        float w = row[2];
        float h = row[3];
        float conf = row[4];
        if (conf < score_threshold) {
          continue;
        }
        float x1 = cx - w * 0.5f;
        float y1 = cy - h * 0.5f;
        float x2 = cx + w * 0.5f;
        float y2 = cy + h * 0.5f;
        x1 = x1 > 0 ? x1 : 0;
        y1 = y1 > 0 ? y1 : 0;
        x2 = x2 < param->model_w_ ? x2 : param->model_w_;
        y2 = y2 < param->model_h_ ? y2 : param->model_h_;

        Detection candidate;
        candidate.index_ = b;
        candidate.label_id_ = 0;
        candidate.score_ = conf;
        candidate.bbox_[0] = x1;
        candidate.bbox_[1] = y1;
        candidate.bbox_[2] = x2;
        candidate.bbox_[3] = y2;

        // (cx,cy,w,h,conf, kp0_x,kp0_y,kp0_conf, kp1_x,...)
        float model_w_f = static_cast<float>(param->model_w_);
        float model_h_f = static_cast<float>(param->model_h_);
        for (int k = 0; k < num_keypoints; ++k) {
          KeypointKeyPoint kp;
          kp.x_ = row[5 + k * 3] / model_w_f;
          kp.y_ = row[5 + k * 3 + 1] / model_h_f;
          kp.confidence_ = row[5 + k * 3 + 2];
          candidate.keypoints_.push_back(kp);
        }
        candidates.push_back(std::move(candidate));
      }

      std::vector<int> keep_idxs;
      computeKeypointNMS(candidates, keep_idxs, param->nms_threshold_);
      for (auto i = 0; i < (int)keep_idxs.size(); ++i) {
        auto n = keep_idxs[i];
        if (n < 0) {
          continue;
        }
        candidates[n].bbox_[0] /= param->model_w_;
        candidates[n].bbox_[1] /= param->model_h_;
        candidates[n].bbox_[2] /= param->model_w_;
        candidates[n].bbox_[3] /= param->model_h_;

        // BBox output
        detect::BBox bbox_item;
        bbox_item.index_ = candidates[n].index_;
        bbox_item.label_id_ = candidates[n].label_id_;
        bbox_item.score_ = candidates[n].score_;
        bbox_item.bbox_ = candidates[n].bbox_;
        bbox_result->bboxs_.push_back(std::move(bbox_item));

        // Keypoint output
        KpSkeleton skel;
        skel.index_ = candidates[n].index_;
        skel.label_id_ = candidates[n].label_id_;
        skel.score_ = candidates[n].score_;
        skel.keypoints_ = std::move(candidates[n].keypoints_);
        kp_result->skeletons_.push_back(std::move(skel));
      }
    }
  }
  outputs_[0]->set(bbox_result, false);
  outputs_[1]->set(kp_result, false);
  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::keypoint::KeypointPostProcess", KeypointPostProcess);
REGISTER_NODE("nndeploy::keypoint::KeypointGraph", KeypointGraph);

}  // namespace keypoint
}  // namespace nndeploy
