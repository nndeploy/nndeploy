#include "nndeploy/detect/rf_detr/rf_detr.h"

#include <algorithm>
#include <cmath>
#include <limits>

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
#include "nndeploy/detect/result.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/infer/infer.h"
#include "nndeploy/preprocess/cvt_resize_norm_trans.h"

namespace nndeploy {
namespace detect {

base::Status RfDetrPostParam::serialize(
    rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
  json.AddMember("score_threshold_", score_threshold_, allocator);
  json.AddMember("num_classes_", num_classes_, allocator);
  json.AddMember("model_h_", model_h_, allocator);
  json.AddMember("model_w_", model_w_, allocator);
  json.AddMember("nms_threshold_", nms_threshold_, allocator);
  return base::kStatusCodeOk;
}

base::Status RfDetrPostParam::deserialize(rapidjson::Value &json) {
  if (json.HasMember("score_threshold_") &&
      json["score_threshold_"].IsFloat()) {
    score_threshold_ = json["score_threshold_"].GetFloat();
  }
  if (json.HasMember("num_classes_") && json["num_classes_"].IsInt()) {
    num_classes_ = json["num_classes_"].GetInt();
  }
  if (json.HasMember("model_h_") && json["model_h_"].IsInt()) {
    model_h_ = json["model_h_"].GetInt();
  }
  if (json.HasMember("model_w_") && json["model_w_"].IsInt()) {
    model_w_ = json["model_w_"].GetInt();
  }
  if (json.HasMember("nms_threshold_") &&
      json["nms_threshold_"].IsFloat()) {
    nms_threshold_ = json["nms_threshold_"].GetFloat();
  }
  return base::kStatusCodeOk;
}

static void rfdetrComputeNMS(std::vector<BBox> &bboxes,
                             std::vector<int> &keep, float nms_threshold) {
  keep.clear();
  if (bboxes.empty()) return;
  std::vector<float> areas(bboxes.size());
  for (size_t i = 0; i < bboxes.size(); ++i) {
    float w = std::max(0.0f, bboxes[i].bbox_[2] - bboxes[i].bbox_[0]);
    float h = std::max(0.0f, bboxes[i].bbox_[3] - bboxes[i].bbox_[1]);
    areas[i] = w * h;
  }
  std::vector<int> idxs(bboxes.size());
  for (size_t i = 0; i < bboxes.size(); ++i) idxs[i] = (int)i;
  std::sort(idxs.begin(), idxs.end(), [&bboxes](int a, int b) {
    return bboxes[a].score_ > bboxes[b].score_;
  });
  std::vector<bool> removed(bboxes.size(), false);
  for (size_t i = 0; i < idxs.size(); ++i) {
    if (removed[idxs[i]]) continue;
    keep.push_back(idxs[i]);
    for (size_t j = i + 1; j < idxs.size(); ++j) {
      if (removed[idxs[j]]) continue;
      float ix1 = std::max(bboxes[idxs[i]].bbox_[0], bboxes[idxs[j]].bbox_[0]);
      float iy1 = std::max(bboxes[idxs[i]].bbox_[1], bboxes[idxs[j]].bbox_[1]);
      float ix2 = std::min(bboxes[idxs[i]].bbox_[2], bboxes[idxs[j]].bbox_[2]);
      float iy2 = std::min(bboxes[idxs[i]].bbox_[3], bboxes[idxs[j]].bbox_[3]);
      float iw = std::max(0.0f, ix2 - ix1);
      float ih = std::max(0.0f, iy2 - iy1);
      float inter = iw * ih;
      float ovr = inter / (areas[idxs[i]] + areas[idxs[j]] - inter);
      if (ovr > nms_threshold) {
        removed[idxs[j]] = true;
      }
    }
  }
}

base::Status RfDetrPostProcess::run() {
  RfDetrPostParam *param = (RfDetrPostParam *)param_.get();
  float score_threshold = param->score_threshold_;
  float nms_threshold = param->nms_threshold_;
  int num_classes = param->num_classes_;

  // RF-DETR model outputs two tensors (sorted alphabetically by ONNX name):
  //   inputs_[0] = "dets"   — [batch, num_queries, 4]  boxes in [cx,cy,w,h] normalized
  //   inputs_[1] = "labels" — [batch, num_queries, 91] raw class logits (COCO 0-90)
  device::Tensor *dets_tensor = inputs_[0]->getTensor(this);
  device::Tensor *labels_tensor = inputs_[1]->getTensor(this);

  if (!dets_tensor || !labels_tensor) {
    NNDEPLOY_LOGE("RfDetrPostProcess: inputs_[0] or inputs_[1] is NULL\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  int batch = dets_tensor->getShapeIndex(0);
  int num_queries = dets_tensor->getShapeIndex(1);
  int label_num_queries = labels_tensor->getShapeIndex(1);
  int label_dim = labels_tensor->getShapeIndex(2);  // 91 for COCO

  if (num_queries != label_num_queries) {
    NNDEPLOY_LOGE(
        "RfDetrPostProcess: shape mismatch dets_queries=%d labels_queries=%d\n",
        num_queries, label_num_queries);
    return base::kStatusCodeErrorInvalidParam;
  }

  if (batch <= 0 || num_queries <= 0 || label_dim <= 0) {
    NNDEPLOY_LOGE("RfDetrPostProcess: invalid shapes batch=%d queries=%d label_dim=%d\n",
                  batch, num_queries, label_dim);
    return base::kStatusCodeErrorInvalidParam;
  }

  float *dets_data = (float *)dets_tensor->getData();
  float *labels_data = (float *)labels_tensor->getData();

  BBoxResult *results = new BBoxResult();

  for (int b = 0; b < batch; ++b) {
    float *batch_dets = dets_data + b * num_queries * 4;
    float *batch_labels = labels_data + b * num_queries * label_dim;

    std::vector<BBox> candidates;

    for (int q = 0; q < num_queries; ++q) {
      float *logits = batch_labels + q * label_dim;
      float *box = batch_dets + q * 4;

      // Softmax over label_dim classes
      float max_logit = -std::numeric_limits<float>::max();
      for (int c = 0; c < label_dim; ++c) {
        if (logits[c] > max_logit) max_logit = logits[c];
      }
      float sum_exp = 0.0f;
      for (int c = 0; c < label_dim; ++c) {
        sum_exp += std::exp(logits[c] - max_logit);
      }
      if (sum_exp <= 0.0f) {
        continue;
      }
      float inv_sum = 1.0f / sum_exp;

      // Find best class (excluding background at index 0)
      float best_prob = 0.0f;
      int best_idx = -1;
      for (int c = 1; c < label_dim; ++c) {
        float prob = std::exp(logits[c] - max_logit) * inv_sum;
        if (prob > best_prob) {
          best_prob = prob;
          best_idx = c;
        }
      }

      if (best_idx < 0 || best_prob < score_threshold) {
        continue;
      }

      // Convert 1-indexed COCO class to 0-indexed label_id
      int label_id = best_idx - 1;
      if (num_classes > 0 && label_id >= num_classes) {
        continue;
      }

      // Box is in [cx, cy, w, h] format, normalized to [0, 1]
      float cx = box[0];
      float cy = box[1];
      float w = box[2];
      float h = box[3];

      // Clamp box to valid range (model may output slightly outside [0,1])
      float x1 = std::max(0.0f, std::min(1.0f, cx - w * 0.5f));
      float y1 = std::max(0.0f, std::min(1.0f, cy - h * 0.5f));
      float x2 = std::max(0.0f, std::min(1.0f, cx + w * 0.5f));
      float y2 = std::max(0.0f, std::min(1.0f, cy + h * 0.5f));

      BBox bbox;
      bbox.index_ = b;
      bbox.label_id_ = label_id;
      bbox.score_ = best_prob;
      bbox.bbox_[0] = x1;
      bbox.bbox_[1] = y1;
      bbox.bbox_[2] = x2;
      bbox.bbox_[3] = y2;
      candidates.emplace_back(bbox);
    }

    // Apply NMS if the threshold is set (> 0)
    if (nms_threshold > 0.0f && !candidates.empty()) {
      std::vector<int> keep;
      rfdetrComputeNMS(candidates, keep, nms_threshold);
      for (int idx : keep) {
        results->bboxs_.emplace_back(candidates[idx]);
      }
    } else {
      for (auto &bbox : candidates) {
        results->bboxs_.emplace_back(bbox);
      }
    }
  }

  outputs_[0]->set(results, false);
  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::detect::RfDetrPostProcess", RfDetrPostProcess);
REGISTER_NODE("nndeploy::detect::RfDetrGraph", RfDetrGraph);

}  // namespace detect
}  // namespace nndeploy
