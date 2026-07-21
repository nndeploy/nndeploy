#include "nndeploy/segment/rf_detr_seg/rf_detr_seg.h"

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
#include "nndeploy/detect/util.h"
#include "nndeploy/segment/result.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/infer/infer.h"
#include "nndeploy/preprocess/cvt_resize_norm_trans.h"

namespace nndeploy {
namespace segment {

// ---------------------------------------------------------------------------
// RfDetrSegPostParam serialization
// ---------------------------------------------------------------------------

base::Status RfDetrSegPostParam::serialize(
    rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
  json.AddMember("score_threshold_", score_threshold_, allocator);
  json.AddMember("num_classes_", num_classes_, allocator);
  json.AddMember("model_h_", model_h_, allocator);
  json.AddMember("model_w_", model_w_, allocator);
  json.AddMember("nms_threshold_", nms_threshold_, allocator);
  return base::kStatusCodeOk;
}

base::Status RfDetrSegPostParam::deserialize(rapidjson::Value &json) {
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

// ---------------------------------------------------------------------------
// NMS helper (operates on lightweight BBox — no mask pointer ownership)
// ---------------------------------------------------------------------------

static void rfdetrSegComputeNMS(std::vector<detect::BBox> &bboxes,
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

// ---------------------------------------------------------------------------
// RfDetrSegPostProcess::run() — dual-channel output (BBoxResult + SegMaskResult)
// ---------------------------------------------------------------------------

base::Status RfDetrSegPostProcess::run() {
  RfDetrSegPostParam *param = (RfDetrSegPostParam *)param_.get();
  float score_threshold = param->score_threshold_;
  float nms_threshold = param->nms_threshold_;
  int num_classes = param->num_classes_;
  int model_h = param->model_h_;
  int model_w = param->model_w_;

  // RF-DETR-Seg model outputs three tensors (sorted alphabetically by ONNX name):
  //   inputs_[0] = "dets"   — [batch, num_queries, 4]  boxes [cx,cy,w,h] normalized
  //   inputs_[1] = "labels" — [batch, num_queries, 91] raw class logits
  //   inputs_[2] = "masks"  — [batch, num_queries, mask_h, mask_w] per-query mask logits
  device::Tensor *dets_tensor = inputs_[0]->getTensor(this);
  device::Tensor *labels_tensor = inputs_[1]->getTensor(this);
  device::Tensor *masks_tensor = inputs_[2]->getTensor(this);

  if (!dets_tensor || !labels_tensor || !masks_tensor) {
    NNDEPLOY_LOGE("RfDetrSegPostProcess: input tensor is NULL\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  if (inputs_.size() < 3) {
    NNDEPLOY_LOGE("RfDetrSegPostProcess: expected 3 inputs, got %zu\n",
                  inputs_.size());
    return base::kStatusCodeErrorInvalidParam;
  }

  int batch = dets_tensor->getShapeIndex(0);
  int num_queries = dets_tensor->getShapeIndex(1);
  int label_num_queries = labels_tensor->getShapeIndex(1);
  int label_dim = labels_tensor->getShapeIndex(2);  // 91 for COCO
  int mask_num_queries = masks_tensor->getShapeIndex(1);
  int mask_h = masks_tensor->getShapeIndex(2);
  int mask_w = masks_tensor->getShapeIndex(3);

  if (num_queries != label_num_queries || num_queries != mask_num_queries) {
    NNDEPLOY_LOGE(
        "RfDetrSegPostProcess: query count mismatch dets=%d labels=%d masks=%d\n",
        num_queries, label_num_queries, mask_num_queries);
    return base::kStatusCodeErrorInvalidParam;
  }

  if (batch <= 0 || num_queries <= 0 || label_dim <= 0 || mask_h <= 0 ||
      mask_w <= 0) {
    NNDEPLOY_LOGE("RfDetrSegPostProcess: invalid shapes\n");
    return base::kStatusCodeErrorInvalidParam;
  }

  float *dets_data = (float *)dets_tensor->getData();
  float *labels_data = (float *)labels_tensor->getData();
  float *masks_data = (float *)masks_tensor->getData();

  // Output channels
  detect::BBoxResult *bbox_results = new detect::BBoxResult();
  SegMaskResult *mask_results = new SegMaskResult();

  // Get device for mask tensor allocation
  device::Device *device = dets_tensor->getDevice();

  for (int b = 0; b < batch; ++b) {
    float *batch_dets = dets_data + b * num_queries * 4;
    float *batch_labels = labels_data + b * num_queries * label_dim;
    float *batch_masks = masks_data + b * num_queries * mask_h * mask_w;

    // Lightweight BBox candidates for NMS (no mask pointer)
    std::vector<detect::BBox> candidates;
    candidates.reserve(num_queries);

    // Parallel mask storage (indexed same as candidates)
    std::vector<device::Tensor *> mask_tensors;
    mask_tensors.reserve(num_queries);

    for (int q = 0; q < num_queries; ++q) {
      float *logits = batch_labels + q * label_dim;
      float *box = batch_dets + q * 4;
      float *mask_logits = batch_masks + q * mask_h * mask_w;

      // ---- Softmax over label_dim classes ----
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

      // ---- Decode box [cx, cy, w, h] -> [x1, y1, x2, y2] (normalized [0,1]) ----
      float cx = box[0];
      float cy = box[1];
      float w = box[2];
      float h = box[3];

      float x1 = std::max(0.0f, std::min(1.0f, cx - w * 0.5f));
      float y1 = std::max(0.0f, std::min(1.0f, cy - h * 0.5f));
      float x2 = std::max(0.0f, std::min(1.0f, cx + w * 0.5f));
      float y2 = std::max(0.0f, std::min(1.0f, cy + h * 0.5f));

      // ---- Decode mask: sigmoid -> threshold -> upsample ----
      cv::Mat mask_float(mask_h, mask_w, CV_32FC1);
      for (int r = 0; r < mask_h; ++r) {
        float *src_row = mask_logits + r * mask_w;
        float *dst_row = mask_float.ptr<float>(r);
        for (int c = 0; c < mask_w; ++c) {
          dst_row[c] = 1.0f / (1.0f + std::exp(-src_row[c]));
        }
      }

      cv::Mat mask_binary;
      cv::threshold(mask_float, mask_binary, 0.5f, 1.0f, cv::THRESH_BINARY);

      cv::Mat mask_uint8;
      mask_binary.convertTo(mask_uint8, CV_8UC1, 255.0);

      cv::Mat mask_resized;
      cv::resize(mask_uint8, mask_resized, cv::Size(model_w, model_h), 0, 0,
                 cv::INTER_NEAREST);

      // Create device::Tensor for mask storage
      device::TensorDesc mask_desc;
      mask_desc.data_type_ = base::dataTypeOf<uint8_t>();
      mask_desc.shape_ = {1, model_h, model_w};
      device::Tensor *mask_tensor = new device::Tensor(device, mask_desc);
      uint8_t *mask_data = (uint8_t *)mask_tensor->getData();
      for (int r = 0; r < model_h; ++r) {
        uint8_t *row_data = mask_resized.ptr<uint8_t>(r);
        memcpy(mask_data + r * model_w, row_data, model_w * sizeof(uint8_t));
      }

      // ---- Build lightweight BBox (no mask pointer) ----
      detect::BBox bbox;
      bbox.index_ = b;
      bbox.label_id_ = label_id;
      bbox.score_ = best_prob;
      bbox.bbox_[0] = x1;
      bbox.bbox_[1] = y1;
      bbox.bbox_[2] = x2;
      bbox.bbox_[3] = y2;
      candidates.emplace_back(bbox);
      mask_tensors.push_back(mask_tensor);
    }

    // ---- Apply NMS if threshold > 0 ----
    std::vector<int> keep_indices;
    if (nms_threshold > 0.0f && !candidates.empty()) {
      rfdetrSegComputeNMS(candidates, keep_indices, nms_threshold);
    } else {
      keep_indices.resize(candidates.size());
      for (size_t i = 0; i < candidates.size(); ++i) keep_indices[i] = (int)i;
    }

    // Populate output channels from kept indices
    bbox_results->bboxs_.reserve(bbox_results->bboxs_.size() + keep_indices.size());
    mask_results->masks_.reserve(mask_results->masks_.size() + keep_indices.size());

    for (int idx : keep_indices) {
      // BBoxResult — lightweight, no mask pointer
      bbox_results->bboxs_.push_back(candidates[idx]);

      // SegMaskResult — mask tensor owned here
      SegMaskItem item;
      item.index_ = candidates[idx].index_;
      item.label_id_ = candidates[idx].label_id_;
      item.score_ = candidates[idx].score_;
      item.mask_ = mask_tensors[idx];
      mask_tensors[idx] = nullptr;  // ownership transferred to SegMaskItem
      mask_results->masks_.push_back(item);
    }

    // Free mask tensors for non-kept candidates
    for (size_t i = 0; i < mask_tensors.size(); ++i) {
      if (mask_tensors[i] != nullptr) {
        delete mask_tensors[i];
        mask_tensors[i] = nullptr;
      }
    }
  }

  outputs_[0]->set(bbox_results, false);
  outputs_[1]->set(mask_results, false);
  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::segment::RfDetrSegPostProcess", RfDetrSegPostProcess);
REGISTER_NODE("nndeploy::segment::RfDetrSegGraph", RfDetrSegGraph);

}  // namespace segment
}  // namespace nndeploy
