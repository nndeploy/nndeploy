#include "nndeploy/detect/yolo/yolo_nas.h"

#include <algorithm>
#include <cmath>

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
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/detect/result.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/infer/infer.h"
#include "nndeploy/inference/inference.h"
#include "nndeploy/preprocess/cvt_resize_norm_trans.h"

namespace nndeploy {
namespace detect {

static void nasComputeNMS(std::vector<BBox>& bboxes,
                          std::vector<int>& keep, float nms_threshold) {
  keep.clear();
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

base::Status YoloNasPostParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.AddMember("score_threshold_", score_threshold_, allocator);
  json.AddMember("nms_threshold_", nms_threshold_, allocator);
  json.AddMember("num_classes_", num_classes_, allocator);
  json.AddMember("model_h_", model_h_, allocator);
  json.AddMember("model_w_", model_w_, allocator);
  return base::kStatusCodeOk;
}

base::Status YoloNasPostParam::deserialize(rapidjson::Value& json) {
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
  if (json.HasMember("model_h_") && json["model_h_"].IsInt()) {
    model_h_ = json["model_h_"].GetInt();
  }
  if (json.HasMember("model_w_") && json["model_w_"].IsInt()) {
    model_w_ = json["model_w_"].GetInt();
  }
  return base::kStatusCodeOk;
}

base::Status YoloNasPostProcess::run() {
  YoloNasPostParam* param = (YoloNasPostParam*)param_.get();
  float score_threshold = param->score_threshold_;
  float nms_threshold = param->nms_threshold_;
  int model_h = param->model_h_;
  int model_w = param->model_w_;

  if (this->graph_) {
    auto infer_nodes = this->graph_->getNodesByKey("nndeploy::infer::Infer");
    if (!infer_nodes.empty()) {
      infer::Infer* infer = dynamic_cast<infer::Infer*>(infer_nodes[0]);
      if (infer) {
        auto inference_ptr = infer->getInference();
        if (inference_ptr) {
          auto input_tensors = inference_ptr->getAllInputTensorVector();
          if (!input_tensors.empty() && input_tensors[0] != nullptr) {
            auto shape = input_tensors[0]->getShape();
            if (shape.size() >= 4) {
              int h = static_cast<int>(shape[2]);
              int w = static_cast<int>(shape[3]);
              if (h > 0 && w > 0) {
                model_h = h;
                model_w = w;
                NNDEPLOY_LOGD(
                    "YoloNAS: auto-detected model input %dx%d"
                    " from inference input tensor\n",
                    model_h, model_w);
              }
            } else if (shape.size() == 3) {
              int h = static_cast<int>(shape[1]);
              int w = static_cast<int>(shape[2]);
              if (h > 0 && w > 0) {
                model_h = h;
                model_w = w;
                NNDEPLOY_LOGD(
                    "YoloNAS: auto-detected model input %dx%d"
                    " from inference input tensor (3D)\n",
                    model_h, model_w);
              }
            }
          }
        }
      }
    }
  }

  NNDEPLOY_LOGD("YoloNAS: inputs_.size=%zu\n", inputs_.size());
  for (size_t ei = 0; ei < inputs_.size(); ++ei) {
    device::Tensor* t = inputs_[ei]->getTensor(this);
    if (t) {
      base::IntVector shape = t->getShape();
      int total = 1;
      std::string shape_str;
      for (size_t si = 0; si < shape.size(); ++si) {
        total *= shape[si];
        shape_str +=
            std::to_string(shape[si]) + (si + 1 < shape.size() ? "," : "");
      }
      float* d = (float*)t->getData();
      float mn = d[0], mx = d[0];
      for (int j = 1; j < total; ++j) {
        if (d[j] < mn) mn = d[j];
        if (d[j] > mx) mx = d[j];
      }
      NNDEPLOY_LOGD(
          "YoloNAS:  inputs_[%zu] shape=[%s] total=%d range=[%.4f,%.4f] "
          "first5=[%.4f %.4f %.4f %.4f %.4f]\n",
          ei, shape_str.c_str(), total, mn, mx, d[0], d[std::min(1, total - 1)],
          d[std::min(2, total - 1)], d[std::min(3, total - 1)],
          d[std::min(4, total - 1)]);
    } else {
      NNDEPLOY_LOGD("YoloNAS:  inputs_[%zu] is NULL\n", ei);
    }
  }

  // Infer sorts outputs alphabetically by ONNX name:
  //   inputs_[0] → "904" → scores [1,8400,80] (already sigmoid)
  //   inputs_[1] → "913" → bbox [1,8400,4] (logits, need sigmoid)
  device::Tensor* score_tensor = inputs_[0]->getTensor(this);
  float* score_data = (float*)score_tensor->getData();
  int batch = score_tensor->getShapeIndex(0);
  int num_detections = score_tensor->getShapeIndex(1);
  int num_classes = score_tensor->getShapeIndex(2);

  device::Tensor* bbox_tensor = inputs_[1]->getTensor(this);
  float* bbox_data = (float*)bbox_tensor->getData();
  int bbox_batch = bbox_tensor->getShapeIndex(0);
  int bbox_detections = bbox_tensor->getShapeIndex(1);

  if (batch != bbox_batch || num_detections != bbox_detections) {
    NNDEPLOY_LOGE(
        "YoloNasPostProcess: shape mismatch batch=%d det=%d vs bbox_batch=%d "
        "bbox_det=%d\n",
        batch, num_detections, bbox_batch, bbox_detections);
    return base::kStatusCodeErrorInvalidParam;
  }

  // Print score distribution (sample first 100 detections)
  {
    int n = std::min(num_detections, 100);
    int high_score = 0;
    for (int i = 0; i < n; ++i) {
      float* sc = score_data + i * num_classes;
      float best = sc[0];
      for (int c = 1; c < num_classes; ++c)
        if (sc[c] > best) best = sc[c];
      if (best >= score_threshold) high_score++;
      if (i == 0) NNDEPLOY_LOGD("YoloNAS: best_score[0]=%.4f\n", best);
    }
    NNDEPLOY_LOGD(
        "YoloNAS: out of first %d detections, %d have best_score>=%.2f\n", n,
        high_score, score_threshold);
  }

  // Debug: raw bbox ranges (disabled by default, enable LOGD to see)
  {
    float s_min = 1.0f, s_max = 0.0f, r_min = 1e10f, r_max = -1e10f;
    for (int i = 0; i < std::min(num_detections, 1000); ++i) {
      float* bb = bbox_data + i * 4;
      for (int k = 0; k < 4; ++k) {
        float s = 1.0f / (1.0f + std::exp(-bb[k]));
        if (s < s_min) s_min = s;
        if (s > s_max) s_max = s;
        if (bb[k] < r_min) r_min = bb[k];
        if (bb[k] > r_max) r_max = bb[k];
      }
    }
    NNDEPLOY_LOGD(
        "YoloNAS: bbox raw range=[%.4f, %.4f] sigmoid range=[%.4f, %.4f] "
        "(sample %d)\n",
        r_min, r_max, s_min, s_max, std::min(num_detections, 1000));
  }

  BBoxResult* results = new BBoxResult();

  for (int b = 0; b < batch; ++b) {
    float* batch_bbox = bbox_data + b * num_detections * 4;
    float* batch_score = score_data + b * num_detections * num_classes;

    std::vector<BBox> candidates;

    for (int i = 0; i < num_detections; ++i) {
      float* bb = batch_bbox + i * 4;
      // Raw bbox values are in pixel coordinates (range ~[-10, 650] for 640
      // input) Normalize to [0,1] by dividing by model dimensions
      float x1 = std::max(0.0f, std::min(1.0f, bb[0] / model_w));
      float y1 = std::max(0.0f, std::min(1.0f, bb[1] / model_h));
      float x2 = std::max(0.0f, std::min(1.0f, bb[2] / model_w));
      float y2 = std::max(0.0f, std::min(1.0f, bb[3] / model_h));
      float* sc = batch_score + i * num_classes;
      int best_class = 0;
      float best_score = sc[0];
      for (int c = 1; c < num_classes; ++c) {
        if (sc[c] > best_score) {
          best_score = sc[c];
          best_class = c;
        }
      }

      if (best_score < score_threshold) {
        continue;
      }

      BBox bbox;
      bbox.index_ = b;
      bbox.label_id_ = best_class;
      bbox.score_ = best_score;
      bbox.bbox_[0] = std::max(0.0f, std::min(1.0f, x1));
      bbox.bbox_[1] = std::max(0.0f, std::min(1.0f, y1));
      bbox.bbox_[2] = std::max(0.0f, std::min(1.0f, x2));
      bbox.bbox_[3] = std::max(0.0f, std::min(1.0f, y2));
      candidates.emplace_back(bbox);
    }

    NNDEPLOY_LOGD("YoloNAS: batch %d candidates before NMS: %zu\n", b,
                  candidates.size());

    std::vector<int> keep;
    nasComputeNMS(candidates, keep, nms_threshold);
    for (int idx : keep) {
      results->bboxs_.emplace_back(candidates[idx]);
    }
  }

  NNDEPLOY_LOGD("YoloNAS: final detection count: %zu\n",
                results->bboxs_.size());

  outputs_[0]->set(results, false);
  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::detect::YoloNasPostProcess", YoloNasPostProcess);
REGISTER_NODE("nndeploy::detect::YoloNasGraph", YoloNasGraph);

}  // namespace detect
}  // namespace nndeploy
