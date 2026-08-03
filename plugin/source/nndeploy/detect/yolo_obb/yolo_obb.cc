#include "nndeploy/detect/yolo_obb/yolo_obb.h"

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
#include "nndeploy/dag/node.h"
#include "nndeploy/detect/result.h"
#include "nndeploy/detect/util.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/infer/infer.h"
#include "nndeploy/preprocess/cvt_resize_norm_trans.h"

namespace nndeploy {
namespace detect {

base::Status ObbPostParam::serialize(
    rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
  json.AddMember("score_threshold_", score_threshold_, allocator);
  json.AddMember("nms_threshold_", nms_threshold_, allocator);
  json.AddMember("num_classes_", num_classes_, allocator);
  json.AddMember("model_h_", model_h_, allocator);
  json.AddMember("model_w_", model_w_, allocator);
  json.AddMember("version_", version_, allocator);
  return base::kStatusCodeOk;
}

base::Status ObbPostParam::deserialize(rapidjson::Value &json) {
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

  if (json.HasMember("version_") && json["version_"].IsInt()) {
    version_ = json["version_"].GetInt();
  }

  return base::kStatusCodeOk;
}

static void decodeObbV8(const float *row, int num_classes,
                         float model_w, float model_h,
                         float score_threshold,
                         std::vector<RotatedBox> &candidates,
                         std::vector<BBox> &nms_boxes, int b) {
  // yolo11n-obb/v8-obb ONNX output per-column format (after transpose):
  // [cx, cy, w, h, cls_0..cls_{N-1}, angle]
  //  0   1   2  3  4..4+N-1        4+N
  float cx = row[0];
  float cy = row[1];
  float w = row[2];
  float h = row[3];

  float max_score = 0.0f;
  int max_id = -1;
  for (int c = 0; c < num_classes; ++c) {
    float score = row[4 + c];
    if (score > max_score) {
      max_score = score;
      max_id = c;
    }
  }

  if (max_score < score_threshold || max_id < 0) return;

  float angle = row[4 + num_classes];

  RotatedBox box;
  box.index_ = b;
  box.label_id_ = max_id;
  box.score_ = max_score;
  box.cx_ = cx / model_w;
  box.cy_ = cy / model_h;
  box.w_ = w / model_w;
  box.h_ = h / model_h;
  box.angle_ = angle;
  candidates.push_back(box);

  float x1 = (cx - w * 0.5f) / model_w;
  float y1 = (cy - h * 0.5f) / model_h;
  float x2 = (cx + w * 0.5f) / model_w;
  float y2 = (cy + h * 0.5f) / model_h;

  BBox nms_box;
  nms_box.index_ = b;
  nms_box.label_id_ = max_id;
  nms_box.score_ = max_score;
  nms_box.bbox_[0] = x1;
  nms_box.bbox_[1] = y1;
  nms_box.bbox_[2] = x2;
  nms_box.bbox_[3] = y2;
  nms_boxes.push_back(nms_box);
}

static void decodeObbV26NmsFree(const float *row, int num_classes,
                                 float model_w, float model_h,
                                 float score_threshold,
                                 std::vector<RotatedBox> &candidates,
                                 std::vector<BBox> &nms_boxes,
                                 int b) {
  // yolo26n-obb NMS-free output per-box format:
  // [cx, cy, w, h, score, class_id, angle]
  //  0   1   2  3     4        5        6
  float cx = row[0];
  float cy = row[1];
  float w = row[2];
  float h = row[3];
  float score = row[4];
  int class_id = static_cast<int>(row[5]);
  float angle = row[6];

  if (score < score_threshold || class_id < 0 || class_id >= num_classes) return;

  RotatedBox box;
  box.index_ = b;
  box.label_id_ = class_id;
  box.score_ = score;
  box.cx_ = cx / model_w;
  box.cy_ = cy / model_h;
  box.w_ = w / model_w;
  box.h_ = h / model_h;
  box.angle_ = angle;
  candidates.push_back(box);

  float x1 = (cx - w * 0.5f) / model_w;
  float y1 = (cy - h * 0.5f) / model_h;
  float x2 = (cx + w * 0.5f) / model_w;
  float y2 = (cy + h * 0.5f) / model_h;

  BBox nms_box;
  nms_box.index_ = b;
  nms_box.label_id_ = class_id;
  nms_box.score_ = score;
  nms_box.bbox_[0] = x1;
  nms_box.bbox_[1] = y1;
  nms_box.bbox_[2] = x2;
  nms_box.bbox_[3] = y2;
  nms_boxes.push_back(nms_box);
}

base::Status ObbPostProcess::run() {
  ObbPostParam *param = (ObbPostParam *)param_.get();
  float score_threshold = param->score_threshold_;
  int num_classes = param->num_classes_;

  device::Tensor *tensor = inputs_[0]->getTensor(this);
  float *data = (float *)tensor->getData();
  int batch = tensor->getShapeIndex(0);
  int dim1 = tensor->getShapeIndex(1);
  int dim2 = tensor->getShapeIndex(2);

  ObbResult *results = new ObbResult();

  for (int b = 0; b < batch; ++b) {
    std::vector<RotatedBox> candidates;
    std::vector<BBox> nms_boxes;

    if (param->version_ == 8 || param->version_ == 11) {
      // v8/v11: [batch, channels, num_predictions] format
      //   e.g. yolo11n-obb: [1, 20, 21504]
      //   layout: [cx,cy,w,h, cls_0..cls_{N-1}, angle] × num_predictions
      int channels = dim1;
      int num_predictions = dim2;

      // Transpose to [num_predictions, channels] for row-by-row decoding
      cv::Mat cv_mat_src(channels, num_predictions, CV_32FC1,
                         data + b * channels * num_predictions);
      cv::Mat cv_mat_dst(num_predictions, channels, CV_32FC1);
      cv::transpose(cv_mat_src, cv_mat_dst);
      float *transposed_data = (float *)cv_mat_dst.data;

      for (int i = 0; i < num_predictions; ++i) {
        float *row = transposed_data + i * channels;
        decodeObbV8(row, num_classes, param->model_w_, param->model_h_,
                    score_threshold, candidates, nms_boxes, b);
      }

    } else if (param->version_ == 26) {
      // yolo26 NMS-free: [batch, num_candidates, fields] format
      //   e.g. yolo26n-obb: [1, 300, 7]
      //   fields: [cx, cy, w, h, score, class_id, angle]
      int num_candidates = dim1;
      int fields = dim2;
      float *batch_data = data + b * num_candidates * fields;

      for (int i = 0; i < num_candidates; ++i) {
        float *row = batch_data + i * fields;
        decodeObbV26NmsFree(row, num_classes, param->model_w_, param->model_h_,
                            score_threshold, candidates, nms_boxes, b);
      }

    } else {
      NNDEPLOY_LOGE("Unsupported OBB version: %d", param->version_);
      return base::kStatusCodeErrorInvalidValue;
    }

    if (candidates.empty()) {
      continue;
    }

    if (param->version_ == 26) {
      // yolo26 is NMS-free — the model itself already applies NMS internally.
      // Pushing all candidates that passed the score threshold directly.
      for (auto &candidate : candidates) {
        results->boxes_.push_back(candidate);
      }
    } else {
      // v8/v11: raw predictions — apply axis-aligned NMS to filter duplicates
      BBoxResult nms_result;
      nms_result.bboxs_ = nms_boxes;
      std::vector<int> keep_idxs(nms_boxes.size());
      computeNMS(nms_result, keep_idxs, param->nms_threshold_);

      for (auto i = 0; i < (int)keep_idxs.size(); ++i) {
        auto n = keep_idxs[i];
        if (n < 0) {
          continue;
        }
        results->boxes_.push_back(candidates[n]);
      }
    }
  }

  outputs_[0]->set(results, false);
  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::detect::ObbPostProcess", ObbPostProcess);
REGISTER_NODE("nndeploy::detect::ObbGraph", ObbGraph);

}  // namespace detect
}  // namespace nndeploy
