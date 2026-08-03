#include "nndeploy/detect/detr/detr.h"

#include <algorithm>

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

base::Status DetrPostParam::serialize(
    rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
  json.AddMember("score_threshold_", score_threshold_, allocator);
  json.AddMember("num_classes_", num_classes_, allocator);
  json.AddMember("model_h_", model_h_, allocator);
  json.AddMember("model_w_", model_w_, allocator);
  return base::kStatusCodeOk;
}

base::Status DetrPostParam::deserialize(rapidjson::Value &json) {
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
  return base::kStatusCodeOk;
}

base::Status DetrPostProcess::run() {
  DetrPostParam *param = (DetrPostParam *)param_.get();
  float score_threshold = param->score_threshold_;

  device::Tensor *tensor = inputs_[0]->getTensor(this);
  float *data = (float *)tensor->getData();
  int batch = tensor->getShapeIndex(0);
  int num_detections = tensor->getShapeIndex(1);
  int channels = tensor->getShapeIndex(2);  // 6: cx, cy, w, h, score, class_id

  DetectResult *results = new DetectResult();

  for (int b = 0; b < batch; ++b) {
    float *batch_data = data + b * num_detections * channels;

    for (int i = 0; i < num_detections; ++i) {
      float *row = batch_data + i * channels;
      float cx = row[0];
      float cy = row[1];
      float w = row[2];
      float h = row[3];
      float score = row[4];
      int class_id = (int)row[5];

      if (score < score_threshold) {
        continue;
      }

      // Convert cx,cy,w,h (pixel space) to x1,y1,x2,y2 (normalized)
      float x1 = (cx - w * 0.5f) / param->model_w_;
      float y1 = (cy - h * 0.5f) / param->model_h_;
      float x2 = (cx + w * 0.5f) / param->model_w_;
      float y2 = (cy + h * 0.5f) / param->model_h_;

      // Clamp to [0, 1]
      x1 = std::max(0.0f, std::min(1.0f, x1));
      y1 = std::max(0.0f, std::min(1.0f, y1));
      x2 = std::max(0.0f, std::min(1.0f, x2));
      y2 = std::max(0.0f, std::min(1.0f, y2));

      DetectBBoxResult bbox;
      bbox.index_ = b;
      bbox.label_id_ = class_id;
      bbox.score_ = score;
      bbox.bbox_[0] = x1;
      bbox.bbox_[1] = y1;
      bbox.bbox_[2] = x2;
      bbox.bbox_[3] = y2;
      results->bboxs_.emplace_back(bbox);
    }
  }

  outputs_[0]->set(results, false);
  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::detect::DetrPostProcess", DetrPostProcess);
REGISTER_NODE("nndeploy::detect::DetrGraph", DetrGraph);

}  // namespace detect
}  // namespace nndeploy
