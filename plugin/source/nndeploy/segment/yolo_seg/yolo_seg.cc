#include "nndeploy/segment/yolo_seg/yolo_seg.h"

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
namespace segment {

base::Status YoloSegPostParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.AddMember("score_threshold_", score_threshold_, allocator);
  json.AddMember("nms_threshold_", nms_threshold_, allocator);
  json.AddMember("num_classes_", num_classes_, allocator);
  json.AddMember("model_h_", model_h_, allocator);
  json.AddMember("model_w_", model_w_, allocator);
  json.AddMember("version_", version_, allocator);
  return base::kStatusCodeOk;
}

base::Status YoloSegPostParam::deserialize(rapidjson::Value& json) {
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

base::Status YoloSegPostProcess::run() {
  YoloSegPostParam* param = (YoloSegPostParam*)param_.get();
  float score_threshold = param->score_threshold_;
  int num_classes = param->num_classes_;

  // Input 0: detection tensor (batch, 4+num_classes+32, num_predictions)
  device::Tensor* det_tensor = inputs_[0]->getTensor(this);
  float* det_data = (float*)det_tensor->getData();
  int batch = det_tensor->getShapeIndex(0);
  int channels = det_tensor->getShapeIndex(1);
  int num_predictions = det_tensor->getShapeIndex(2);

  float* transposed_data = nullptr;
  cv::Mat cv_mat_dst;
  if (param->version_ == 26) {
    // YOLO v26+ output is (batch, num_predictions, channels) — already in
    // predictions-first format. Swap variable names so channels = elements per
    // row.
    std::swap(channels, num_predictions);
    transposed_data = det_data;
  } else {
    // YOLO v11/v8 output is (batch, channels, num_predictions) — transpose to
    // (batch, num_predictions, channels) so each row is one prediction.
    cv::Mat cv_mat_src(channels, num_predictions, CV_32FC1, det_data);
    cv_mat_dst.create(num_predictions, channels, CV_32FC1);
    cv::transpose(cv_mat_src, cv_mat_dst);
    transposed_data = (float*)cv_mat_dst.data;
  }

  // Now channels = elements per row (after swap/transpose),
  // num_predictions = number of rows
  // YOLO26 format: [bbox(4), score(1), class_id(1), mask_coeffs(32)] = 38
  // YOLOv8/v11 format: [bbox(4), class_scores(N), mask_coeffs(32)]
  int num_mask_coeffs;
  if (param->version_ == 26) {
    num_mask_coeffs = channels - 4 - 2;  // bbox(4) + score(1) + class_id(1)
  } else {
    num_mask_coeffs = channels - 4 - num_classes;
  }

  // Input 1: proto mask tensor (batch, 32, proto_h, proto_w)
  device::Tensor* proto_tensor = inputs_[1]->getTensor(this);
  float* proto_data = (float*)proto_tensor->getData();
  int proto_h = proto_tensor->getShapeIndex(2);
  int proto_w = proto_tensor->getShapeIndex(3);

  detect::BBoxResult* bbox_result = new detect::BBoxResult();
  SegMaskResult* mask_result = new SegMaskResult();

  for (int b = 0; b < batch; ++b) {
    float* batch_data = transposed_data + b * num_predictions * channels;
    float* batch_proto = proto_data + b * num_mask_coeffs * proto_h * proto_w;

    detect::BBoxResult candidates_result;
    std::vector<const float*> candidate_coeffs;

    for (int i = 0; i < num_predictions; ++i) {
      float* row = batch_data + i * channels;

      // Decode bbox
      float x1, y1, x2, y2;
      if (param->version_ == 26) {
        // YOLO26 format: [x1, y1, x2, y2, score, class_id, ...]
        // bbox is already in xyxy pixel coordinates
        x1 = row[0];
        y1 = row[1];
        x2 = row[2];
        y2 = row[3];
      } else {
        // YOLOv8/v11 format: [cx, cy, w, h, class_scores(N), ...]
        float cx = row[0];
        float cy = row[1];
        float w = row[2];
        float h = row[3];
        x1 = cx - w * 0.5f;
        y1 = cy - h * 0.5f;
        x2 = cx + w * 0.5f;
        y2 = cy + h * 0.5f;
      }
      x1 = x1 > 0 ? x1 : 0;
      y1 = y1 > 0 ? y1 : 0;
      x2 = x2 < param->model_w_ ? x2 : param->model_w_;
      y2 = y2 < param->model_h_ ? y2 : param->model_h_;

      float max_score;
      int max_id;
      const float* mask_coeffs;
      if (param->version_ == 26) {
        // YOLO26 format: [bbox(4), score(1), class_id(1), mask_coeffs(32)]
        max_score = row[4];
        max_id = (int)row[5];
        mask_coeffs = row + 6;
      } else {
        // YOLOv8/v11 format: [bbox(4), class_scores(N), mask_coeffs(32)]
        max_score = 0.0f;
        max_id = -1;
        for (int c = 0; c < num_classes; ++c) {
          float score = row[4 + c];
          if (score > max_score) {
            max_score = score;
            max_id = c;
          }
        }
        mask_coeffs = row + 4 + num_classes;
      }

      if (max_score < score_threshold || max_id < 0) {
        continue;
      }

      detect::BBox bbox;
      bbox.index_ = b;
      bbox.label_id_ = max_id;
      bbox.score_ = max_score;
      bbox.bbox_[0] = x1;
      bbox.bbox_[1] = y1;
      bbox.bbox_[2] = x2;
      bbox.bbox_[3] = y2;
      candidates_result.bboxs_.emplace_back(bbox);

      // Store mask coefficient pointer
      candidate_coeffs.push_back(mask_coeffs);
    }

    if (candidates_result.bboxs_.empty()) {
      continue;
    }

    // Apply NMS (YOLO v26+ models are NMS-free - skip NMS)
    std::vector<int> keep_idxs(candidates_result.bboxs_.size());
    if (param->version_ < 26) {
      detect::computeNMS(candidates_result, keep_idxs, param->nms_threshold_);
    } else {
      for (int k = 0; k < (int)keep_idxs.size(); ++k) {
        keep_idxs[k] = k;
      }
    }

    // Pre-reserve to prevent vector reallocation
    bbox_result->bboxs_.reserve(bbox_result->bboxs_.size() +
                                 candidates_result.bboxs_.size());
    mask_result->masks_.reserve(mask_result->masks_.size() +
                                 candidates_result.bboxs_.size());

    for (auto i = 0; i < (int)keep_idxs.size(); ++i) {
      auto n = keep_idxs[i];
      if (n < 0) {
        continue;
      }

      // Normalize bbox coordinates
      candidates_result.bboxs_[n].bbox_[0] /= param->model_w_;
      candidates_result.bboxs_[n].bbox_[1] /= param->model_h_;
      candidates_result.bboxs_[n].bbox_[2] /= param->model_w_;
      candidates_result.bboxs_[n].bbox_[3] /= param->model_h_;

      // Generate mask from proto and coefficients
      const float* coeffs = candidate_coeffs[n];

      // Weighted sum of proto masks
      cv::Mat mask_proto(proto_h, proto_w, CV_32FC1, cv::Scalar(0.0f));
      for (int c = 0; c < num_mask_coeffs; ++c) {
        cv::Mat proto_channel(
            proto_h, proto_w, CV_32FC1,
            const_cast<float*>(batch_proto + c * proto_h * proto_w));
        mask_proto += coeffs[c] * proto_channel;
      }

      // Sigmoid
      cv::Mat mask_sigmoid(proto_h, proto_w, CV_32FC1);
      for (int r = 0; r < proto_h; ++r) {
        float* src_row = mask_proto.ptr<float>(r);
        float* dst_row = mask_sigmoid.ptr<float>(r);
        for (int c = 0; c < proto_w; ++c) {
          dst_row[c] = 1.0f / (1.0f + std::exp(-src_row[c]));
        }
      }

      // Threshold at 0.5
      cv::Mat mask_binary_float;
      cv::threshold(mask_sigmoid, mask_binary_float, 0.5f, 1.0f,
                    cv::THRESH_BINARY);
      cv::Mat mask_binary;
      mask_binary_float.convertTo(mask_binary, CV_8UC1, 255.0);

      // Resize to model dimensions
      cv::Mat mask_resized;
      cv::resize(mask_binary, mask_resized,
                 cv::Size(param->model_w_, param->model_h_), 0, 0,
                 cv::INTER_NEAREST);

      // Create device::Tensor for mask
      device::Device* device = proto_tensor->getDevice();
      device::TensorDesc mask_desc;
      mask_desc.data_type_ = base::dataTypeOf<uint8_t>();
      mask_desc.shape_ = {1, param->model_h_, param->model_w_};
      device::Tensor* mask_tensor = new device::Tensor(device, mask_desc);
      uint8_t* mask_data = (uint8_t*)mask_tensor->getData();
      for (int r = 0; r < param->model_h_; ++r) {
        uint8_t* row_data = mask_resized.ptr<uint8_t>(r);
        memcpy(mask_data + r * param->model_w_, row_data,
               param->model_w_ * sizeof(uint8_t));
      }

      // Push bbox to BBoxResult
      detect::BBox bbox_item;
      bbox_item.index_ = candidates_result.bboxs_[n].index_;
      bbox_item.label_id_ = candidates_result.bboxs_[n].label_id_;
      bbox_item.label_name_ = candidates_result.bboxs_[n].label_name_;
      bbox_item.score_ = candidates_result.bboxs_[n].score_;
      bbox_item.bbox_ = candidates_result.bboxs_[n].bbox_;
      bbox_result->bboxs_.push_back(std::move(bbox_item));

      // Push mask to SegMaskResult
      SegMaskItem mask_item;
      mask_item.index_ = candidates_result.bboxs_[n].index_;
      mask_item.label_id_ = candidates_result.bboxs_[n].label_id_;
      mask_item.score_ = candidates_result.bboxs_[n].score_;
      mask_item.mask_ = mask_tensor;
      mask_result->masks_.push_back(std::move(mask_item));
    }
  }

  outputs_[0]->set(bbox_result, false);
  outputs_[1]->set(mask_result, false);
  return base::kStatusCodeOk;
}

REGISTER_NODE("nndeploy::segment::YoloSegPostProcess", YoloSegPostProcess);
REGISTER_NODE("nndeploy::segment::YoloSegGraph", YoloSegGraph);

}  // namespace segment
}  // namespace nndeploy
