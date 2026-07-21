#include "nndeploy/detect/yolo/yolo.h"

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
namespace detect {

base::Status YoloPostParam::serialize(
    rapidjson::Value& json, rapidjson::Document::AllocatorType& allocator) {
  json.AddMember("version_", version_, allocator);
  json.AddMember("e2e_", e2e_, allocator);
  json.AddMember("score_threshold_", score_threshold_, allocator);
  json.AddMember("nms_threshold_", nms_threshold_, allocator);
  json.AddMember("num_classes_", num_classes_, allocator);
  json.AddMember("model_h_", model_h_, allocator);
  json.AddMember("model_w_", model_w_, allocator);
  return base::kStatusCodeOk;
}

base::Status YoloPostParam::deserialize(rapidjson::Value& json) {
  if (json.HasMember("version_") && json["version_"].IsInt()) {
    version_ = json["version_"].GetInt();
  }

  if (json.HasMember("e2e_") && json["e2e_"].IsBool()) {
    e2e_ = json["e2e_"].GetBool();
  }

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

base::Status YoloPostProcess::run() {
  // NNDEPLOY_LOGE("YoloPostProcess::run!Thread ID: %d.\n",
  //               std::this_thread::get_id());
  YoloPostParam* param = (YoloPostParam*)param_.get();

  if (param->e2e_) {
    // E2E (end-to-end): (batch, 300, 6) format, NMS already applied
    // Supported: YOLO26 E2E
    return runE2E();
  }

  if (param->version_ == 5 || param->version_ == 6 || param->version_ == 7) {
    // YOLOv5/6/7: (batch, num_predictions, 85) format with objectness
    return runV5V6();
  } else if (param->version_ == 8 || param->version_ == 9 ||
             param->version_ == 11 || param->version_ == 12) {
    // YOLOv8/v9/11/12: (batch, channels, num_predictions) format
    // accordingly
    return runV8V11();
  } else if (param->version_ == 10 || param->version_ == 26) {
    // E2E (end-to-end): (batch, 300, 6) format, NMS already applied,
    // [x1,y1,x2,y2,score,class]
    return runE2E();
  } else {
    NNDEPLOY_LOGE("Unsupported version: %d", param->version_);
    return base::kStatusCodeErrorInvalidValue;
  }
  return base::kStatusCodeOk;
}

base::Status YoloPostProcess::runV5V6() {
  YoloPostParam* param = (YoloPostParam*)param_.get();
  float score_threshold = param->score_threshold_;
  int num_classes = param->num_classes_;

  device::Tensor* tensor = inputs_[0]->getTensor(this);
  float* data = (float*)tensor->getData();
  int batch = tensor->getShapeIndex(0);
  int height = tensor->getShapeIndex(1);
  int width = tensor->getShapeIndex(2);

  DetectResult* results = new DetectResult();

  for (int b = 0; b < batch; ++b) {
    // NNDEPLOY_LOGE("bk\n");
    float* data_batch = data + b * height * width;
    DetectResult results_batch;
    for (int h = 0; h < height; ++h) {
      float* data_row = data_batch + h * width;
      float x_center = data_row[0];
      float y_center = data_row[1];
      float object_w = data_row[2];
      float object_h = data_row[3];
      float x0 = x_center - object_w * 0.5f;
      x0 = x0 > 0.0 ? x0 : 0.0;
      float y0 = y_center - object_h * 0.5f;
      y0 = y0 > 0.0 ? y0 : 0.0;
      float x1 = x_center + object_w * 0.5f;
      x1 = x1 < param->model_w_ ? x1 : param->model_w_;
      float y1 = y_center + object_h * 0.5f;
      y1 = y1 < param->model_h_ ? y1 : param->model_h_;
      float box_objectness = data_row[4];
      for (int class_idx = 0; class_idx < num_classes; ++class_idx) {
        float score = box_objectness * data_row[5 + class_idx];
        if (score > score_threshold) {
          DetectBBoxResult bbox;
          bbox.index_ = b;
          bbox.label_id_ = class_idx;
          bbox.score_ = score;
          bbox.bbox_[0] = x0;
          bbox.bbox_[1] = y0;
          bbox.bbox_[2] = x1;
          bbox.bbox_[3] = y1;
          // NNDEPLOY_LOGE("score:%f, x0:%f, y0:%f, x1:%f, y1:%f\n", score, x0,
          // y0,
          //               x1, y1);
          results_batch.bboxs_.emplace_back(bbox);
        }
      }
    }
    std::vector<int> keep_idxs(results_batch.bboxs_.size());
    computeNMS(results_batch, keep_idxs, param->nms_threshold_);
    for (auto i = 0; i < keep_idxs.size(); ++i) {
      auto n = keep_idxs[i];
      if (n < 0) {
        continue;
      }
      results_batch.bboxs_[n].bbox_[0] /= param->model_w_;
      results_batch.bboxs_[n].bbox_[1] /= param->model_h_;
      results_batch.bboxs_[n].bbox_[2] /= param->model_w_;
      results_batch.bboxs_[n].bbox_[3] /= param->model_h_;
      results->bboxs_.emplace_back(results_batch.bboxs_[n]);
    }
  }
  // Build lightweight BBoxResult for DrawBBox (output_0)
  BBoxResult* bbox_result = new BBoxResult();
  for (const auto& det : results->bboxs_) {
    BBox bbox;
    bbox.index_ = det.index_;
    bbox.label_id_ = det.label_id_;
    bbox.label_name_ = det.label_name_;
    bbox.score_ = det.score_;
    bbox.bbox_ = det.bbox_;
    bbox_result->bboxs_.push_back(std::move(bbox));
  }
  outputs_[0]->set(bbox_result, false);
  if (outputs_.size() > 1) {
    outputs_[1]->set(results, false);
  }
  return base::kStatusCodeOk;
}

base::Status YoloPostProcess::runV8V11() {
  YoloPostParam* param = (YoloPostParam*)param_.get();
  float score_threshold = param->score_threshold_;
  int num_classes = param->num_classes_;

  device::Tensor* tensor = inputs_[0]->getTensor(this);
  float* data = (float*)tensor->getData();
  int batch = tensor->getShapeIndex(0);
  int dim1 = tensor->getShapeIndex(1);
  int dim2 = tensor->getShapeIndex(2);
  int expected_channels = 4 + num_classes;

  {
    auto shape = tensor->getShape();
    std::string shape_str;
    for (size_t di = 0; di < shape.size(); ++di) {
      if (di > 0) shape_str += ",";
      shape_str += std::to_string(shape[di]);
    }
    float* dbg = (float*)tensor->getData();
    char buf[512];
    int off = 0;
    for (int k = 0; k < 16 && k < (int)(tensor->getSize() / sizeof(float)); ++k)
      off += snprintf(buf + off, sizeof(buf) - off, "%s%.6f", k == 0 ? "" : ",",
                      dbg[k]);
    NNDEPLOY_LOGE("[YoloV8V11] shape=[%s] expected_c=%d first_16=%s",
                  shape_str.c_str(), expected_channels, buf);
  }

  // Auto-detect: NCHW (batch,C,N) vs NHWC (batch,N,C)
  bool need_transpose = false;
  if (dim1 == expected_channels) {
    need_transpose = true;
  } else if (dim2 == expected_channels) {
    need_transpose = false;
  } else {
    need_transpose = (dim1 <= dim2);
  }

  int height, width;
  if (need_transpose) {
    NNDEPLOY_LOGE("[YoloV8V11] layout=NCHW, transpose (%d,%d) -> (%d,%d)\n",
                  dim1, dim2, dim2, dim1);
    cv::Mat cv_mat_src(dim1, dim2, CV_32FC1, data);
    cv::Mat cv_mat_dst(dim2, dim1, CV_32FC1);
    cv::transpose(cv_mat_src, cv_mat_dst);
    height = dim2;
    width = dim1;
    data = (float*)cv_mat_dst.data;
  } else {
    NNDEPLOY_LOGE("[YoloV8V11] layout=NHWC, use as-is (N=%d, C=%d)\n", dim1,
                  dim2);
    height = dim1;
    width = dim2;
  }

  DetectResult* results = new DetectResult();

  for (int b = 0; b < batch; ++b) {
    float* data_batch = data + b * height * width;
    DetectResult results_batch;
    for (int h = 0; h < height; ++h) {
      float* data_row = data_batch + h * width;
      float x_center = data_row[0];
      float y_center = data_row[1];
      float object_w = data_row[2];
      float object_h = data_row[3];
      float x0 = x_center - object_w * 0.5f;
      x0 = x0 > 0.0 ? x0 : 0.0;
      float y0 = y_center - object_h * 0.5f;
      y0 = y0 > 0.0 ? y0 : 0.0;
      float x1 = x_center + object_w * 0.5f;
      x1 = x1 < param->model_w_ ? x1 : param->model_w_;
      float y1 = y_center + object_h * 0.5f;
      y1 = y1 < param->model_h_ ? y1 : param->model_h_;
      for (int class_idx = 0; class_idx < num_classes; ++class_idx) {
        float score = data_row[4 + class_idx];
        if (score > score_threshold) {
          DetectBBoxResult bbox;
          bbox.index_ = b;
          bbox.label_id_ = class_idx;
          bbox.score_ = score;
          bbox.bbox_[0] = x0;
          bbox.bbox_[1] = y0;
          bbox.bbox_[2] = x1;
          bbox.bbox_[3] = y1;
          results_batch.bboxs_.emplace_back(bbox);
        }
      }
    }
    std::vector<int> keep_idxs(results_batch.bboxs_.size());
    computeNMS(results_batch, keep_idxs, param->nms_threshold_);
    for (auto i = 0; i < keep_idxs.size(); ++i) {
      auto n = keep_idxs[i];
      if (n < 0) {
        continue;
      }
      results_batch.bboxs_[n].bbox_[0] /= param->model_w_;
      results_batch.bboxs_[n].bbox_[1] /= param->model_h_;
      results_batch.bboxs_[n].bbox_[2] /= param->model_w_;
      results_batch.bboxs_[n].bbox_[3] /= param->model_h_;
      results->bboxs_.emplace_back(results_batch.bboxs_[n]);
    }
  }
  BBoxResult* bbox_result = new BBoxResult();
  for (const auto& det : results->bboxs_) {
    BBox bbox;
    bbox.index_ = det.index_;
    bbox.label_id_ = det.label_id_;
    bbox.label_name_ = det.label_name_;
    bbox.score_ = det.score_;
    bbox.bbox_ = det.bbox_;
    bbox_result->bboxs_.push_back(std::move(bbox));
  }
  outputs_[0]->set(bbox_result, false);
  if (outputs_.size() > 1) {
    outputs_[1]->set(results, false);
  }
  return base::kStatusCodeOk;
}

base::Status YoloPostProcess::runE2E() {
  YoloPostParam* param = (YoloPostParam*)param_.get();
  float score_threshold = param->score_threshold_;

  device::Tensor* tensor = inputs_[0]->getTensor(this);
  float* data = (float*)tensor->getData();
  int batch = tensor->getShapeIndex(0);
  int num_detections = tensor->getShapeIndex(1);
  int channels = tensor->getShapeIndex(2);

  // Guard: E2E format is always (batch, N, 6) where channels=6
  // with fields [x1, y1, x2, y2, score, class_id].
  // If channels != 6, the model is likely in non-E2E format routed here
  // incorrectly (e.g. a future YOLO variant that restored raw output).
  if (channels != 6) {
    NNDEPLOY_LOGE(
        "[runE2E] expected 6 channels per detection but got %d — "
        "model may not be E2E format; shape=(%d,%d,%d)",
        channels, batch, num_detections, channels);
    return base::kStatusCodeErrorInvalidValue;
  }

  DetectResult* results = new DetectResult();

  for (int b = 0; b < batch; ++b) {
    float* batch_data = data + b * num_detections * channels;

    for (int i = 0; i < num_detections; ++i) {
      float* row = batch_data + i * channels;
      // NOTE: E2E model outputs are already in xyxy format
      // [x1,y1,x2,y2,score,class] in pixel coordinates of the model input
      // (model_w_ × model_h_). Normalize to [0,1] for the downstream DrawBox
      // node.
      float x1 = row[0] / param->model_w_;
      float y1 = row[1] / param->model_h_;
      float x2 = row[2] / param->model_w_;
      float y2 = row[3] / param->model_h_;
      float score = row[4];
      int class_id = (int)row[5];

      if (score < score_threshold) {
        continue;
      }

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

  BBoxResult* bbox_result = new BBoxResult();
  for (const auto& det : results->bboxs_) {
    BBox bbox;
    bbox.index_ = det.index_;
    bbox.label_id_ = det.label_id_;
    bbox.label_name_ = det.label_name_;
    bbox.score_ = det.score_;
    bbox.bbox_ = det.bbox_;
    bbox_result->bboxs_.push_back(std::move(bbox));
  }
  outputs_[0]->set(bbox_result, false);
  if (outputs_.size() > 1) {
    outputs_[1]->set(results, false);
  }
  return base::kStatusCodeOk;
}

// dag::Graph *createYoloV5Graph(const std::string &name,
//                               base::InferenceType inference_type,
//                               base::DeviceType device_type, dag::Edge *input,
//                               dag::Edge *output, base::ModelType model_type,
//                               bool is_path,
//                               std::vector<std::string> model_value) {
//   dag::Graph *graph = new dag::Graph(name, {input}, {output});
//   dag::Edge *infer_input = graph->createEdge("images");
//   dag::Edge *infer_output = graph->createEdge("output0");

//   dag::Node *pre = graph->createNode<preprocess::CvtResizeNormTrans>(
//       "preprocess", {input}, {infer_input});

//   dag::Node *infer = graph->createNode<infer::Infer>(
//       "infer", {infer_input}, {infer_output});

//   dag::Node *post =
//       graph->createNode<YoloPostProcess>("postprocess", {infer_output},
//       {output});

//   preprocess::CvtResizeNormTransParam *pre_param =
//       dynamic_cast<preprocess::CvtResizeNormTransParam *>(pre->getParam());
//   pre_param->src_pixel_type_ = base::kPixelTypeBGR;
//   pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
//   pre_param->interp_type_ = base::kInterpTypeLinear;
//   pre_param->h_ = 640;
//   pre_param->w_ = 640;

//   inference::InferenceParam *inference_param =
//       (inference::InferenceParam *)(infer->getParam());
//   inference_param->is_path_ = is_path;
//   inference_param->model_value_ = model_value;
//   inference_param->device_type_ = device_type;
//   inference_param->model_type_ = model_type;

//   // TODO: 很多信息可以从 preprocess 和 infer 中获取
//   YoloPostParam *post_param = dynamic_cast<YoloPostParam
//   *>(post->getParam()); post_param->score_threshold_ = 0.5;
//   post_param->nms_threshold_ = 0.45;
//   post_param->num_classes_ = 80;
//   post_param->model_h_ = 640;
//   post_param->model_w_ = 640;
//   post_param->version_ = 5;

//   return graph;
// }

// dag::Graph *createYoloV6Graph(const std::string &name,
//                               base::InferenceType inference_type,
//                               base::DeviceType device_type, dag::Edge *input,
//                               dag::Edge *output, base::ModelType model_type,
//                               bool is_path,
//                               std::vector<std::string> model_value) {
//   dag::Graph *graph = new dag::Graph(name, {input}, {output});
//   dag::Edge *infer_input = graph->createEdge("images");
//   dag::Edge *infer_output = graph->createEdge("outputs");

//   dag::Node *pre = graph->createNode<preprocess::CvtResizeNormTrans>(
//       "preprocess", {input}, {infer_input});

//   dag::Node *infer = graph->createNode<infer::Infer>(
//       "infer", {infer_input}, {infer_output});

//   dag::Node *post =
//       graph->createNode<YoloPostProcess>("postprocess", {infer_output},
//       {output});

//   preprocess::CvtResizeNormTransParam *pre_param =
//       dynamic_cast<preprocess::CvtResizeNormTransParam *>(pre->getParam());
//   pre_param->src_pixel_type_ = base::kPixelTypeBGR;
//   pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
//   pre_param->interp_type_ = base::kInterpTypeLinear;
//   pre_param->h_ = 640;
//   pre_param->w_ = 640;

//   inference::InferenceParam *inference_param =
//       (inference::InferenceParam *)(infer->getParam());
//   inference_param->is_path_ = is_path;
//   inference_param->model_value_ = model_value;
//   inference_param->device_type_ = device_type;
//   inference_param->model_type_ = model_type;

//   // TODO: 很多信息可以从 preprocess 和 infer 中获取
//   YoloPostParam *post_param = dynamic_cast<YoloPostParam
//   *>(post->getParam()); post_param->score_threshold_ = 0.5;
//   post_param->nms_threshold_ = 0.45;
//   post_param->num_classes_ = 80;
//   post_param->model_h_ = 640;
//   post_param->model_w_ = 640;
//   post_param->version_ = 6;

//   return graph;
// }

// dag::Graph *createYoloV8Graph(const std::string &name,
//                               base::InferenceType inference_type,
//                               base::DeviceType device_type, dag::Edge *input,
//                               dag::Edge *output, base::ModelType model_type,
//                               bool is_path,
//                               std::vector<std::string> model_value) {
//   dag::Graph *graph = new dag::Graph(name, {input}, {output});
//   dag::Edge *infer_input = graph->createEdge("images");
//   dag::Edge *infer_output = graph->createEdge("output0");

//   dag::Node *pre = graph->createNode<preprocess::CvtResizeNormTrans>(
//       "preprocess", {input}, {infer_input});

//   dag::Node *infer = graph->createNode<infer::Infer>(
//       "infer", {infer_input}, {infer_output});

//   dag::Node *post =
//       graph->createNode<YoloPostProcess>("postprocess", {infer_output},
//       {output});

//   preprocess::CvtResizeNormTransParam *pre_param =
//       dynamic_cast<preprocess::CvtResizeNormTransParam *>(pre->getParam());
//   pre_param->src_pixel_type_ = base::kPixelTypeBGR;
//   pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
//   pre_param->interp_type_ = base::kInterpTypeLinear;
//   pre_param->h_ = 640;
//   pre_param->w_ = 640;

//   inference::InferenceParam *inference_param =
//       (inference::InferenceParam *)(infer->getParam());
//   inference_param->is_path_ = is_path;
//   inference_param->model_value_ = model_value;
//   inference_param->device_type_ = device_type;
//   inference_param->model_type_ = model_type;

//   // TODO: 很多信息可以从 preprocess 和 infer 中获取
//   YoloPostParam *post_param = dynamic_cast<YoloPostParam
//   *>(post->getParam()); post_param->score_threshold_ = 0.5;
//   post_param->nms_threshold_ = 0.45;
//   post_param->num_classes_ = 80;
//   post_param->model_h_ = 640;
//   post_param->model_w_ = 640;
//   post_param->version_ = 8;

//   return graph;
// }

// dag::Graph *createYoloV11Graph(const std::string &name,
//                                base::InferenceType inference_type,
//                                base::DeviceType device_type, dag::Edge
//                                *input, dag::Edge *output, base::ModelType
//                                model_type, bool is_path,
//                                std::vector<std::string> model_value) {
//   dag::Graph *graph = new dag::Graph(name, {input}, {output});
//   dag::Edge *infer_input = graph->createEdge("images");
//   dag::Edge *infer_output = graph->createEdge("output0");

//   dag::Node *pre = graph->createNode<preprocess::CvtResizeNormTrans>(
//       "preprocess", {input}, {infer_input});

//   dag::Node *infer = graph->createNode<infer::Infer>(
//       "infer", {infer_input}, {infer_output});

//   dag::Node *post =
//       graph->createNode<YoloPostProcess>("postprocess", {infer_output},
//       {output});

//   preprocess::CvtResizeNormTransParam *pre_param =
//       dynamic_cast<preprocess::CvtResizeNormTransParam *>(pre->getParam());
//   pre_param->src_pixel_type_ = base::kPixelTypeBGR;
//   pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
//   pre_param->interp_type_ = base::kInterpTypeLinear;
//   pre_param->h_ = 640;
//   pre_param->w_ = 640;

//   inference::InferenceParam *inference_param =
//       (inference::InferenceParam *)(infer->getParam());
//   inference_param->is_path_ = is_path;
//   inference_param->model_value_ = model_value;
//   inference_param->device_type_ = device_type;
//   inference_param->model_type_ = model_type;

//   // TODO: 很多信息可以从 preprocess 和 infer 中获取
//   YoloPostParam *post_param = dynamic_cast<YoloPostParam
//   *>(post->getParam()); post_param->score_threshold_ = 0.5;
//   post_param->nms_threshold_ = 0.45;
//   post_param->num_classes_ = 80;
//   post_param->model_h_ = 640;
//   post_param->model_w_ = 640;
//   post_param->version_ = 8;

//   return graph;
// }

REGISTER_NODE("nndeploy::detect::YoloPostProcess", YoloPostProcess);
REGISTER_NODE("nndeploy::detect::YoloGraph", YoloGraph);

}  // namespace detect
}  // namespace nndeploy
