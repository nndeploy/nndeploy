#include "nndeploy/detect/yolo/yolo.h"

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/any.h"
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

dag::TypeGraphRegister g_register_yolov5_graph(NNDEPLOY_YOLOV5,
                                               createYoloV5Graph);
dag::TypeGraphRegister g_register_yolov6_graph(NNDEPLOY_YOLOV6,
                                               createYoloV6Graph);
dag::TypeGraphRegister g_register_yolov8_graph(NNDEPLOY_YOLOV8,
                                               createYoloV8Graph);
dag::TypeGraphRegister g_register_yolov11_graph(NNDEPLOY_YOLOV11,
                                               createYoloV11Graph);

base::Status YoloPostProcess::run() {
  // NNDEPLOY_LOGE("YoloPostProcess::run!Thread ID: %d.\n",
  //               std::this_thread::get_id());
  YoloPostParam *param = (YoloPostParam *)param_.get();

  if (param->version_ == 5 || param->version_ == 6) {
    return runV5V6();
  } else if (param->version_ == 8 || param->version_ == 11) {
    return runV8();
  } else {
    NNDEPLOY_LOGE("Unsupported version: %d", param->version_);
    return base::kStatusCodeErrorInvalidValue;
  }
  return base::kStatusCodeOk;
}

base::Status YoloPostProcess::runV5V6() {
  YoloPostParam *param = (YoloPostParam *)param_.get();
  float score_threshold = param->score_threshold_;
  int num_classes = param->num_classes_;

  device::Tensor *tensor = inputs_[0]->getTensor(this);
  float *data = (float *)tensor->getData();
  int batch = tensor->getShapeIndex(0);
  int height = tensor->getShapeIndex(1);
  int width = tensor->getShapeIndex(2);

  DetectResult *results = new DetectResult();

  for (int b = 0; b < batch; ++b) {
    // NNDEPLOY_LOGE("bk\n");
    float *data_batch = data + b * height * width;
    DetectResult results_batch;
    for (int h = 0; h < height; ++h) {
      float *data_row = data_batch + h * width;
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
  outputs_[0]->set(results, inputs_[0]->getIndex(this), false);
  return base::kStatusCodeOk;
}

base::Status YoloPostProcess::runV8() {
  YoloPostParam *param = (YoloPostParam *)param_.get();
  float score_threshold = param->score_threshold_;
  int num_classes = param->num_classes_;

  device::Tensor *tensor = inputs_[0]->getTensor(this);
  // NNDEPLOY_LOGI("**********%d,%d,%d,%d", tensor->getBatch(),
  //               tensor->getChannel(), tensor->getHeight(),
  //               tensor->getWidth());
  float *data = (float *)tensor->getData();
  int batch = tensor->getShapeIndex(0);
  int height = tensor->getShapeIndex(1);
  int width = tensor->getShapeIndex(2);

  cv::Mat cv_mat_src(height, width, CV_32FC1, data);
  cv::Mat cv_mat_dst(width, height, CV_32FC1);
  cv::transpose(cv_mat_src, cv_mat_dst);
  std::swap(height, width);
  data = (float *)cv_mat_dst.data;

  DetectResult *results = new DetectResult();

  for (int b = 0; b < batch; ++b) {
    float *data_batch = data + b * height * width;
    DetectResult results_batch;
    for (int h = 0; h < height; ++h) {
      float *data_row = data_batch + h * width;
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
  outputs_[0]->set(results, inputs_[0]->getIndex(this), false);
  // NNDEPLOY_LOGE("postprocess!\n");
  return base::kStatusCodeOk;
}

dag::Graph *createYoloV5Graph(const std::string &name,
                              base::InferenceType inference_type,
                              base::DeviceType device_type, dag::Edge *input,
                              dag::Edge *output, base::ModelType model_type,
                              bool is_path,
                              std::vector<std::string> model_value) {
  dag::Graph *graph = new dag::Graph(name, input, output);
  dag::Edge *infer_input = graph->createEdge("images");
  dag::Edge *infer_output = graph->createEdge("output0");

  dag::Node *pre = graph->createNode<preprocess::CvtColorResize>(
      "preprocess", input, infer_input);

  dag::Node *infer = graph->createInfer<infer::Infer>(
      "infer", inference_type, infer_input, infer_output);

  dag::Node *post =
      graph->createNode<YoloPostProcess>("postprocess", infer_output, output);

  preprocess::CvtclorResizeParam *pre_param =
      dynamic_cast<preprocess::CvtclorResizeParam *>(pre->getParam());
  pre_param->src_pixel_type_ = base::kPixelTypeBGR;
  pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
  pre_param->interp_type_ = base::kInterpTypeLinear;
  pre_param->h_ = 640;
  pre_param->w_ = 640;

  inference::InferenceParam *inference_param =
      (inference::InferenceParam *)(infer->getParam());
  inference_param->is_path_ = is_path;
  inference_param->model_value_ = model_value;
  inference_param->device_type_ = device_type;
  inference_param->model_type_ = model_type;

  // TODO: 很多信息可以从 preprocess 和 infer 中获取
  YoloPostParam *post_param = dynamic_cast<YoloPostParam *>(post->getParam());
  post_param->score_threshold_ = 0.5;
  post_param->nms_threshold_ = 0.45;
  post_param->num_classes_ = 80;
  post_param->model_h_ = 640;
  post_param->model_w_ = 640;
  post_param->version_ = 5;

  return graph;
}

dag::Graph *createYoloV6Graph(const std::string &name,
                              base::InferenceType inference_type,
                              base::DeviceType device_type, dag::Edge *input,
                              dag::Edge *output, base::ModelType model_type,
                              bool is_path,
                              std::vector<std::string> model_value) {
  dag::Graph *graph = new dag::Graph(name, input, output);
  dag::Edge *infer_input = graph->createEdge("images");
  dag::Edge *infer_output = graph->createEdge("outputs");

  dag::Node *pre = graph->createNode<preprocess::CvtColorResize>(
      "preprocess", input, infer_input);

  dag::Node *infer = graph->createInfer<infer::Infer>(
      "infer", inference_type, infer_input, infer_output);

  dag::Node *post =
      graph->createNode<YoloPostProcess>("postprocess", infer_output, output);

  preprocess::CvtclorResizeParam *pre_param =
      dynamic_cast<preprocess::CvtclorResizeParam *>(pre->getParam());
  pre_param->src_pixel_type_ = base::kPixelTypeBGR;
  pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
  pre_param->interp_type_ = base::kInterpTypeLinear;
  pre_param->h_ = 640;
  pre_param->w_ = 640;

  inference::InferenceParam *inference_param =
      (inference::InferenceParam *)(infer->getParam());
  inference_param->is_path_ = is_path;
  inference_param->model_value_ = model_value;
  inference_param->device_type_ = device_type;
  inference_param->model_type_ = model_type;

  // TODO: 很多信息可以从 preprocess 和 infer 中获取
  YoloPostParam *post_param = dynamic_cast<YoloPostParam *>(post->getParam());
  post_param->score_threshold_ = 0.5;
  post_param->nms_threshold_ = 0.45;
  post_param->num_classes_ = 80;
  post_param->model_h_ = 640;
  post_param->model_w_ = 640;
  post_param->version_ = 6;

  return graph;
}

dag::Graph *createYoloV8Graph(const std::string &name,
                              base::InferenceType inference_type,
                              base::DeviceType device_type, dag::Edge *input,
                              dag::Edge *output, base::ModelType model_type,
                              bool is_path,
                              std::vector<std::string> model_value) {
  dag::Graph *graph = new dag::Graph(name, input, output);
  dag::Edge *infer_input = graph->createEdge("images");
  dag::Edge *infer_output = graph->createEdge("output0");

  dag::Node *pre = graph->createNode<preprocess::CvtColorResize>(
      "preprocess", input, infer_input);

  dag::Node *infer = graph->createInfer<infer::Infer>(
      "infer", inference_type, infer_input, infer_output);

  dag::Node *post =
      graph->createNode<YoloPostProcess>("postprocess", infer_output, output);

  preprocess::CvtclorResizeParam *pre_param =
      dynamic_cast<preprocess::CvtclorResizeParam *>(pre->getParam());
  pre_param->src_pixel_type_ = base::kPixelTypeBGR;
  pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
  pre_param->interp_type_ = base::kInterpTypeLinear;
  pre_param->h_ = 640;
  pre_param->w_ = 640;

  inference::InferenceParam *inference_param =
      (inference::InferenceParam *)(infer->getParam());
  inference_param->is_path_ = is_path;
  inference_param->model_value_ = model_value;
  inference_param->device_type_ = device_type;
  inference_param->model_type_ = model_type;

  // TODO: 很多信息可以从 preprocess 和 infer 中获取
  YoloPostParam *post_param = dynamic_cast<YoloPostParam *>(post->getParam());
  post_param->score_threshold_ = 0.5;
  post_param->nms_threshold_ = 0.45;
  post_param->num_classes_ = 80;
  post_param->model_h_ = 640;
  post_param->model_w_ = 640;
  post_param->version_ = 8;

  return graph;
}


dag::Graph *createYoloV11Graph(const std::string &name,
                              base::InferenceType inference_type,
                              base::DeviceType device_type, dag::Edge *input,
                              dag::Edge *output, base::ModelType model_type,
                              bool is_path,
                              std::vector<std::string> model_value) {
  dag::Graph *graph = new dag::Graph(name, input, output);
  dag::Edge *infer_input = graph->createEdge("images");
  dag::Edge *infer_output = graph->createEdge("output0");

  dag::Node *pre = graph->createNode<preprocess::CvtColorResize>(
      "preprocess", input, infer_input);

  dag::Node *infer = graph->createInfer<infer::Infer>(
      "infer", inference_type, infer_input, infer_output);

  dag::Node *post =
      graph->createNode<YoloPostProcess>("postprocess", infer_output, output);

  preprocess::CvtclorResizeParam *pre_param =
      dynamic_cast<preprocess::CvtclorResizeParam *>(pre->getParam());
  pre_param->src_pixel_type_ = base::kPixelTypeBGR;
  pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
  pre_param->interp_type_ = base::kInterpTypeLinear;
  pre_param->h_ = 640;
  pre_param->w_ = 640;

  inference::InferenceParam *inference_param =
      (inference::InferenceParam *)(infer->getParam());
  inference_param->is_path_ = is_path;
  inference_param->model_value_ = model_value;
  inference_param->device_type_ = device_type;
  inference_param->model_type_ = model_type;

  // TODO: 很多信息可以从 preprocess 和 infer 中获取
  YoloPostParam *post_param = dynamic_cast<YoloPostParam *>(post->getParam());
  post_param->score_threshold_ = 0.5;
  post_param->nms_threshold_ = 0.45;
  post_param->num_classes_ = 80;
  post_param->model_h_ = 640;
  post_param->model_w_ = 640;
  post_param->version_ = 8;

  return graph;
}

}  // namespace detect
}  // namespace nndeploy
