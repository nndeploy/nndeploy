#include "nndeploy/model/detect/yolo/yolo_multi_output.h"

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/model/detect/util.h"
#include "nndeploy/model/infer.h"
#include "nndeploy/model/preprocess/cvtcolor_resize.h"

namespace nndeploy {
namespace model {

dag::TypeGraphRegister g_register_yolov5_multi_output_graph(
    NNDEPLOY_YOLOV5_MULTI_OUTPUT, createYoloV5MultiOutputGraph);

static inline float sigmoid(float x) {
  return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generateProposals(const int *anchors, int stride, const int model_w,
                              const int model_h, device::Tensor *tensor, float obj_threshold,
                              float score_threshold, DetectResult *results) {
//  const int num_grid = tensor->getHeight();

  int num_grid_y = tensor->getShapeIndex(2);
  int num_grid_x = tensor->getShapeIndex(3);
  int num_grid = num_grid_y * num_grid_x;

  const int num_class = tensor->getShapeIndex(1) / 3 - 5;
  const int num_anchors = 3;

  float *data = (float *)tensor->getPtr();

  for (int q = 0; q < num_anchors; q++) {
    const float anchor_w = anchors[q * 2];
    const float anchor_h = anchors[q * 2 + 1];

//    float *data_channel = data + q * num_grid * (num_class + 5);

    for (int i = 0; i < num_grid_y; i++) {
      for (int j = 0; j < num_grid_x; j++) {
        float *feat_ptr = data + (num_class + 5) * q * num_grid + i * num_grid_x + j;
        float box_confidence = feat_ptr[4 * num_grid];
        if (box_confidence >= obj_threshold) {
          // find class index with max class score
          int class_index = 0;
          float class_score = -FLT_MAX;
          for (int k = 0; k < num_class; k++) {
            float score = feat_ptr[(5 + k) * num_grid];
            if (score > class_score) {
              class_index = k;
              class_score = score;
            }
          }
          float confidence = box_confidence * class_score;
          if (confidence >= score_threshold) {
            // yolov5/models/yolo.py Detect forward
            // y = x[i].sigmoid()
            // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 +
            // self.grid[i].to(x[i].device)) * self.stride[i]  # xy y[..., 2:4]
            // = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

            float dx = *feat_ptr;
            float dy = feat_ptr[num_grid];
            float dw = feat_ptr[2 * num_grid];
            float dh = feat_ptr[3 * num_grid];

            float pb_cx = (dx * 2.f - 0.5f + j) * stride;
            float pb_cy = (dy * 2.f - 0.5f + i) * stride;

            float pb_w = pow(dw * 2.f, 2) * anchor_w;
            float pb_h = pow(dh * 2.f, 2) * anchor_h;

            float x0 = pb_cx - pb_w * 0.5f;
            float y0 = pb_cy - pb_h * 0.5f;
            float x1 = pb_cx + pb_w * 0.5f;
            float y1 = pb_cy + pb_h * 0.5f;

            DetectBBoxResult bbox;
            bbox.index_ = 0;
            bbox.bbox_[0] = x0 > 0 ? x0 : 0;
            bbox.bbox_[1] = y0 > 0 ? y0 : 0;
            bbox.bbox_[2] = x1 < model_w ? x1 : model_w;
            bbox.bbox_[3] = y1 < model_h ? y1 : model_h;
            bbox.label_id_ = class_index;
            bbox.score_ = confidence;

            results->bboxs_.emplace_back(bbox);
          }
        }
      }
    }
  }
}

base::Status YoloMultiOutputPostProcess::run() {
  YoloMultiOutputPostParam *param = (YoloMultiOutputPostParam *)param_.get();
  DetectResult *results = new DetectResult();
  outputs_[0]->set(results, inputs_[0]->getIndex(this), false);
  DetectResult results_batch;
  device::Tensor *tensor_0 = inputs_[0]->getTensor(this);
  generateProposals(param->anchors_stride_8, 8, param->model_w_,
                    param->model_h_, tensor_0, param->obj_threshold_, param->score_threshold_,
                    &results_batch);
  device::Tensor *tensor_1 = inputs_[1]->getTensor(this);
  generateProposals(param->anchors_stride_16, 16, param->model_w_,
                    param->model_h_, tensor_1, param->obj_threshold_, param->score_threshold_,
                    &results_batch);
  device::Tensor *tensor_2 = inputs_[2]->getTensor(this);
  generateProposals(param->anchors_stride_32, 32, param->model_w_,
                    param->model_h_, tensor_2, param->obj_threshold_, param->score_threshold_,
                    &results_batch);
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
  return base::kStatusCodeOk;
}

dag::Graph *createYoloV5MultiOutputGraph(const std::string &name,
                                         base::InferenceType inference_type,
                                         base::DeviceType device_type,
                                         dag::Edge *input, dag::Edge *output,
                                         base::ModelType model_type,
                                         bool is_path,
                                         std::vector<std::string> model_value) {
  dag::Graph *graph = new dag::Graph(name, input, output);
  dag::Edge *infer_input = graph->createEdge("images_205");
  dag::Edge *infer_0 = graph->createEdge("Sigmoid_Sigmoid_199/out0_0");
  dag::Edge *infer_1 = graph->createEdge("Sigmoid_Sigmoid_201/out0_1");
  dag::Edge *infer_2 = graph->createEdge("Sigmoid_Sigmoid_203/out0_2");

//  dag::Edge *infer_input = graph->createEdge("image");
//  dag::Edge *infer_0 = graph->createEdge("output0");
//  dag::Edge *infer_1 = graph->createEdge("output1");
//  dag::Edge *infer_2 = graph->createEdge("output2");

//  dag::Edge *infer_input = graph->createEdge("images");
//  dag::Edge *infer_0 = graph->createEdge("output0");
//  dag::Edge *infer_1 = graph->createEdge("329");
//  dag::Edge *infer_2 = graph->createEdge("331");

  dag::Node *pre = graph->createNode<model::CvtColorResize>("preprocess", input,
                                                            infer_input);

  dag::Node *infer = graph->createInfer<model::Infer>(
      "infer", inference_type, {infer_input}, {infer_0, infer_1, infer_2});

  dag::Node *post = graph->createNode<YoloMultiOutputPostProcess>(
      "postprocess", {infer_0, infer_1, infer_2}, {output});

  model::CvtclorResizeParam *pre_param =
      dynamic_cast<model::CvtclorResizeParam *>(pre->getParam());
  pre_param->src_pixel_type_ = base::kPixelTypeBGR;
  pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
  pre_param->interp_type_ = base::kInterpTypeLinear;
  pre_param->h_ = 640;
  pre_param->w_ = 640;
  pre_param->normalize_ = false;

  inference::InferenceParam *inference_param =
      (inference::InferenceParam *)(infer->getParam());
  inference_param->is_path_ = is_path;
  inference_param->model_value_ = model_value;
  inference_param->device_type_ = device_type;
  inference_param->model_type_ = model_type;

  // TODO: 很多信息可以从 preprocess 和 infer 中获取
  YoloMultiOutputPostParam *post_param =
      dynamic_cast<YoloMultiOutputPostParam *>(post->getParam());
  post_param->score_threshold_ = 0.5 * 0.5;
  post_param->obj_threshold_ = 0.5;
  post_param->nms_threshold_ = 0.3;
  post_param->num_classes_ = 80;
  post_param->model_h_ = 640;
  post_param->model_w_ = 640;
  post_param->version_ = 5;

//  post_param->name_stride_8 = "output";
//  post_param->name_stride_16 = "376";
//  post_param->name_stride_32 = "401";

  return graph;
}

}  // namespace model
}  // namespace nndeploy
