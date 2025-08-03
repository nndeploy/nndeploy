#include "nndeploy/detect/yolo/yolo_multi_conv_output.h"

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
#include "nndeploy/detect/util.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/infer/infer.h"
#include "nndeploy/preprocess/cvt_resize_norm_trans.h"
#include "nndeploy/preprocess/cvt_resize_pad_norm_trans.h"
#include "nndeploy/preprocess/warp_affine_cvt_norm_trans.h"

namespace nndeploy {
namespace detect {

static inline float sigmoid(float x) {
  return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void nhwc_to_nchw(float *data, int h, int w, int c) {
  float *dst = new float[h * w * c];
  for (int i = 0; i < h * w; ++i) {
    float *src_pt = data + i * c;
    float *dst_pt = dst + i;
    for (int j = 0; j < c; ++j) {
      *dst_pt = (c == 4) ? std::exp(*src_pt++) : (*src_pt++);
      dst_pt += h * w;
    }
  }
  memcpy(data, dst, sizeof(float) * h * w * c);
  delete[] dst;
}

static void generateProposals(const int *anchors, const int *strides,
                              int stride, const int model_w, const int model_h,
                              const int det_len, device::Tensor *tensor,
                              float score_threshold, DetectResult *results) {
  int num_grid_x = (int)(model_w / strides[stride]);
  int num_grid_y = (int)(model_h / strides[stride]);

  float *data = (float *)tensor->getData();

  base::IntVector shapevector = tensor->getShape();

  int h = tensor->getHeight();
  int w = tensor->getWidth();
  int c = tensor->getChannel();

  if (tensor->getDataFormat() == base::kDataFormatNHWC) {
    nhwc_to_nchw(data, h, w, c);
  }

  for (int anchor = 0; anchor < 3; ++anchor) {
    const float anchor_w = anchors[anchor * 2];
    const float anchor_h = anchors[anchor * 2 + 1];

    for (int i = 0; i < num_grid_x * num_grid_y; ++i) {
      int obj_index = i + 4 * num_grid_x * num_grid_y +
                      anchor * 85 * num_grid_x * num_grid_y;
      float objness = sigmoid(data[obj_index]);

      if (objness < score_threshold) continue;

      int label = 0;
      float prob = 0.0;
      for (int index = 5; index < 85; index++) {
        int class_index = i + index * num_grid_x * num_grid_y +
                          anchor * 85 * num_grid_x * num_grid_y;
        if (sigmoid(data[class_index]) > prob) {
          label = index - 5;
          prob = sigmoid(data[class_index]);
        }
      }

      float confidence = prob * objness;
      if (confidence < score_threshold) continue;

      int grid_y = (i / num_grid_x) % num_grid_x;
      int grid_x = i - grid_y * num_grid_x;

      int x_index = i + 0 * num_grid_x * num_grid_y +
                    anchor * 85 * num_grid_x * num_grid_y;
      float x_data = sigmoid(data[x_index]);
      x_data = (x_data * 2.0f + grid_x - 0.5f) * strides[stride];

      int y_index = i + 1 * num_grid_x * num_grid_y +
                    anchor * 85 * num_grid_x * num_grid_y;
      float y_data = sigmoid(data[y_index]);
      y_data = (y_data * 2.0f + grid_y - 0.5f) * strides[stride];

      int w_index = i + 2 * num_grid_x * num_grid_y +
                    anchor * 85 * num_grid_x * num_grid_y;
      float w_data = sigmoid(data[w_index]);
      w_data = (w_data * 2.0f) * (w_data * 2.0f) * anchor_w;

      int h_index = i + 3 * num_grid_x * num_grid_y +
                    anchor * 85 * num_grid_x * num_grid_y;
      float h_data = sigmoid(data[h_index]);
      h_data = (h_data * 2.0f) * (h_data * 2.0f) * anchor_h;

      float x = x_data;
      float y = y_data;
      float width = w_data;
      float height = h_data;
      float x0 = x - width * 0.5;
      float y0 = y - height * 0.5;
      float x1 = x + width * 0.5;
      float y1 = y + height * 0.5;

      DetectBBoxResult bbox;
      bbox.index_ = 0;
      bbox.bbox_[0] = x0 > 0 ? x0 : 0;
      bbox.bbox_[1] = y0 > 0 ? y0 : 0;
      bbox.bbox_[2] = x1 < model_w ? x1 : model_w;
      bbox.bbox_[3] = y1 < model_h ? y1 : model_h;
      bbox.label_id_ = label;
      bbox.score_ = confidence;
      results->bboxs_.emplace_back(bbox);
    }
  }
}

base::Status YoloMultiConvOutputPostParam::serialize(
    rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
  json.AddMember("version_", version_, allocator);
  json.AddMember("score_threshold_", score_threshold_, allocator);
  json.AddMember("nms_threshold_", nms_threshold_, allocator);
  json.AddMember("obj_threshold_", obj_threshold_, allocator);
  json.AddMember("num_classes_", num_classes_, allocator);
  json.AddMember("model_h_", model_h_, allocator);
  json.AddMember("model_w_", model_w_, allocator);

  rapidjson::Value anchors_array(rapidjson::kArrayType);
  for (int i = 0; i < 3; i++) {
    rapidjson::Value anchor_group(rapidjson::kArrayType);
    for (int j = 0; j < 6; j++) {
      anchor_group.PushBack(anchors_[i][j], allocator);
    }
    anchors_array.PushBack(anchor_group, allocator);
  }
  json.AddMember("anchors", anchors_array, allocator);

  rapidjson::Value strides_array(rapidjson::kArrayType);
  for (int i = 0; i < 3; i++) {
    strides_array.PushBack(strides_[i], allocator);
  }
  json.AddMember("strides", strides_array, allocator);

  return base::kStatusCodeOk;
}

base::Status YoloMultiConvOutputPostParam::deserialize(rapidjson::Value &json) {
  if (!json.HasMember("version_") || !json["version_"].IsInt()) {
    return base::kStatusCodeErrorInvalidValue;
  }
  version_ = json["version_"].GetInt();

  if (!json.HasMember("score_threshold_") || !json["score_threshold_"].IsFloat()) {
    return base::kStatusCodeErrorInvalidValue;
  }
  score_threshold_ = json["score_threshold_"].GetFloat();

  if (!json.HasMember("nms_threshold_") || !json["nms_threshold_"].IsFloat()) {
    return base::kStatusCodeErrorInvalidValue;
  }
  nms_threshold_ = json["nms_threshold_"].GetFloat();

  if (!json.HasMember("obj_threshold_") || !json["obj_threshold_"].IsFloat()) {
    return base::kStatusCodeErrorInvalidValue;
  }
  obj_threshold_ = json["obj_threshold_"].GetFloat();

  if (!json.HasMember("num_classes_") || !json["num_classes_"].IsInt()) {
    return base::kStatusCodeErrorInvalidValue;
  }
  num_classes_ = json["num_classes_"].GetInt();

  if (!json.HasMember("model_h_") || !json["model_h_"].IsInt()) {
    return base::kStatusCodeErrorInvalidValue;
  }
  model_h_ = json["model_h_"].GetInt();

  if (!json.HasMember("model_w_") || !json["model_w_"].IsInt()) {
    return base::kStatusCodeErrorInvalidValue;
  }
  model_w_ = json["model_w_"].GetInt();

  if (!json.HasMember("anchors") || !json["anchors"].IsArray() || 
      json["anchors"].Size() != 3) {
    return base::kStatusCodeErrorInvalidValue;
  }
  for (int i = 0; i < 3; i++) {
    if (!json["anchors"][i].IsArray() || json["anchors"][i].Size() != 6) {
      return base::kStatusCodeErrorInvalidValue;
    }
    for (int j = 0; j < 6; j++) {
      if (!json["anchors"][i][j].IsInt()) {
        return base::kStatusCodeErrorInvalidValue;
      }
      const_cast<int(&)[3][6]>(anchors_)[i][j] = json["anchors"][i][j].GetInt();
    }
  }

  if (!json.HasMember("strides") || !json["strides"].IsArray() || 
      json["strides"].Size() != 3) {
    return base::kStatusCodeErrorInvalidValue;
  }
  for (int i = 0; i < 3; i++) {
    if (!json["strides"][i].IsInt()) {
      return base::kStatusCodeErrorInvalidValue;
    }
    const_cast<int(&)[3]>(strides_)[i] = json["strides"][i].GetInt();
  }

  return base::kStatusCodeOk;
}

base::Status YoloMultiConvOutputPostProcess::run() {
  YoloMultiConvOutputPostParam *param =
      (YoloMultiConvOutputPostParam *)param_.get();
  DetectResult *results = new DetectResult();
  DetectResult results_batch;

  cv::Mat *src = inputs_[0]->getCvMat(this);
  int h = param->model_h_;
  int w = param->model_w_;

  int origin_h = src->rows;
  int origin_w = src->cols;
  float scale_h = (float)h / origin_h;
  float scale_w = (float)w / origin_w;
  float scale = std::min(scale_h, scale_w);

  float i2d[6];
  float d2i[6];

  i2d[0] = scale;
  i2d[1] = 0;
  i2d[2] = (-scale * origin_w + w + scale - 1) * 0.5;
  i2d[3] = 0;
  i2d[4] = scale;
  i2d[5] = (-scale * origin_h + h + scale - 1) * 0.5;

  cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);
  cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);
  cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);

  for (int stride = 1; stride < 4; stride++) {
    device::Tensor *tensor_stride = inputs_[stride]->getTensor(this);
    generateProposals(param->anchors_[stride - 1], param->strides_, stride - 1,
                      param->model_w_, param->model_h_, param->det_len_,
                      tensor_stride, param->score_threshold_, &results_batch);
  }
  std::vector<int> keep_idxs(results_batch.bboxs_.size());
  fastNMS(results_batch, keep_idxs, param->nms_threshold_);
  for (auto i = 0; i < keep_idxs.size(); ++i) {
    auto n = keep_idxs[i];
    if (n < 0) {
      continue;
    }

    results_batch.bboxs_[n].bbox_[0] =
        results_batch.bboxs_[n].bbox_[0] * d2i[0] + d2i[2];
    results_batch.bboxs_[n].bbox_[1] =
        results_batch.bboxs_[n].bbox_[1] * d2i[0] + d2i[5];
    results_batch.bboxs_[n].bbox_[2] =
        results_batch.bboxs_[n].bbox_[2] * d2i[0] + d2i[2];
    results_batch.bboxs_[n].bbox_[3] =
        results_batch.bboxs_[n].bbox_[3] * d2i[0] + d2i[5];

    results->bboxs_.emplace_back(results_batch.bboxs_[n]);
  }
  outputs_[0]->set(results, false);
  return base::kStatusCodeOk;
}



// dag::Graph *createYoloV5MultiConvOutputGraph(
//     const std::string &name, base::InferenceType inference_type,
//     base::DeviceType device_type, dag::Edge *input, dag::Edge *output,
//     base::ModelType model_type, bool is_path,
//     std::vector<std::string> model_value) {
//   dag::Graph *graph = new dag::Graph(name, {input}, {output});
//   dag::Edge *infer_input = graph->createEdge("images");
//   dag::Edge *edge_stride_8 = graph->createEdge("output0");   // [1, 80, 80,
//   255] dag::Edge *edge_stride_16 = graph->createEdge("output1");  // [1, 40,
//   40, 255] dag::Edge *edge_stride_32 = graph->createEdge("output2");  // [1,
//   20, 20, 255]

//   dag::Node *pre = graph->createNode<preprocess::WarpAffineCvtNormTrans>(
//       "preprocess", {input}, {infer_input});

//   infer::Infer *infer =
//       dynamic_cast<infer::Infer *>(graph->createNode<infer::Infer>(
//           "infer", {infer_input},
//           {edge_stride_8, edge_stride_16, edge_stride_32}));
//   infer->setInferenceType(inference_type);

//   dag::Node *post = graph->createNode<YoloMultiConvOutputPostProcess>(
//       "postprocess", {input, edge_stride_8, edge_stride_16, edge_stride_32},
//       {output});

//   preprocess::WarpAffineCvtNormTransParam *pre_param =
//       dynamic_cast<preprocess::WarpAffineCvtNormTransParam *>(pre->getParam());
//   pre_param->src_pixel_type_ = base::kPixelTypeBGR;
//   pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
//   pre_param->interp_type_ = base::kInterpTypeLinear;
//   pre_param->data_format_ = base::kDataFormatNHWC;
//   pre_param->h_ = 640;
//   pre_param->w_ = 640;
//   pre_param->scale_[1] = 1.0f;
//   pre_param->scale_[2] = 1.0f;
//   pre_param->scale_[3] = 1.0f;
//   pre_param->mean_[1] = 0.0f;
//   pre_param->mean_[2] = 0.0f;
//   pre_param->mean_[3] = 0.0f;
//   pre_param->std_[1] = 255.0f;
//   pre_param->std_[2] = 255.0f;
//   pre_param->std_[3] = 255.0f;
//   pre_param->const_value_ = 114;

//   inference::InferenceParam *inference_param =
//       (inference::InferenceParam *)(infer->getParam());
//   inference_param->is_path_ = is_path;
//   inference_param->model_value_ = model_value;
//   inference_param->device_type_ = device_type;

//   // inference_param->runtime_ = "dsp";
//   base::Any runtime = "dsp";
//   inference_param->set("runtime", runtime);
//   // inference_param->perf_mode_ = 5;
//   base::Any perf_mode = 5;
//   inference_param->set("perf_mode", perf_mode);
//   // inference_param->profiling_level_ = 0;
//   base::Any profiling_level = 0;
//   inference_param->set("profiling_level", profiling_level);
//   // inference_param->buffer_type_ = 0;
//   base::Any buffer_type = 0;
//   inference_param->set("buffer_type", buffer_type);
//   // inference_param->input_names_ = {"images"};
//   base::Any input_names = std::vector<std::string>{"images"};
//   inference_param->set("input_names", input_names);
//   // inference_param->output_tensor_names_ = {"output0", "output1",
//   "output2"}; base::Any output_tensor_names =
//       std::vector<std::string>{"output0", "output1", "output2"};
//   inference_param->set("output_tensor_names", output_tensor_names);
//   // inference_param->output_layer_names_ = {"Conv_199", "Conv_200",
//   // "Conv_201"};
//   base::Any output_layer_names =
//       std::vector<std::string>{"Conv_199", "Conv_200", "Conv_201"};
//   inference_param->set("output_layer_names", output_layer_names);

//   // TODO: 很多信息可以从 preprocess 和 infer 中获取
//   YoloMultiConvOutputPostParam *post_param =
//       dynamic_cast<YoloMultiConvOutputPostParam *>(post->getParam());
//   post_param->score_threshold_ = 0.5;
//   post_param->nms_threshold_ = 0.5;
//   post_param->num_classes_ = 80;
//   post_param->model_h_ = 640;
//   post_param->model_w_ = 640;
//   post_param->version_ = 5;

//   return graph;
// }

REGISTER_NODE("nndeploy::detect::YoloMultiConvOutputPostProcess",
              YoloMultiConvOutputPostProcess);
REGISTER_NODE("nndeploy::detect::YoloMultiConvOutputGraph",
              YoloMultiConvOutputGraph);

}  // namespace detect
}  // namespace nndeploy