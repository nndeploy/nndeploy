#include "nndeploy/model/detect/yolo/yolo.h"

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/base/value.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/buffer_pool.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/model/detect/util.h"
#include "nndeploy/model/packet.h"
#include "nndeploy/model/preprocess/cvtcolor_resize.h"
#include "nndeploy/model/task.h"

namespace nndeploy {
namespace model {

class TypePipelineRegister g_register_yolov6_pipeline(NNDEPLOY_YOLOV6,
                                                      createYoloV6Pipeline);

template <typename T>
int softmax(const T* src, T* dst, int length) {
  T denominator{0};
  for (int i = 0; i < length; ++i) {
    dst[i] = std::exp(src[i]);
    denominator += dst[i];
  }
  for (int i = 0; i < length; ++i) {
    dst[i] /= denominator;
  }
  return 0;
}

base::Status YoloPostProcess::run() {
  YoloPostParam* param = (YoloPostParam*)param_.get();
  float score_threshold = param->score_threshold_;
  int num_classes = param->num_classes_;

  device::Tensor* tensor = inputs_[0]->getTensor();
  float* data = (float*)tensor->getPtr();
  int batch = tensor->getBatch();
  int height = tensor->getHeight();
  int width = tensor->getWidth();

  DetectResult* results = (DetectResult*)outputs_[0]->getParam();
  results->bboxs_.clear();

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
      float y0 = y_center - object_h * 0.5f;
      float x1 = x_center + object_w * 0.5f;
      float y1 = y_center + object_h * 0.5f;
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

  return base::kStatusCodeOk;
}

model::Pipeline* createYoloV6Pipeline(const std::string& name,
                                      base::InferenceType inference_type,
                                      base::DeviceType device_type,
                                      Packet* input, Packet* output,
                                      base::ModelType model_type, bool is_path,
                                      std::vector<std::string>& model_value) {
  model::Pipeline* pipeline = new model::Pipeline(name, input, output);
  model::Packet* infer_input = pipeline->createPacket("infer_input");
  model::Packet* infer_output = pipeline->createPacket("infer_output");

  model::Task* pre = pipeline->createTask<model::CvtColrResize>(
      "preprocess", input, infer_input);

  model::Task* infer = pipeline->createInfer<model::Infer>(
      "infer", inference_type, infer_input, infer_output);

  model::Task* post = pipeline->createTask<YoloPostProcess>(
      "postprocess", infer_output, output);

  model::CvtclorResizeParam* pre_param =
      dynamic_cast<model::CvtclorResizeParam*>(pre->getParam());
  pre_param->src_pixel_type_ = base::kPixelTypeBGR;
  pre_param->dst_pixel_type_ = base::kPixelTypeBGR;
  pre_param->interp_type_ = base::kInterpTypeLinear;
  pre_param->mean_[0] = 0.485f;
  pre_param->mean_[1] = 0.456f;
  pre_param->mean_[2] = 0.406f;
  pre_param->std_[0] = 0.229f;
  pre_param->std_[1] = 0.224f;
  pre_param->std_[2] = 0.225f;

  inference::InferenceParam* inference_param =
      (inference::InferenceParam*)(infer->getParam());
  inference_param->is_path_ = is_path;
  inference_param->model_value_ = model_value;
  inference_param->device_type_ = device_type;

  // TODO: 很多信息可以从 preprocess 和 infer 中获取
  YoloPostParam* post_param = dynamic_cast<YoloPostParam*>(post->getParam());
  post_param->score_threshold_ = 0.5;
  post_param->nms_threshold_ = 0.45;
  post_param->num_classes_ = 80;
  post_param->model_h_ = 640;
  post_param->model_w_ = 640;

  return pipeline;
}

}  // namespace model
}  // namespace nndeploy
