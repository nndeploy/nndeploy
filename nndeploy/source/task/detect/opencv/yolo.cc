#include "nndeploy/source/task/detect/opencv/yolo.h"

#include "nndeploy/source/base/basic.h"
#include "nndeploy/source/base/glic_stl_include.h"
#include "nndeploy/source/base/log.h"
#include "nndeploy/source/base/macro.h"
#include "nndeploy/source/base/object.h"
#include "nndeploy/source/base/status.h"
#include "nndeploy/source/base/string.h"
#include "nndeploy/source/base/value.h"
#include "nndeploy/source/device/buffer.h"
#include "nndeploy/source/device/buffer_pool.h"
#include "nndeploy/source/device/device.h"
#include "nndeploy/source/device/tensor.h"
#include "nndeploy/source/inference/inference_param.h"
#include "nndeploy/source/task/execution.h"
#include "nndeploy/source/task/opencv_include.h"
#include "nndeploy/source/task/packet.h"
#include "nndeploy/source/task/pre_process/opencv/cvtcolor_resize.h"
#include "nndeploy/source/task/task.h"

namespace nndeploy {
namespace task {

static TypeTaskRegister g_internal_task_register("opencv_yolov5",
                                                 creatYoloTask);

static inline float sigmoid(float x) {
  return static_cast<float>(1.f / (1.f + std::exp(-x)));
}

base::Status YoloPostProcess::run() {
  device::Tensor* tensor_scores = nullptr;
  device::Tensor* tensor_boxes = nullptr;
  device::Tensor* tensor_anchors = nullptr;

  for (int i = 0; i < input_->getTensorSize(); ++i) {
    device::Tensor* tensor = input_->getTensor(i);
    if (tensor->getName() == "output") {
      tensor_scores = tensor;
    } else if (tensor->getName() == "417") {
      tensor_boxes = tensor;
    } else if (tensor->getName() == "437") {
      tensor_anchors = tensor;
    }
  }

  DetectResult* result = (DetectResult*)output_->getParam();

  decodeOutputTensor(tensor_scores, stride_scores_, size_scores_, *result);
  decodeOutputTensor(tensor_boxes, stride_boxes_, size_boxes_, *result);
  decodeOutputTensor(tensor_anchors, stride_anchors_, size_anchors_, *result);

  nms(*result, nms_threshold_);

  return base::kStatusCodeOk;
}

void YoloPostProcess::decodeOutputTensor(device::Tensor* data, int stride,
                                         std::vector<cv::Size>& anchors,
                                         DetectResult& result) {
  int batchs = data->getShapeIndex(0);
  int channels = data->getShapeIndex(1);
  int height = data->getShapeIndex(2);
  int width = data->getShapeIndex(3);
  int pred_item = data->getShapeIndex(4);

  float* data_ptr = (float*)data->getPtr();
  for (int bi = 0; bi < batchs; bi++) {
    auto batch_ptr = data_ptr + bi * (channels * height * width * pred_item);
    for (int ci = 0; ci < channels; ci++) {
      auto channel_ptr = batch_ptr + ci * (height * width * pred_item);
      for (int hi = 0; hi < height; hi++) {
        auto height_ptr = channel_ptr + hi * (width * pred_item);
        for (int wi = 0; wi < width; wi++) {
          auto width_ptr = height_ptr + wi * pred_item;
          auto cls_ptr = width_ptr + 5;

          auto confidence = sigmoid(width_ptr[4]);

          for (int cls_id = 0; cls_id < num_classes_; cls_id++) {
            float score = sigmoid(cls_ptr[cls_id]) * confidence;
            if (score > threshold_) {
              float cx =
                  (sigmoid(width_ptr[0]) * 2.f - 0.5f + wi) * (float)stride;
              float cy =
                  (sigmoid(width_ptr[1]) * 2.f - 0.5f + hi) * (float)stride;
              float w =
                  std::pow(sigmoid(width_ptr[2]) * 2.f, 2) * anchors[ci].width;
              float h =
                  std::pow(sigmoid(width_ptr[3]) * 2.f, 2) * anchors[ci].height;

              std::array<float, 4> box;

              box[0] = std::max(0, std::min(max_width_, int((cx - w / 2.f))));
              box[1] = std::max(0, std::min(max_height_, int((cy - h / 2.f))));
              box[2] = std::max(0, std::min(max_width_, int((cx + w / 2.f))));
              box[3] = std::max(0, std::min(max_height_, int((cy + h / 2.f))));

              result.boxes_.push_back(box);
              result.scores_.push_back(score);
              result.label_ids_.push_back(cls_id);
            }
          }
        }
      }
    }
  }

  return;
}

void nms(DetectResult& result, float NMS_THRESH) {
  std::vector<float> vArea(result.boxes_.size());
  for (int i = 0; i < int(result.boxes_.size()); ++i) {
    vArea[i] = (result.boxes_[i][2] - result.boxes_[i][0] + 1) *
               (result.boxes_[i][3] - result.boxes_[i][1] + 1);
  }
  for (int i = 0; i < int(result.boxes_.size()); ++i) {
    for (int j = i + 1; j < int(result.boxes_.size());) {
      float xx1 = std::max(result.boxes_[i][0], result.boxes_[i][0]);
      float yy1 = std::max(result.boxes_[i][1], result.boxes_[i][1]);
      float xx2 = std::min(result.boxes_[i][2], result.boxes_[i][2]);
      float yy2 = std::min(result.boxes_[i][3], result.boxes_[i][3]);
      float w = std::max(float(0), xx2 - xx1 + 1);
      float h = std::max(float(0), yy2 - yy1 + 1);
      float inter = w * h;
      float ovr = inter / (vArea[i] + vArea[j] - inter);
      if (ovr >= NMS_THRESH) {
        result.boxes_.erase(result.boxes_.begin() + j);
        result.scores_.erase(result.scores_.begin() + j);
        result.label_ids_.erase(result.label_ids_.begin() + j);
        vArea.erase(vArea.begin() + j);
      } else {
        j++;
      }
    }
  }
}

task::Task* creatYoloTask(base::InferenceType type,
                          base::DeviceType device_type, const std::string& name,
                          bool model_is_path,
                          std::vector<std::string> model_value) {
  task::Task* task = new task::Task(base::kInferenceTypeMnn, device_type, name);
  task->createPreprocess<task::OpencvCvtColrResize>();
  task->createPostprocess<task::YoloPostProcess>();

  CvtclorResizeParam* pre_param =
      dynamic_cast<CvtclorResizeParam*>(task->getPreProcessParam());
  pre_param->src_pixel_type_ = base::kPixelTypeBGR;
  pre_param->dst_pixel_type_ = base::kPixelTypeBGR;
  pre_param->interp_type_ = base::kInterpTypeLinear;
  pre_param->mean_[0] = 0.0f;
  pre_param->mean_[1] = 0.0f;
  pre_param->mean_[2] = 0.0f;
  pre_param->mean_[3] = 0.0f;
  pre_param->std_[0] = 255.0f;
  pre_param->std_[1] = 255.0f;
  pre_param->std_[2] = 255.0f;
  pre_param->std_[3] = 255.0f;
  inference::InferenceParam* inference_param =
      (inference::InferenceParam*)(task->getInferenceParam());
  inference_param->is_path_ = model_is_path;
  inference_param->model_value_ = model_value;

  return task;
}

}  // namespace task
}  // namespace nndeploy
