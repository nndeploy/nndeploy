#include "nndeploy/model/detect/opencv/detr/detr.h"

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
#include "nndeploy/pipeline/packet.h"
#include "nndeploy/pipeline/preprocess/opencv/cvtcolor_resize.h"
#include "nndeploy/pipeline/task.h"

namespace nndeploy {
namespace model {
namespace opencv {

// static TypeTaskRegister g_internal_opencv_detr_task_register("opencv_detr",
//                                                              creatDetrTask);

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

base::Status DetrPostProcess::run() {
  results_.result_.clear();

  DetrPostParam* temp_param = (DetrPostParam*)param_.get();
  float score_threshold = temp_param->score_threshold_;

  device::Tensor* tensor_logits = nullptr;
  device::Tensor* tensor_boxes = nullptr;
  for (int i = 0; i < inputs_[0]->sizeTensor(); ++i) {
    device::Tensor* tensor = inputs_[0]->getTensor(i);
    if (tensor->getName() == "pred_logits") {
      tensor_logits = tensor;
    } else if (tensor->getName() == "pred_boxes") {
      tensor_boxes = tensor;
    }
  }
  float* logits = (float*)tensor_logits->getPtr();
  float* boxes = (float*)tensor_boxes->getPtr();

  int num_qurrey = tensor_logits->getShape()[1];
  int num_class = tensor_logits->getShape()[2];

  for (int i = 0; i < num_qurrey; i++) {
    std::vector<float> scores(num_class);
    softmax(logits + i * num_class, scores.data(), num_class);

    auto maxPosition = std::max_element(scores.begin(), scores.end() - 1);
    if (*maxPosition < score_threshold) {
      continue;
    } else {
      DetectResult result;
      result.score_ = *maxPosition;
      result.label_id_ = maxPosition - scores.begin();

      float cx = boxes[i * 4];
      float cy = boxes[i * 4 + 1];
      float cw = boxes[i * 4 + 2];
      float ch = boxes[i * 4 + 3];

      result.bbox_[0] = (cx - 0.5 * cw);
      result.bbox_[1] = (cy - 0.5 * ch);
      result.bbox_[2] = (cx + 0.5 * cw);
      result.bbox_[3] = (cy + 0.5 * ch);

      results_.result_.emplace_back(result);
    }
  }
  outputs_[0]->set(results_);

  return base::kStatusCodeOk;
}

pipeline::Pipeline* creatDetrPipeline(const std::string& name,
                                      base::InferenceType type,
                                      pipeline::Packet* input,
                                      pipeline::Packet* output, bool is_path,
                                      std::vector<std::string>& model_value) {
  pipeline::Pipeline* pipeline = new pipeline::Pipeline(name, input, output);
  pipeline::Packet* infer_input = pipeline->createPacket();
  pipeline::Packet* infer_output = pipeline->createPacket();

  pipeline::Task* pre =
      pipeline->createTask<pipeline::opencv::CvtColrResize>("", input, infer_input);
  pipeline::Task* infer = pipeline->createInfer<pipeline::Infer>(
      "", type, infer_input, infer_output);
  pipeline::Task* post =
      pipeline->createTask<DetrPostProcess>("", infer_output, output);

  pipeline::CvtclorResizeParam* pre_param =
      dynamic_cast<pipeline::CvtclorResizeParam*>(pre->getParam());
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
  inference_param->device_type_ = device::getDefaultHostDeviceType();

  return pipeline;
}

}  // namespace opencv
}  // namespace model
}  // namespace nndeploy
