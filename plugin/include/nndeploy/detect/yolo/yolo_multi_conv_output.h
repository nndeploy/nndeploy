#ifndef _NNDEPLOY_DETECT_DETECT_YOLO_YOLO_MULTI_CONV_OUTPUT_H_
#define _NNDEPLOY_DETECT_DETECT_YOLO_YOLO_MULTI_CONV_OUTPUT_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/detect/result.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/infer/infer.h"
#include "nndeploy/preprocess/cvtcolor_resize.h"
#include "nndeploy/preprocess/cvtcolor_resize_pad.h"
#include "nndeploy/preprocess/warpaffine_preprocess.h"

namespace nndeploy {
namespace detect {

static float i2d[6];
static float d2i[6];

class NNDEPLOY_CC_API YoloMultiConvOutputPostParam : public base::Param {
 public:
  float score_threshold_;
  float nms_threshold_;
  float obj_threshold_;
  int num_classes_;
  int model_h_;
  int model_w_;

  int det_obj_len_ = 1;
  int det_bbox_len_ = 4;
  int det_cls_len_ = 80;
  int det_len_ = (det_cls_len_ + det_bbox_len_ + det_obj_len_) * 3;

  const int anchors_[3][6] = {{10, 13, 16, 30, 33, 23},
                              {30, 61, 62, 45, 59, 119},
                              {116, 90, 156, 198, 373, 326}};

  const int strides_[3] = {8, 16, 32};

  int version_ = -1;
};

class NNDEPLOY_CC_API YoloMultiConvOutputPostProcess : public dag::Node {
 public:
  YoloMultiConvOutputPostProcess(const std::string &name) : dag::Node(name) {
    param_ = std::make_shared<YoloMultiConvOutputPostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<DetectResult>();
  }
  YoloMultiConvOutputPostProcess(const std::string &name,
                                 std::vector<dag::Edge *> inputs,
                                 std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    param_ = std::make_shared<YoloMultiConvOutputPostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<DetectResult>();
  }
  virtual ~YoloMultiConvOutputPostProcess() {}

  virtual base::Status run();
};

class NNDEPLOY_CC_API YoloMultiConvOutputGraph : public dag::Graph {
 public:
  YoloMultiConvOutputGraph(const std::string &name) : dag::Graph(name) {
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<DetectResult>();
  }
  YoloMultiConvOutputGraph(const std::string &name,
                           std::vector<dag::Edge *> inputs,
                           std::vector<dag::Edge *> outputs)
      : dag::Graph(name, inputs, outputs) {
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<DetectResult>();
  }

  virtual ~YoloMultiConvOutputGraph() {}

  base::Status make(const dag::NodeDesc &pre_desc,
                    const dag::NodeDesc &infer_desc,
                    base::InferenceType inference_type,
                    const dag::NodeDesc &post_desc) {
    // Create preprocessing node for image preprocessing
    pre_ = this->createNode<preprocess::WarpaffinePreprocess>(pre_desc);
    if (pre_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create preprocessing node");
      return base::kStatusCodeErrorInvalidParam;
    }
    preprocess::WarpAffineParam *pre_param =
        dynamic_cast<preprocess::WarpAffineParam *>(pre_->getParam());
    pre_param->src_pixel_type_ = base::kPixelTypeBGR;
    pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
    pre_param->interp_type_ = base::kInterpTypeLinear;
    pre_param->data_format_ = base::kDataFormatNHWC;
    pre_param->h_ = 640;
    pre_param->w_ = 640;
    pre_param->scale_[1] = 1.0f;
    pre_param->scale_[2] = 1.0f;
    pre_param->scale_[3] = 1.0f;
    pre_param->mean_[1] = 0.0f;
    pre_param->mean_[2] = 0.0f;
    pre_param->mean_[3] = 0.0f;
    pre_param->std_[1] = 255.0f;
    pre_param->std_[2] = 255.0f;
    pre_param->std_[3] = 255.0f;
    pre_param->const_value_ = 114;

    // Create inference node for ResNet model execution
    infer_ = dynamic_cast<infer::Infer *>(
        this->createNode<infer::Infer>(infer_desc));
    if (infer_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create inference node");
      return base::kStatusCodeErrorInvalidParam;
    }
    infer_->setInferenceType(inference_type);
    inference::InferenceParam *inference_param =
        (inference::InferenceParam *)(infer_->getParam());
    // inference_param->runtime_ = "dsp";
    base::Any runtime = "dsp";
    inference_param->set("runtime", runtime);
    // inference_param->perf_mode_ = 5;
    base::Any perf_mode = 5;
    inference_param->set("perf_mode", perf_mode);
    // inference_param->profiling_level_ = 0;
    base::Any profiling_level = 0;
    inference_param->set("profiling_level", profiling_level);
    // inference_param->buffer_type_ = 0;
    base::Any buffer_type = 0;
    inference_param->set("buffer_type", buffer_type);
    // inference_param->input_names_ = {"images"};
    base::Any input_names = std::vector<std::string>{"images"};
    inference_param->set("input_names", input_names);
    // inference_param->output_tensor_names_ = {"output0", "output1",
    // "output2"};
    base::Any output_tensor_names =
        std::vector<std::string>{"output0", "output1", "output2"};
    inference_param->set("output_tensor_names", output_tensor_names);
    // inference_param->output_layer_names_ = {"Conv_199", "Conv_200",
    // "Conv_201"};
    base::Any output_layer_names =
        std::vector<std::string>{"Conv_199", "Conv_200", "Conv_201"};
    inference_param->set("output_layer_names", output_layer_names);

    // Create postprocessing node for classification results
    post_ = this->createNode<YoloMultiConvOutputPostProcess>(post_desc);
    if (post_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create postprocessing node");
      return base::kStatusCodeErrorInvalidParam;
    }
    YoloMultiConvOutputPostParam *post_param =
        dynamic_cast<YoloMultiConvOutputPostParam *>(post_->getParam());
    post_param->score_threshold_ = 0.5;
    post_param->nms_threshold_ = 0.5;
    post_param->num_classes_ = 80;
    post_param->model_h_ = 640;
    post_param->model_w_ = 640;
    post_param->version_ = 5;

    return base::kStatusCodeOk;
  }

  base::Status setInferParam(base::DeviceType device_type,
                             base::ModelType model_type, bool is_path,
                             std::vector<std::string> &model_value) {
    auto param = dynamic_cast<inference::InferenceParam *>(infer_->getParam());
    param->device_type_ = device_type;
    param->model_type_ = model_type;
    param->is_path_ = is_path;
    param->model_value_ = model_value;
    return base::kStatusCodeOk;
  }

  /**
   * @brief Set preprocessing parameters
   * @param pixel_type Input image pixel format (e.g. RGB, BGR)
   * @return kStatusCodeOk on success
   */
  base::Status setSrcPixelType(base::PixelType pixel_type) {
    preprocess::CvtclorResizeParam *param =
        dynamic_cast<preprocess::CvtclorResizeParam *>(pre_->getParam());
    param->src_pixel_type_ = pixel_type;
    return base::kStatusCodeOk;
  }

  base::Status setScoreThreshold(float score_threshold) {
    YoloMultiConvOutputPostParam *param =
        dynamic_cast<YoloMultiConvOutputPostParam *>(post_->getParam());
    param->score_threshold_ = score_threshold;
    return base::kStatusCodeOk;
  }

  base::Status setNmsThreshold(float nms_threshold) {
    YoloMultiConvOutputPostParam *param =
        dynamic_cast<YoloMultiConvOutputPostParam *>(post_->getParam());
    param->nms_threshold_ = nms_threshold;
    return base::kStatusCodeOk;
  }

  base::Status setNumClasses(int num_classes) {
    YoloMultiConvOutputPostParam *param =
        dynamic_cast<YoloMultiConvOutputPostParam *>(post_->getParam());
    param->num_classes_ = num_classes;
    return base::kStatusCodeOk;
  }

  base::Status setModelHW(int model_h, int model_w) {
    YoloMultiConvOutputPostParam *param =
        dynamic_cast<YoloMultiConvOutputPostParam *>(post_->getParam());
    param->model_h_ = model_h;
    param->model_w_ = model_w;
    return base::kStatusCodeOk;
  }

  base::Status setVersion(int version) {
    YoloMultiConvOutputPostParam *param =
        dynamic_cast<YoloMultiConvOutputPostParam *>(post_->getParam());
    param->version_ = version;
    return base::kStatusCodeOk;
  }

 private:
  dag::Node *pre_ = nullptr;       ///< Preprocessing node pointer
  infer::Infer *infer_ = nullptr;  ///< Inference node pointer
  dag::Node *post_ = nullptr;      ///< Postprocessing node pointer
};

}  // namespace detect
}  // namespace nndeploy

#endif