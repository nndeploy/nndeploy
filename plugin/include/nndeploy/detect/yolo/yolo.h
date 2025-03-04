
#ifndef _NNDEPLOY_DETECT_DETECT_YOLO_YOLO_H_
#define _NNDEPLOY_DETECT_DETECT_YOLO_YOLO_H_

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

namespace nndeploy {
namespace detect {

class NNDEPLOY_CC_API YoloPostParam : public base::Param {
 public:
  float score_threshold_;
  float nms_threshold_;
  int num_classes_;
  int model_h_;
  int model_w_;

  int version_ = -1;
};

class NNDEPLOY_CC_API YoloPostProcess : public dag::Node {
 public:
  YoloPostProcess(const std::string &name) : dag::Node(name) {
    param_ = std::make_shared<YoloPostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<DetectResult>();
  }
  YoloPostProcess(const std::string &name, std::vector<dag::Edge *> inputs,
                  std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    param_ = std::make_shared<YoloPostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<DetectResult>();
  }
  virtual ~YoloPostProcess() {}

  virtual base::Status run();

  base::Status runV5V6();
  base::Status runV8();
};

class NNDEPLOY_CC_API YoloGraph : public dag::Graph {
 public:
  YoloGraph(const std::string &name) : dag::Graph(name) {
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<DetectResult>();
  }

  YoloGraph(const std::string &name, std::vector<dag::Edge *> inputs,
            std::vector<dag::Edge *> outputs)
      : dag::Graph(name, inputs, outputs) {
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<DetectResult>();
  }

  virtual ~YoloGraph() {}

  base::Status make(const dag::NodeDesc &pre_desc,
                    const dag::NodeDesc &infer_desc,
                    base::InferenceType inference_type,
                    const dag::NodeDesc &post_desc) {
    // Create preprocessing node for image preprocessing
    pre_ = this->createNode<preprocess::CvtColorResize>(pre_desc);
    if (pre_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create preprocessing node");
      return base::kStatusCodeErrorInvalidParam;
    }
    preprocess::CvtclorResizeParam *pre_param =
        dynamic_cast<preprocess::CvtclorResizeParam *>(pre_->getParam());
    pre_param->src_pixel_type_ = base::kPixelTypeBGR;
    pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
    pre_param->interp_type_ = base::kInterpTypeLinear;
    pre_param->h_ = 640;
    pre_param->w_ = 640;

    // Create inference node for ResNet model execution
    infer_ = dynamic_cast<infer::Infer *>(
        this->createNode<infer::Infer>(infer_desc));
    if (infer_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create inference node");
      return base::kStatusCodeErrorInvalidParam;
    }
    infer_->setInferenceType(inference_type);

    // Create postprocessing node for classification results
    post_ = this->createNode<YoloPostProcess>(post_desc);
    if (post_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create postprocessing node");
      return base::kStatusCodeErrorInvalidParam;
    }
    YoloPostParam *post_param =
        dynamic_cast<YoloPostParam *>(post_->getParam());
    post_param->score_threshold_ = 0.5;
    post_param->nms_threshold_ = 0.45;
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
    YoloPostParam *param = dynamic_cast<YoloPostParam *>(post_->getParam());
    param->score_threshold_ = score_threshold;
    return base::kStatusCodeOk;
  }

  base::Status setNmsThreshold(float nms_threshold) {
    YoloPostParam *param = dynamic_cast<YoloPostParam *>(post_->getParam());
    param->nms_threshold_ = nms_threshold;
    return base::kStatusCodeOk;
  }

  base::Status setNumClasses(int num_classes) {
    YoloPostParam *param = dynamic_cast<YoloPostParam *>(post_->getParam());
    param->num_classes_ = num_classes;
    return base::kStatusCodeOk;
  }

  base::Status setModelHW(int model_h, int model_w) {
    YoloPostParam *param = dynamic_cast<YoloPostParam *>(post_->getParam());
    param->model_h_ = model_h;
    param->model_w_ = model_w;
    return base::kStatusCodeOk;
  }

  base::Status setVersion(int version) {
    YoloPostParam *param = dynamic_cast<YoloPostParam *>(post_->getParam());
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

#endif /* _NNDEPLOY_DETECT_DETECT_YOLO_YOLO_H_ */