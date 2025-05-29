#ifndef _NNDEPLOY_DETECT_DETECT_YOLO_YOLOX_H_
#define _NNDEPLOY_DETECT_DETECT_YOLO_YOLOX_H_

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

namespace nndeploy {
namespace detect {

/**
 * @brief YOLOX后处理参数类
 *
 * 该类用于定义YOLOX模型后处理阶段所需的参数。包括分数阈值、非最大抑制(NMS)阈值、类别数量以及模型输入图像的尺寸。
 */
class NNDEPLOY_CC_API YoloXPostParam : public base::Param {
 public:
  float score_threshold_;  // 分数阈值，用于决定哪些检测框被保留
  float nms_threshold_;  // 非最大抑制(NMS)阈值，用于合并重叠的检测框
  int num_classes_;  // 模型可以识别的类别数量
  int model_h_;      // 模型输入图像的高度
  int model_w_;      // 模型输入图像的宽度
};

class NNDEPLOY_CC_API YoloXPostProcess : public dag::Node {
 public:
  YoloXPostProcess(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::detect::YoloXPostProcess";
    param_ = std::make_shared<YoloXPostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<DetectResult>();
  }

  YoloXPostProcess(const std::string &name,
                   const std::vector<dag::Edge *> &inputs,
                   const std::vector<dag::Edge *> &outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::detect::YoloXPostProcess";
    param_ = std::make_shared<YoloXPostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<DetectResult>();
  }

  virtual ~YoloXPostProcess() {}

  virtual base::Status run();
};

class NNDEPLOY_CC_API YoloXGraph : public dag::Graph {
 public:
  YoloXGraph(const std::string &name) : dag::Graph(name) {
    key_ = "nndeploy::detect::YoloXGraph";
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<DetectResult>();
  }

  YoloXGraph(const std::string &name, const std::vector<dag::Edge *> &inputs,
             const std::vector<dag::Edge *> &outputs)
      : dag::Graph(name, inputs, outputs) {
    key_ = "nndeploy::detect::YoloXGraph";
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<DetectResult>();
  }

  virtual ~YoloXGraph(){};

  base::Status make(const dag::NodeDesc &pre_desc,
                    const dag::NodeDesc &infer_desc,
                    base::InferenceType inference_type,
                    const dag::NodeDesc &post_desc) {
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
    pre_param->normalize_ = false;

    infer_ = dynamic_cast<infer::Infer *>(
        this->createNode<infer::Infer>(infer_desc));
    if (infer_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create inference node");
      return base::kStatusCodeErrorInvalidParam;
    }
    infer_->setInferenceType(inference_type);

    post_ = this->createNode<YoloXPostProcess>(post_desc);
    if (post_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create postprocessing node");
      return base::kStatusCodeErrorInvalidParam;
    }
    YoloXPostParam *post_param =
        dynamic_cast<YoloXPostParam *>(post_->getParam());
    post_param->score_threshold_ = 0.5;
    post_param->nms_threshold_ = 0.45;
    post_param->num_classes_ = 80;
    post_param->model_h_ = 640;
    post_param->model_w_ = 640;

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

  std::vector<dag::Edge *> forward(std::vector<dag::Edge *> inputs) {
    std::vector<dag::Edge *> pre_outputs = (*pre_)(inputs);
    std::vector<dag::Edge *> infer_outputs = (*infer_)(pre_outputs);
    std::vector<dag::Edge *> post_outputs = (*post_)(infer_outputs);
    return post_outputs;
  }

 private:
  dag::Node *pre_ = nullptr;       ///< Preprocessing node pointer
  infer::Infer *infer_ = nullptr;  ///< Inference node pointer
  dag::Node *post_ = nullptr;      ///< Postprocessing node pointer
};

}  // namespace detect
}  // namespace nndeploy

#endif  // _NNDEPLOY_DETECT_DETECT_YOLO_YOLOX_H_