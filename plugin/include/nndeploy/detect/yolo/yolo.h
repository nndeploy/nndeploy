
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
#include "nndeploy/preprocess/cvt_resize_norm_trans.h"

namespace nndeploy {
namespace detect {

/**
 * @brief YOLO后处理参数类
 *
 * 该类用于定义YOLO模型后处理阶段所需的参数。包括分数阈值、非最大抑制(NMS)阈值、类别数量以及模型输入图像的尺寸。
 */
class NNDEPLOY_CC_API YoloPostParam : public base::Param {
 public:
  float score_threshold_;  // 分数阈值，用于决定哪些检测框被保留
  float nms_threshold_;    // 非最大抑制(NMS)阈值，用于合并重叠的检测框
  int num_classes_;        // 模型可以识别的类别数量
  int model_h_;            // 模型输入图像的高度
  int model_w_;            // 模型输入图像的宽度

  int version_ = -1;  // YOLO模型的版本号，默认为-1表示未指定

  using base::Param::serialize;
  virtual base::Status serialize(rapidjson::Value &json,
                                 rapidjson::Document::AllocatorType &allocator);
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value &json);
};

class NNDEPLOY_CC_API YoloPostProcess : public dag::Node {
 public:
  YoloPostProcess(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::detect::YoloPostProcess";
    param_ = std::make_shared<YoloPostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<DetectResult>();
  }
  YoloPostProcess(const std::string &name, std::vector<dag::Edge *> inputs,
                  std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::detect::YoloPostProcess";
    param_ = std::make_shared<YoloPostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<DetectResult>();
  }
  virtual ~YoloPostProcess() {}

  virtual base::Status run();

  base::Status runV5V6();
  base::Status runV8V11();
  base::Status runX();
};

class NNDEPLOY_CC_API YoloGraph : public dag::Graph {
 public:
  YoloGraph(const std::string &name) : dag::Graph(name) {
    key_ = "nndeploy::detect::YoloGraph";
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<DetectResult>();
    pre_ = dynamic_cast<preprocess::CvtResizeNormTrans *>(
        this->createNode<preprocess::CvtResizeNormTrans>("preprocess"));
    infer_ =
        dynamic_cast<infer::Infer *>(this->createNode<infer::Infer>("infer"));
    post_ = dynamic_cast<YoloPostProcess *>(
        this->createNode<YoloPostProcess>("postprocess"));
  }

  YoloGraph(const std::string &name, std::vector<dag::Edge *> inputs,
            std::vector<dag::Edge *> outputs)
      : dag::Graph(name, inputs, outputs) {
    key_ = "nndeploy::detect::YoloGraph";
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<DetectResult>();
    pre_ = dynamic_cast<preprocess::CvtResizeNormTrans *>(
        this->createNode<preprocess::CvtResizeNormTrans>("preprocess"));
    infer_ =
        dynamic_cast<infer::Infer *>(this->createNode<infer::Infer>("infer"));
    post_ = dynamic_cast<YoloPostProcess *>(
        this->createNode<YoloPostProcess>("postprocess"));
  }

  virtual ~YoloGraph() {}

  virtual base::Status defaultParam() {
    preprocess::CvtResizeNormTransParam *pre_param =
        dynamic_cast<preprocess::CvtResizeNormTransParam *>(pre_->getParam());
    pre_param->src_pixel_type_ = base::kPixelTypeBGR;
    pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
    pre_param->interp_type_ = base::kInterpTypeLinear;
    pre_param->h_ = 640;
    pre_param->w_ = 640;

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
  base::Status make(const dag::NodeDesc &pre_desc,
                    const dag::NodeDesc &infer_desc,
                    base::InferenceType inference_type,
                    const dag::NodeDesc &post_desc) {
    this->setNodeDesc(pre_, pre_desc);
    this->setNodeDesc(infer_, infer_desc);
    this->setNodeDesc(post_, post_desc);
    this->defaultParam();
    base::Status status = infer_->setInferenceType(inference_type);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Failed to set inference type");
      return status;
    }
    return base::kStatusCodeOk;
  }

  base::Status setInferenceType(base::InferenceType inference_type) {
    base::Status status = infer_->setInferenceType(inference_type);
    if (status != base::kStatusCodeOk) {
      NNDEPLOY_LOGE("Failed to set inference type");
      return status;
    }
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

  base::Status setSrcPixelType(base::PixelType pixel_type) {
    preprocess::CvtResizeNormTransParam *param =
        dynamic_cast<preprocess::CvtResizeNormTransParam *>(pre_->getParam());
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

#endif /* _NNDEPLOY_DETECT_DETECT_YOLO_YOLO_H_ */