
#ifndef _NNDEPLOY_CLASSIFICATION_CLASSIFICATION_H_
#define _NNDEPLOY_CLASSIFICATION_CLASSIFICATION_H_

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
#include "nndeploy/classification/result.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/infer/infer.h"
#include "nndeploy/preprocess/cvtcolor_resize.h"
#include "nndeploy/preprocess/params.h"

namespace nndeploy {
namespace classification {

class NNDEPLOY_CC_API ClassificationPostParam : public base::Param {
 public:
  int topk_ = 1;

  int version_ = -1;
};

class NNDEPLOY_CC_API ClassificationPostProcess : public dag::Node {
 public:
  ClassificationPostProcess(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::classification::ClassificationPostProcess";
    param_ = std::make_shared<ClassificationPostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<ClassificationResult>();
  }
  ClassificationPostProcess(const std::string &name,
                            std::vector<dag::Edge *> inputs,
                            std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::classification::ClassificationPostProcess";
    param_ = std::make_shared<ClassificationPostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<ClassificationResult>();
  }
  virtual ~ClassificationPostProcess() {}

  virtual base::Status run();
};

/**
 * @brief Implementation of ResNet classification network graph structure
 * @details This class sits between static and dynamic graphs, with each desc
 * specifying outputs_ Contains three main nodes:
 * 1. Preprocessing node (pre_): Performs image color conversion and resizing
 * 2. Inference node (infer_): Executes ResNet model inference
 * 3. Postprocessing node (post_): Processes classification results
 */
class NNDEPLOY_CC_API ClassificationResnetGraph : public dag::Graph {
 public:
  ClassificationResnetGraph(const std::string &name) : dag::Graph(name) {
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<ClassificationResult>();
  }
  ClassificationResnetGraph(const std::string &name,
                            std::vector<dag::Edge *> inputs,
                            std::vector<dag::Edge *> outputs)
      : dag::Graph(name, inputs, outputs) {
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<ClassificationResult>();
  }

  virtual ~ClassificationResnetGraph() {}

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
    pre_param->h_ = 224;
    pre_param->w_ = 224;
    pre_param->mean_[0] = 0.485;
    pre_param->mean_[1] = 0.456;
    pre_param->mean_[2] = 0.406;
    pre_param->std_[0] = 0.229;
    pre_param->std_[1] = 0.224;
    pre_param->std_[2] = 0.225;

    // Create inference node for ResNet model execution
    infer_ = dynamic_cast<infer::Infer *>(
        this->createNode<infer::Infer>(infer_desc));
    if (infer_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create inference node");
      return base::kStatusCodeErrorInvalidParam;
    }
    infer_->setInferenceType(inference_type);

    // Create postprocessing node for classification results
    post_ = this->createNode<ClassificationPostProcess>(post_desc);
    if (post_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create postprocessing node");
      return base::kStatusCodeErrorInvalidParam;
    }
    ClassificationPostParam *post_param =
        dynamic_cast<ClassificationPostParam *>(post_->getParam());
    post_param->topk_ = 1;

    return base::kStatusCodeOk;
  }

  base::Status make(base::InferenceType inference_type) {
    // Create preprocessing node for image preprocessing
    pre_ = this->createNode<preprocess::CvtColorResize>(
        "preprocess::CvtColorResize");
    if (pre_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create preprocessing node");
      return base::kStatusCodeErrorInvalidParam;
    }
    pre_->setGraph(this);
    preprocess::CvtclorResizeParam *pre_param =
        dynamic_cast<preprocess::CvtclorResizeParam *>(pre_->getParam());
    pre_param->src_pixel_type_ = base::kPixelTypeBGR;
    pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
    pre_param->interp_type_ = base::kInterpTypeLinear;
    pre_param->h_ = 224;
    pre_param->w_ = 224;
    pre_param->mean_[0] = 0.485;
    pre_param->mean_[1] = 0.456;
    pre_param->mean_[2] = 0.406;
    pre_param->std_[0] = 0.229;
    pre_param->std_[1] = 0.224;
    pre_param->std_[2] = 0.225;

    // Create inference node for ResNet model execution
    infer_ = dynamic_cast<infer::Infer *>(
        this->createNode<infer::Infer>("infer::Infer"));
    if (infer_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create inference node");
      return base::kStatusCodeErrorInvalidParam;
    }
    infer_->setGraph(this);
    infer_->setInferenceType(inference_type);
    infer_->setInputName("data", 0);
    infer_->setOutputName("resnetv17_dense0_fwd", 0);

    // Create postprocessing node for classification results
    post_ = this->createNode<ClassificationPostProcess>(
        "ClassificationPostProcess");
    if (post_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create postprocessing node");
      return base::kStatusCodeErrorInvalidParam;
    }
    post_->setGraph(this);
    ClassificationPostParam *post_param =
        dynamic_cast<ClassificationPostParam *>(post_->getParam());
    post_param->topk_ = 1;

    return base::kStatusCodeOk;
  }

  base::Status setInferParam(base::DeviceType device_type,
                             base::ModelType model_type, bool is_path,
                             std::vector<std::string> &model_value) {
    // auto infer = dynamic_cast<infer::Infer *>(infer_);
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

  base::Status setTopk(int topk) {
    ClassificationPostParam *param =
        dynamic_cast<ClassificationPostParam *>(post_->getParam());
    param->topk_ = topk;
    return base::kStatusCodeOk;
  }

  std::vector<dag::Edge *> forward(std::vector<dag::Edge *> inputs) {
    inputs = (*pre_)(inputs);
    inputs = (*infer_)(inputs);
    std::vector<dag::Edge *> outputs = (*post_)(inputs);
    return outputs;
  }

 private:
  dag::Node *pre_ = nullptr;       ///< Preprocessing node pointer
  infer::Infer *infer_ = nullptr;  ///< Inference node pointer
  dag::Node *post_ = nullptr;      ///< Postprocessing node pointer
};

}  // namespace classification
}  // namespace nndeploy

#endif /* _NNDEPLOY_CLASSIFICATION_CLASSIFICATION_H_ */