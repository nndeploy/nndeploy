
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
#include "nndeploy/preprocess/cvt_resize_crop_norm_trans.h"
#include "nndeploy/preprocess/cvt_resize_norm_trans.h"
#include "nndeploy/preprocess/params.h"

namespace nndeploy {
namespace classification {

class NNDEPLOY_CC_API ClassificationPostParam : public base::Param {
 public:
  int topk_ = 1;
  bool is_softmax_ = true;
  int version_ = -1;

  using base::Param::serialize;
  virtual base::Status serialize(rapidjson::Value &json,
                                 rapidjson::Document::AllocatorType &allocator);
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value &json);
};

class NNDEPLOY_CC_API ClassificationPostProcess : public dag::Node {
 public:
  ClassificationPostProcess(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::classification::ClassificationPostProcess";
    desc_ = "Classification postprocess[device::Tensor->ClassificationResult]";
    param_ = std::make_shared<ClassificationPostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<ClassificationResult>();
  }
  ClassificationPostProcess(const std::string &name,
                            std::vector<dag::Edge *> inputs,
                            std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::classification::ClassificationPostProcess";
    desc_ = "Classification postprocess[device::Tensor->ClassificationResult]";
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
class NNDEPLOY_CC_API ClassificationGraph : public dag::Graph {
 public:
  ClassificationGraph(const std::string &name) : dag::Graph(name) {
    key_ = "nndeploy::classification::ClassificationGraph";
    desc_ =
        "Classification "
        "graph[cv::Mat->preprocess->infer->postprocess->ClassificationResult]";
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<ClassificationResult>();
    pre_ = dynamic_cast<preprocess::CvtResizeCropNormTrans *>(
        this->createNode<preprocess::CvtResizeCropNormTrans>(
            "preprocess"));
    infer_ =
        dynamic_cast<infer::Infer *>(this->createNode<infer::Infer>("infer"));
    post_ = dynamic_cast<ClassificationPostProcess *>(
        this->createNode<ClassificationPostProcess>("postprocess"));
  }
  ClassificationGraph(const std::string &name, std::vector<dag::Edge *> inputs,
                      std::vector<dag::Edge *> outputs)
      : dag::Graph(name, inputs, outputs) {
    key_ = "nndeploy::classification::ClassificationGraph";
    desc_ =
        "Classification "
        "graph[cv::Mat->preprocess->infer->postprocess->ClassificationResult]";
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<ClassificationResult>();
    pre_ = dynamic_cast<preprocess::CvtResizeCropNormTrans *>(
        this->createNode<preprocess::CvtResizeCropNormTrans>(
            "preprocess"));
    infer_ =
        dynamic_cast<infer::Infer *>(this->createNode<infer::Infer>("infer"));
    post_ = dynamic_cast<ClassificationPostProcess *>(
        this->createNode<ClassificationPostProcess>("postprocess"));
  }

  virtual ~ClassificationGraph() {}

  virtual base::Status defaultParam() {
    // preprocess::CvtResizeNormTransParam *pre_param =
    //     dynamic_cast<preprocess::CvtResizeNormTransParam
    //     *>(pre_->getParam());
    // pre_param->src_pixel_type_ = base::kPixelTypeBGR;
    // pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
    // pre_param->interp_type_ = base::kInterpTypeLinear;
    // pre_param->h_ = 224;
    // pre_param->w_ = 224;
    // pre_param->mean_[0] = 0.485;
    // pre_param->mean_[1] = 0.456;
    // pre_param->mean_[2] = 0.406;
    // pre_param->std_[0] = 0.229;
    // pre_param->std_[1] = 0.224;
    // pre_param->std_[2] = 0.225;
    preprocess::CvtResizeCropNormTransParam *pre_param =
        dynamic_cast<preprocess::CvtResizeCropNormTransParam *>(
            pre_->getParam());
    pre_param->src_pixel_type_ = base::kPixelTypeBGR;
    pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
    pre_param->interp_type_ = base::kInterpTypeLinear;
    pre_param->resize_h_ = 256;
    pre_param->resize_w_ = 256;
    pre_param->mean_[0] = 0.485;
    pre_param->mean_[1] = 0.456;
    pre_param->mean_[2] = 0.406;
    pre_param->std_[0] = 0.229;
    pre_param->std_[1] = 0.224;
    pre_param->std_[2] = 0.225;
    pre_param->width_ = 224;
    pre_param->height_ = 224;

    ClassificationPostParam *post_param =
        dynamic_cast<ClassificationPostParam *>(post_->getParam());
    post_param->topk_ = 1;

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
    preprocess::CvtResizeNormTransParam *param =
        dynamic_cast<preprocess::CvtResizeNormTransParam *>(pre_->getParam());
    param->src_pixel_type_ = pixel_type;
    return base::kStatusCodeOk;
  }

  base::Status setTopk(int topk) {
    ClassificationPostParam *param =
        dynamic_cast<ClassificationPostParam *>(post_->getParam());
    param->topk_ = topk;
    return base::kStatusCodeOk;
  }

  base::Status setSoftmax(bool is_softmax) {
    ClassificationPostParam *param =
        dynamic_cast<ClassificationPostParam *>(post_->getParam());
    param->is_softmax_ = is_softmax;
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