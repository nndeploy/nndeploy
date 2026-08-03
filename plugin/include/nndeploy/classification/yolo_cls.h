
#ifndef _NNDEPLOY_CLASSIFICATION_YOLO_CLS_H_
#define _NNDEPLOY_CLASSIFICATION_YOLO_CLS_H_

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
#include "nndeploy/classification/classification.h"
#include "nndeploy/classification/result.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/infer/infer.h"
#include "nndeploy/preprocess/cvt_resize_norm_trans.h"

namespace nndeploy {
namespace classification {

class NNDEPLOY_CC_API YoloClsGraph : public dag::Graph {
 public:
  YoloClsGraph(const std::string &name) : dag::Graph(name) {
    key_ = "nndeploy::classification::YoloClsGraph";
    desc_ =
        "YOLOv8/11/26 classification "
        "graph[cv::Mat->preprocess->infer->postprocess->ClassificationResult]";
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<ClassificationResult>();
    pre_ = dynamic_cast<preprocess::CvtResizeNormTrans *>(
        this->createNode<preprocess::CvtResizeNormTrans>("preprocess"));
    infer_ =
        dynamic_cast<infer::Infer *>(this->createNode<infer::Infer>("infer"));
    post_ = dynamic_cast<ClassificationPostProcess *>(
        this->createNode<ClassificationPostProcess>("postprocess"));
  }
  YoloClsGraph(const std::string &name, std::vector<dag::Edge *> inputs,
               std::vector<dag::Edge *> outputs)
      : dag::Graph(name, inputs, outputs) {
    key_ = "nndeploy::classification::YoloClsGraph";
    desc_ =
        "YOLOv8/11/26 classification "
        "graph[cv::Mat->preprocess->infer->postprocess->ClassificationResult]";
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<ClassificationResult>();
    pre_ = dynamic_cast<preprocess::CvtResizeNormTrans *>(
        this->createNode<preprocess::CvtResizeNormTrans>("preprocess"));
    infer_ =
        dynamic_cast<infer::Infer *>(this->createNode<infer::Infer>("infer"));
    post_ = dynamic_cast<ClassificationPostProcess *>(
        this->createNode<ClassificationPostProcess>("postprocess"));
  }

  virtual ~YoloClsGraph() {}

  virtual base::Status defaultParam() {
    preprocess::CvtResizeNormTransParam *pre_param =
        dynamic_cast<preprocess::CvtResizeNormTransParam *>(pre_->getParam());
    pre_param->src_pixel_type_ = base::kPixelTypeBGR;
    pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
    pre_param->interp_type_ = base::kInterpTypeLinear;
    pre_param->h_ = 640;
    pre_param->w_ = 640;
    pre_param->mean_[0] = 0.0f;
    pre_param->mean_[1] = 0.0f;
    pre_param->mean_[2] = 0.0f;
    pre_param->std_[0] = 1.0f;
    pre_param->std_[1] = 1.0f;
    pre_param->std_[2] = 1.0f;

    ClassificationPostParam *post_param =
        dynamic_cast<ClassificationPostParam *>(post_->getParam());
    post_param->topk_ = 5;
    post_param->is_softmax_ = true;

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
      if (param == nullptr) {
        NNDEPLOY_LOGE("param is nullptr");
        return base::kStatusCodeErrorInvalidParam;
      }
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
  dag::Node *pre_ = nullptr;
  infer::Infer *infer_ = nullptr;
  dag::Node *post_ = nullptr;
};

}  // namespace classification
}  // namespace nndeploy

#endif /* _NNDEPLOY_CLASSIFICATION_YOLO_CLS_H_ */
