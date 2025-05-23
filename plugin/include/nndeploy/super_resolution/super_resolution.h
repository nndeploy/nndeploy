
#ifndef _NNDEPLOY_SuperResolution_SuperResolution_H_
#define _NNDEPLOY_SuperResolution_SuperResolution_H_

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
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/infer/infer.h"
#include "nndeploy/preprocess/cvtcolor_bn.h"
#include "nndeploy/preprocess/params.h"
#include "nndeploy/preprocess/batch_preprocess.h"

namespace nndeploy {
namespace super_resolution {


class NNDEPLOY_CC_API SuperResolutionPostProcess : public dag::Node {
 public:
  SuperResolutionPostProcess(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::super_resolution::SuperResolutionPostProcess";
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<std::vector<cv::Mat>>();
  }
  SuperResolutionPostProcess(const std::string &name,
                            std::vector<dag::Edge *> inputs,
                            std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::super_resolution::SuperResolutionPostProcess";
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<std::vector<cv::Mat>>();
  }
  virtual ~SuperResolutionPostProcess() {}

  virtual base::Status run();
};

/**
 * @brief Implementation of ResNet SuperResolution network graph structure
 * @details This class sits between static and dynamic graphs, with each desc
 * specifying outputs_ Contains three main nodes:
 * 1. Preprocessing node (pre_): Performs image color conversion and resizing
 * 2. Inference node (infer_): Executes ResNet model inference
 * 3. Postprocessing node (post_): Processes SuperResolution results
 */
class NNDEPLOY_CC_API SuperResolutionGraph : public dag::Graph {
 public:
  SuperResolutionGraph(const std::string &name) : dag::Graph(name) {
    key_ = "nndeploy::super_resolution::SuperResolutionGraph";
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<std::vector<cv::Mat>>();
  }
  SuperResolutionGraph(const std::string &name,
                            std::vector<dag::Edge *> inputs,
                            std::vector<dag::Edge *> outputs)
      : dag::Graph(name, inputs, outputs) {
    key_ = "nndeploy::super_resolution::SuperResolutionGraph";
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<std::vector<cv::Mat>>();
  }

  virtual ~SuperResolutionGraph() {}

  base::Status make(const dag::NodeDesc &pre_desc,
                    const dag::NodeDesc &infer_desc,
                    base::InferenceType inference_type,
                    const dag::NodeDesc &post_desc) {
    // Create preprocessing node for image preprocessing
    pre_ = (preprocess::BatchPreprocess *)this->createNode<preprocess::BatchPreprocess>(pre_desc);
    if (pre_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create preprocessing node");
      return base::kStatusCodeErrorInvalidParam;
    }
    pre_->setNodeKey("nndeploy::preprocess::CvtColorBn");
    pre_->make();
    preprocess::CvtcolorBnParam *pre_param =
        dynamic_cast<preprocess::CvtcolorBnParam *>(pre_->getParam());
    pre_param->src_pixel_type_ = base::kPixelTypeBGR;
    pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
    pre_param->mean_[0] = 0.0;
    pre_param->mean_[1] = 0.0;
    pre_param->mean_[2] = 0.0;
    pre_param->std_[0] = 1.0;
    pre_param->std_[1] = 1.0;
    pre_param->std_[2] = 1.0;

    // Create inference node for ResNet model execution
    infer_ = dynamic_cast<infer::Infer *>(
        this->createNode<infer::Infer>(infer_desc));
    if (infer_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create inference node");
      return base::kStatusCodeErrorInvalidParam;
    }
    infer_->setInferenceType(inference_type);

    // Create postprocessing node for SuperResolution results
    post_ = this->createNode<SuperResolutionPostProcess>(post_desc);
    if (post_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create postprocessing node");
      return base::kStatusCodeErrorInvalidParam;
    }
    
    return base::kStatusCodeOk;
  }

  base::Status make(base::InferenceType inference_type) {
    // Create preprocessing node for image preprocessing
    pre_ = (preprocess::BatchPreprocess *)this->createNode<preprocess::BatchPreprocess>(
        "preprocess::BatchPreprocess");
    if (pre_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create preprocessing node");
      return base::kStatusCodeErrorInvalidParam;
    }
    pre_->setGraph(this);
    pre_->setNodeKey("nndeploy::preprocess::CvtColorBn");
    pre_->make();
    preprocess::CvtcolorBnParam *pre_param =
        dynamic_cast<preprocess::CvtcolorBnParam *>(pre_->getParam());
    if (pre_param == nullptr) {
      NNDEPLOY_LOGE("Failed to get preprocessing node parameter");
      return base::kStatusCodeErrorInvalidParam;
    }
    pre_param->src_pixel_type_ = base::kPixelTypeBGR;
    pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
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

    // Create postprocessing node for SuperResolution results
    post_ = this->createNode<SuperResolutionPostProcess>(
        "SuperResolutionPostProcess");
    if (post_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create postprocessing node");
      return base::kStatusCodeErrorInvalidParam;
    }
    post_->setGraph(this);

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
    preprocess::CvtcolorBnParam *param =
        dynamic_cast<preprocess::CvtcolorBnParam *>(pre_->getParam());
    param->src_pixel_type_ = pixel_type;
    return base::kStatusCodeOk;
  }

  std::vector<dag::Edge *> forward(std::vector<dag::Edge *> inputs) {
    inputs = (*pre_)(inputs);
    inputs = (*infer_)(inputs);
    std::vector<dag::Edge *> outputs = (*post_)(inputs);
    return outputs;
  }

 private:
  preprocess::BatchPreprocess *pre_ = nullptr;       ///< Preprocessing node pointer
  infer::Infer *infer_ = nullptr;  ///< Inference node pointer
  dag::Node *post_ = nullptr;      ///< Postprocessing node pointer
};

}  // namespace SuperResolution
}  // namespace nndeploy

#endif /* _NNDEPLOY_SuperResolution_SuperResolution_H_ */