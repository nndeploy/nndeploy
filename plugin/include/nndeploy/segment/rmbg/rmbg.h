
#ifndef _NNDEPLOY_SEGMENT_SEGMENT_RMBG_RMBG_H_
#define _NNDEPLOY_SEGMENT_SEGMENT_RMBG_RMBG_H_

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
#include "nndeploy/preprocess/cvtcolor_resize.h"
#include "nndeploy/segment/result.h"

namespace nndeploy {
namespace segment {

#define NNDEPLOY_RMBGV5 "NNDEPLOY_RMBGV1.4"

class NNDEPLOY_CC_API RMBGPostParam : public base::Param {
 public:
  int version_ = -1;
};

class NNDEPLOY_CC_API RMBGPostProcess : public dag::Node {
 public:
  RMBGPostProcess(const std::string &name) : Node(name) {
    param_ = std::make_shared<RMBGPostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<SegmentResult>();
  }

  RMBGPostProcess(const std::string &name, std::vector<dag::Edge *> inputs,
                  std::vector<dag::Edge *> outputs)
      : Node(name, inputs, outputs) {
    param_ = std::make_shared<RMBGPostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<SegmentResult>();
  }
  virtual ~RMBGPostProcess() {}

  virtual base::Status run();
};

class NNDEPLOY_CC_API SegmentRMBGGraph : public dag::Graph {
 public:
  SegmentRMBGGraph(const std::string &name) : dag::Graph(name) {
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<SegmentResult>();
  }

  SegmentRMBGGraph(const std::string &name, std::vector<dag::Edge *> inputs,
                   std::vector<dag::Edge *> outputs)
      : dag::Graph(name, inputs, outputs) {
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<SegmentResult>();
  }

  virtual ~SegmentRMBGGraph() {}

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
    pre_param->h_ = 1024;
    pre_param->w_ = 1024;
    pre_param->mean_[0] = 0.5f;
    pre_param->mean_[1] = 0.5f;
    pre_param->mean_[2] = 0.5f;
    pre_param->mean_[3] = 0.5f;

    // Create inference node for ResNet model execution
    infer_ = dynamic_cast<infer::Infer *>(
        this->createNode<infer::Infer>(infer_desc));
    if (infer_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create inference node");
      return base::kStatusCodeErrorInvalidParam;
    }
    infer_->setInferenceType(inference_type);

    // Create postprocessing node for classification results
    post_ = this->createNode<RMBGPostProcess>(post_desc);
    if (post_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create postprocessing node");
      return base::kStatusCodeErrorInvalidParam;
    }
    RMBGPostParam *post_param =
        dynamic_cast<RMBGPostParam *>(post_->getParam());
    post_param->version_ = 14;

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

  base::Status setVersion(int version) {
    RMBGPostParam *param = dynamic_cast<RMBGPostParam *>(post_->getParam());
    param->version_ = version;
    return base::kStatusCodeOk;
  }

  std::vector<dag::Edge *> forward(std::vector<dag::Edge *> inputs) {
    std::vector<dag::Edge *> pre_outputs = (*pre_)(inputs);
    std::vector<dag::Edge *> infer_outputs = (*infer_)(pre_outputs);
    std::vector<dag::Edge *> post_inputs;
    post_inputs.push_back(inputs[0]);
    post_inputs.push_back(infer_outputs[0]);
    std::vector<dag::Edge *> post_outputs = (*post_)(post_inputs);
    return post_outputs;
  }

 private:
  dag::Node *pre_ = nullptr;       ///< Preprocessing node pointer
  infer::Infer *infer_ = nullptr;  ///< Inference node pointer
  dag::Node *post_ = nullptr;      ///< Postprocessing node pointer
};

}  // namespace segment
}  // namespace nndeploy

#endif /* _NNDEPLOY_SEGMENT_SEGMENT_RMBG_RMBG_H_ */