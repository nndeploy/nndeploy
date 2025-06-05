#ifndef _NNDEPLOY_MATTING_PPMATTING_PP_MATTING_H_
#define _NNDEPLOY_MATTING_PPMATTING_PP_MATTING_H_

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
#include "nndeploy/matting/result.h"
#include "nndeploy/preprocess/cvtcolor_resize_pad.h"

namespace nndeploy {
namespace matting {

class NNDEPLOY_CC_API PPMattingPostParam : public base::Param {
 public:
  int alpha_h_;
  int alpha_w_;
  int output_h_;
  int output_w_;
};

class NNDEPLOY_CC_API PPMattingPostProcess : public dag::Node {
 public:
  PPMattingPostProcess(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::matting::PPMattingPostProcess";
    param_ = std::make_shared<PPMattingPostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<MattingResult>();
  }
  PPMattingPostProcess(const std::string &name, std::vector<dag::Edge *> inputs,
                       std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::matting::PPMattingPostProcess";
    param_ = std::make_shared<PPMattingPostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<MattingResult>();
  }
  virtual ~PPMattingPostProcess() {}

  virtual base::Status run();
};

class NNDEPLOY_CC_API PPMattingGraph : public dag::Graph {
 public:
  PPMattingGraph(const std::string &name) : dag::Graph(name) {
    key_ = "nndeploy::matting::PPMattingGraph";
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<MattingResult>();
  }
  PPMattingGraph(const std::string &name, std::vector<dag::Edge *> inputs,
                 std::vector<dag::Edge *> outputs)
      : dag::Graph(name, inputs, outputs) {
    key_ = "nndeploy::matting::PPMattingGraph";
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<MattingResult>();
  }

  virtual ~PPMattingGraph() {}

  base::Status make(const dag::NodeDesc &pre_desc,
                    const dag::NodeDesc &infer_desc,
                    base::InferenceType inference_type,
                    const dag::NodeDesc &post_desc) {
    // Create preprocessing node for matting
    pre_ = this->createNode<preprocess::CvtColorResizePad>(pre_desc);
    if (pre_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create preprocessing node");
      return base::kStatusCodeErrorInvalidParam;
    }
    preprocess::CvtclorResizePadParam *pre_param =
        dynamic_cast<preprocess::CvtclorResizePadParam *>(pre_->getParam());
    pre_param->src_pixel_type_ = base::kPixelTypeBGR;
    pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
    pre_param->interp_type_ = base::kInterpTypeLinear;
    pre_param->h_ = 1024;
    pre_param->w_ = 1024;
    pre_param->mean_[0] = 0.5f;
    pre_param->mean_[1] = 0.5f;
    pre_param->mean_[2] = 0.5f;
    pre_param->mean_[3] = 0.5f;
    pre_param->std_[0] = 0.5f;
    pre_param->std_[1] = 0.5f;
    pre_param->std_[2] = 0.5f;
    pre_param->std_[3] = 0.5f;

    // Create inference node for ppmatting model execution
    infer_ = dynamic_cast<infer::Infer *>(
        this->createNode<infer::Infer>(infer_desc));
    if (infer_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create inference node");
      return base::kStatusCodeErrorInvalidParam;
    }
    infer_->setInferenceType(inference_type);

    // Create postprocessing node for matting
    post_ = this->createNode<PPMattingPostProcess>(post_desc);
    if (post_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create postprocessing node");
      return base::kStatusCodeErrorInvalidParam;
    }
    PPMattingPostParam *post_param =
        dynamic_cast<PPMattingPostParam *>(post_->getParam());
    post_param->alpha_h_ = 1024;
    post_param->alpha_w_ = 1024;
    post_param->output_h_ = 1024;
    post_param->output_w_ = 1024;

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

  base::Status setModelHW(int model_h, int model_w) {
    preprocess::CvtclorResizePadParam *pre_param =
        dynamic_cast<preprocess::CvtclorResizePadParam *>(pre_->getParam());
    pre_param->h_ = model_h;
    pre_param->w_ = model_w;

    PPMattingPostParam *post_param =
        dynamic_cast<PPMattingPostParam *>(post_->getParam());
    post_param->alpha_h_ = model_h;
    post_param->alpha_w_ = model_w;
    post_param->output_h_ = model_h;
    post_param->output_w_ = model_w;

    return base::kStatusCodeOk;
  }

 private:
  dag::Node *pre_ = nullptr;       ///< Preprocessing node pointer
  infer::Infer *infer_ = nullptr;  ///< Inference node pointer
  dag::Node *post_ = nullptr;      ///< Postprocessing node pointer
};

}  // namespace matting
}  // namespace nndeploy

#endif