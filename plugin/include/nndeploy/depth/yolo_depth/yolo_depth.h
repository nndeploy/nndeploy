#ifndef _NNDEPLOY_DEPTH_YOLO_DEPTH_YOLO_DEPTH_H_
#define _NNDEPLOY_DEPTH_YOLO_DEPTH_YOLO_DEPTH_H_

#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/param.h"
#include "nndeploy/base/status.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/graph.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/depth/depth_anything/depth_anything.h"
#include "nndeploy/infer/infer.h"
#include "nndeploy/preprocess/cvt_resize_norm_trans.h"

#include <vector>

namespace nndeploy {
namespace depth {

/**
 * @brief YOLO Depth 深度估计图
 *
 * 支持 Ultralytics YOLO 系列深度估计模型（YOLO26n/s/m/l/x-depth）。
 * 复用 DepthAnything 的 DepthPostProcess 和 DepthResult，
 * 仅预处理参数不同（YOLO 标准归一化）。
 * 管线：cv::Mat → CvtResizeNormTrans → Infer → DepthPostProcess → DepthResult
 */
class NNDEPLOY_CC_API YoloDepthGraph : public dag::Graph {
 public:
  YoloDepthGraph(const std::string &name) : dag::Graph(name) {
    key_ = "nndeploy::depth::YoloDepthGraph";
    desc_ = "YOLO Depth 深度估计图：cv::Mat->preprocess->infer->postprocess->DepthResult";
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<DepthResult>();
    pre_ = dynamic_cast<preprocess::CvtResizeNormTrans *>(
        this->createNode<preprocess::CvtResizeNormTrans>("preprocess"));
    infer_ = dynamic_cast<infer::Infer *>(
        this->createNode<infer::Infer>("infer"));
    post_ = dynamic_cast<DepthPostProcess *>(
        this->createNode<DepthPostProcess>("postprocess"));
  }

  YoloDepthGraph(const std::string &name, std::vector<dag::Edge *> inputs,
                 std::vector<dag::Edge *> outputs)
      : dag::Graph(name, inputs, outputs) {
    key_ = "nndeploy::depth::YoloDepthGraph";
    desc_ = "YOLO Depth 深度估计图：cv::Mat->preprocess->infer->postprocess->DepthResult";
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<DepthResult>();
    pre_ = dynamic_cast<preprocess::CvtResizeNormTrans *>(
        this->createNode<preprocess::CvtResizeNormTrans>("preprocess"));
    infer_ = dynamic_cast<infer::Infer *>(
        this->createNode<infer::Infer>("infer"));
    post_ = dynamic_cast<DepthPostProcess *>(
        this->createNode<DepthPostProcess>("postprocess"));
  }

  virtual ~YoloDepthGraph() {}

  /**
   * @brief 设置 YOLO Depth 默认参数
   *
   * 预处理（与 Ultralytics YOLO 检测/分割模型一致）：
   *   - 输入尺寸: 768×768
   *   - 归一化: pixel / 255.0（仅缩放，无减均值）
   *   - std_ = 1.0：归一化公式 (input * scale - mean) / std
   *     其中 scale = 1/255，mean = 0，std = 1 → 等价于 input / 255
   *
   * 后处理：复用 DepthPostProcess，设置模型输入尺寸为 768×768
   */
  virtual base::Status defaultParam() {
    preprocess::CvtResizeNormTransParam *pre_param =
        dynamic_cast<preprocess::CvtResizeNormTransParam *>(pre_->getParam());
    if (pre_param == nullptr) {
      NNDEPLOY_LOGE("pre_param is nullptr");
      return base::kStatusCodeErrorInvalidParam;
    }

    pre_param->src_pixel_type_ = base::kPixelTypeBGR;
    pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
    pre_param->interp_type_ = base::kInterpTypeLinear;
    pre_param->h_ = 768;
    pre_param->w_ = 768;
    pre_param->mean_[0] = 0.0f;
    pre_param->mean_[1] = 0.0f;
    pre_param->mean_[2] = 0.0f;
    pre_param->std_[0] = 1.0f;
    pre_param->std_[1] = 1.0f;
    pre_param->std_[2] = 1.0f;
    pre_param->scale_[0] = 1.0f / 255.0f;
    pre_param->scale_[1] = 1.0f / 255.0f;
    pre_param->scale_[2] = 1.0f / 255.0f;
    pre_param->scale_[3] = 1.0f / 255.0f;

    DepthPostParam *post_param = dynamic_cast<DepthPostParam *>(post_->getParam());
    if (post_param == nullptr) {
      NNDEPLOY_LOGE("post_param is nullptr");
      return base::kStatusCodeErrorInvalidParam;
    }
    post_param->model_h_ = 768;
    post_param->model_w_ = 768;

    return base::kStatusCodeOk;
  }

  base::Status make(const dag::NodeDesc &pre_desc,
                    const dag::NodeDesc &infer_desc,
                    base::InferenceType inference_type,
                    const dag::NodeDesc &post_desc) {
    this->setNodeDesc(pre_, pre_desc);
    this->setNodeDesc(infer_, infer_desc);
    this->setNodeDesc(post_, post_desc);
    base::Status status = this->defaultParam();
    if (status != base::kStatusCodeOk) {
      return status;
    }
    return this->setInferenceType(inference_type);
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

  std::vector<dag::Edge *> forward(std::vector<dag::Edge *> inputs) {
    std::vector<dag::Edge *> pre_outputs = (*pre_)(inputs);
    std::vector<dag::Edge *> infer_outputs = (*infer_)(pre_outputs);
    std::vector<dag::Edge *> post_outputs = (*post_)(infer_outputs);
    return post_outputs;
  }

 private:
  dag::Node *pre_ = nullptr;
  infer::Infer *infer_ = nullptr;
  dag::Node *post_ = nullptr;
};

}  // namespace depth
}  // namespace nndeploy

#endif  // _NNDEPLOY_DEPTH_YOLO_DEPTH_YOLO_DEPTH_H_
