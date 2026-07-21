
#ifndef _NNDEPLOY_SEGMENT_RF_DETR_SEG_RF_DETR_SEG_H_
#define _NNDEPLOY_SEGMENT_RF_DETR_SEG_RF_DETR_SEG_H_

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
#include "nndeploy/segment/result.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/infer/infer.h"
#include "nndeploy/preprocess/cvt_resize_norm_trans.h"

namespace nndeploy {
namespace segment {

/**
 * @brief RF-DETR-Seg instance segmentation post-processing parameters.
 *
 * RF-DETR-Seg (Roboflow) is a DETR-series instance segmentation model
 * that outputs three tensors: detection boxes, class logits, and
 * per-query segmentation masks.
 */
class NNDEPLOY_CC_API RfDetrSegPostParam : public base::Param {
 public:
  float score_threshold_ = 0.5;
  int num_classes_ = 80;
  int model_h_ = 640;
  int model_w_ = 640;
  float nms_threshold_ = 0.5;

  using base::Param::serialize;
  virtual base::Status serialize(rapidjson::Value &json,
                                 rapidjson::Document::AllocatorType &allocator);
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value &json);
};

/**
 * @brief RF-DETR-Seg instance segmentation post-processing node.
 *
 * Takes three input tensors from the model:
 *   inputs_[0] = dets   — [batch, num_queries, 4]  boxes [cx,cy,w,h] normalized
 *   inputs_[1] = labels — [batch, num_queries, 91] raw class logits
 *   inputs_[2] = masks  — [batch, num_queries, mask_h, mask_w] per-query mask logits
 *
 * Outputs BBoxResult (edge 0, lightweight bbox) and SegMaskResult
 * (edge 1, per-instance masks) — decoupled design matching YoloSeg pattern.
 */
class NNDEPLOY_CC_API RfDetrSegPostProcess : public dag::Node {
 public:
  RfDetrSegPostProcess(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::segment::RfDetrSegPostProcess";
    desc_ = "RF-DETR-Seg postprocess[device::Tensor->BBoxResult|SegMaskResult]";
    param_ = std::make_shared<RfDetrSegPostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setInputTypeInfo<device::Tensor>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<detect::BBoxResult>("bbox");
    this->setOutputTypeInfo<SegMaskResult>("mask");
  }
  RfDetrSegPostProcess(const std::string &name, std::vector<dag::Edge *> inputs,
                       std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::segment::RfDetrSegPostProcess";
    desc_ = "RF-DETR-Seg postprocess[device::Tensor->BBoxResult|SegMaskResult]";
    param_ = std::make_shared<RfDetrSegPostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setInputTypeInfo<device::Tensor>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<detect::BBoxResult>("bbox");
    this->setOutputTypeInfo<SegMaskResult>("mask");
  }
  virtual ~RfDetrSegPostProcess() {}

  virtual base::Status run();
};

/**
 * @brief RF-DETR-Seg complete instance segmentation pipeline graph.
 *
 * Pipeline: CvtResizeNormTrans -> Infer -> RfDetrSegPostProcess
 *
 * The Infer node produces three outputs (dets, labels, masks)
 * which are fed into the post-processor for instance segmentation.
 * Outputs dual channels: BBoxResult (bbox) + SegMaskResult (mask).
 */
class NNDEPLOY_CC_API RfDetrSegGraph : public dag::Graph {
 public:
  RfDetrSegGraph(const std::string &name) : dag::Graph(name) {
    key_ = "nndeploy::segment::RfDetrSegGraph";
    desc_ =
        "RF-DETR-Seg graph[cv::Mat->preprocess->infer->postprocess->BBoxResult|SegMaskResult]";
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<detect::BBoxResult>("bbox");
    this->setOutputTypeInfo<SegMaskResult>("mask");
    pre_ = dynamic_cast<preprocess::CvtResizeNormTrans *>(
        this->createNode<preprocess::CvtResizeNormTrans>("preprocess"));
    infer_ =
        dynamic_cast<infer::Infer *>(this->createNode<infer::Infer>("infer"));
    post_ = dynamic_cast<RfDetrSegPostProcess *>(
        this->createNode<RfDetrSegPostProcess>("postprocess"));
  }

  RfDetrSegGraph(const std::string &name, std::vector<dag::Edge *> inputs,
                 std::vector<dag::Edge *> outputs)
      : dag::Graph(name, inputs, outputs) {
    key_ = "nndeploy::segment::RfDetrSegGraph";
    desc_ =
        "RF-DETR-Seg graph[cv::Mat->preprocess->infer->postprocess->BBoxResult|SegMaskResult]";
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<detect::BBoxResult>("bbox");
    this->setOutputTypeInfo<SegMaskResult>("mask");
    pre_ = dynamic_cast<preprocess::CvtResizeNormTrans *>(
        this->createNode<preprocess::CvtResizeNormTrans>("preprocess"));
    infer_ =
        dynamic_cast<infer::Infer *>(this->createNode<infer::Infer>("infer"));
    post_ = dynamic_cast<RfDetrSegPostProcess *>(
        this->createNode<RfDetrSegPostProcess>("postprocess"));
  }

  virtual ~RfDetrSegGraph() {}

  virtual base::Status defaultParam() {
    preprocess::CvtResizeNormTransParam *pre_param =
        dynamic_cast<preprocess::CvtResizeNormTransParam *>(pre_->getParam());
    pre_param->src_pixel_type_ = base::kPixelTypeBGR;
    pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
    pre_param->interp_type_ = base::kInterpTypeLinear;
    pre_param->h_ = model_hw_;
    pre_param->w_ = model_hw_;

    RfDetrSegPostParam *post_param =
        dynamic_cast<RfDetrSegPostParam *>(post_->getParam());
    post_param->score_threshold_ = 0.5;
    post_param->num_classes_ = 80;
    post_param->model_h_ = model_hw_;
    post_param->model_w_ = model_hw_;
    post_param->nms_threshold_ = 0.5;

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

  base::Status setScoreThreshold(float score_threshold) {
    RfDetrSegPostParam *param =
        dynamic_cast<RfDetrSegPostParam *>(post_->getParam());
    param->score_threshold_ = score_threshold;
    return base::kStatusCodeOk;
  }

  base::Status setNumClasses(int num_classes) {
    RfDetrSegPostParam *param =
        dynamic_cast<RfDetrSegPostParam *>(post_->getParam());
    param->num_classes_ = num_classes;
    return base::kStatusCodeOk;
  }

  base::Status setModelHW(int model_h, int model_w) {
    RfDetrSegPostParam *param =
        dynamic_cast<RfDetrSegPostParam *>(post_->getParam());
    param->model_h_ = model_h;
    param->model_w_ = model_w;
    return base::kStatusCodeOk;
  }

  std::vector<dag::Edge *> forward(std::vector<dag::Edge *> inputs) {
    std::vector<dag::Edge *> pre_outputs = (*pre_)(inputs);
    // Infer has 3 outputs (dets, labels, masks)
    std::vector<dag::Edge *> infer_outputs = (*infer_)(pre_outputs);
    std::vector<dag::Edge *> post_outputs = (*post_)(infer_outputs);
    return post_outputs;
  }

  void setResolution(int hw) { model_hw_ = hw; }

 private:
  dag::Node *pre_ = nullptr;
  infer::Infer *infer_ = nullptr;
  dag::Node *post_ = nullptr;
  int model_hw_ = 640;
};

}  // namespace segment
}  // namespace nndeploy

#endif /* _NNDEPLOY_SEGMENT_RF_DETR_SEG_RF_DETR_SEG_H_ */
