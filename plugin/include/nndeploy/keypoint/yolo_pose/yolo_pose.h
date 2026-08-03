
#ifndef _NNDEPLOY_KEYPOINT_YOLO_POSE_YOLO_POSE_H_
#define _NNDEPLOY_KEYPOINT_YOLO_POSE_YOLO_POSE_H_

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
#include "nndeploy/keypoint/result.h"
#include "nndeploy/preprocess/cvt_resize_norm_trans.h"

namespace nndeploy {
namespace keypoint {

/**
 * @brief 关键点检测后处理参数类
 *
 * 用于配置 YOLOv8/11/26-Pose 模型的后处理参数，
 * 包括关键点检测的分数阈值、NMS 阈值、关键点数量等。
 */
class NNDEPLOY_CC_API KeypointPostParam : public base::Param {
 public:
  float score_threshold_ = 0.5f;  // 分数阈值，用于过滤低置信度的检测框
  float nms_threshold_ = 0.45f;   // 非最大抑制(NMS)阈值，用于合并重叠的检测框
  int num_classes_ = 1;           // 类别数量（关键点检测通常为 1）
  int num_keypoints_ = 17;        // 关键点数量（COCO 标准为 17 个关键点）
  int model_h_ = 640;             // 模型输入图像的高度
  int model_w_ = 640;             // 模型输入图像的宽度
  int version_ = 8;               // YOLO 版本号，支持 8/11/26

  using base::Param::serialize;
  virtual base::Status serialize(rapidjson::Value &json,
                                 rapidjson::Document::AllocatorType &allocator);
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value &json);
};

/**
 * @brief 关键点检测后处理节点
 *
 * 将模型推理输出的 Tensor 转换为 KeypointResult。
 * 支持 YOLOv8/11/26-Pose 模型的关键点检测后处理，
 * 包括边界框解码、关键点提取和 NMS。
 */
class NNDEPLOY_CC_API KeypointPostProcess : public dag::Node {
 public:
  KeypointPostProcess(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::keypoint::KeypointPostProcess";
    desc_ = "YOLOv8/11/26-Pose postprocess[device::Tensor->BBoxResult|KeypointResult]";
    param_ = std::make_shared<KeypointPostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<detect::BBoxResult>("bbox");
    this->setOutputTypeInfo<KeypointResult>("keypoint");
  }
  KeypointPostProcess(const std::string &name, std::vector<dag::Edge *> inputs,
                  std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::keypoint::KeypointPostProcess";
    desc_ = "YOLOv8/11/26-Pose postprocess[device::Tensor->BBoxResult|KeypointResult]";
    param_ = std::make_shared<KeypointPostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<detect::BBoxResult>("bbox");
    this->setOutputTypeInfo<KeypointResult>("keypoint");
  }
  virtual ~KeypointPostProcess() {}

  virtual base::Status run();
};

/**
 * @brief 关键点检测完整工作流图
 *
 * 封装了预处理(CvtResizeNormTrans) -> 推理(Infer) -> 后处理(KeypointPostProcess)
 * 的完整 YOLOv8/11/26-Pose 关键点检测流水线。
 */
class NNDEPLOY_CC_API KeypointGraph : public dag::Graph {
 public:
  KeypointGraph(const std::string &name) : dag::Graph(name) {
    key_ = "nndeploy::keypoint::KeypointGraph";
    desc_ = "yolov8/11/26-pose graph[cv::Mat->preprocess->infer->postprocess->BBoxResult+KeypointResult]";
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<detect::BBoxResult>("bbox");
    this->setOutputTypeInfo<KeypointResult>("keypoint");
    pre_ = dynamic_cast<preprocess::CvtResizeNormTrans *>(
        this->createNode<preprocess::CvtResizeNormTrans>("preprocess"));
    infer_ =
        dynamic_cast<infer::Infer *>(this->createNode<infer::Infer>("infer"));
    post_ = dynamic_cast<KeypointPostProcess *>(
        this->createNode<KeypointPostProcess>("postprocess"));
  }

  KeypointGraph(const std::string &name, std::vector<dag::Edge *> inputs,
            std::vector<dag::Edge *> outputs)
      : dag::Graph(name, inputs, outputs) {
    key_ = "nndeploy::keypoint::KeypointGraph";
    desc_ = "yolov8/11/26-pose graph[cv::Mat->preprocess->infer->postprocess->BBoxResult+KeypointResult]";
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<detect::BBoxResult>("bbox");
    this->setOutputTypeInfo<KeypointResult>("keypoint");
    pre_ = dynamic_cast<preprocess::CvtResizeNormTrans *>(
        this->createNode<preprocess::CvtResizeNormTrans>("preprocess"));
    infer_ =
        dynamic_cast<infer::Infer *>(this->createNode<infer::Infer>("infer"));
    post_ = dynamic_cast<KeypointPostProcess *>(
        this->createNode<KeypointPostProcess>("postprocess"));
  }

  virtual ~KeypointGraph() {}

  virtual base::Status defaultParam() {
    preprocess::CvtResizeNormTransParam *pre_param =
        dynamic_cast<preprocess::CvtResizeNormTransParam *>(pre_->getParam());
    pre_param->src_pixel_type_ = base::kPixelTypeBGR;
    pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
    pre_param->interp_type_ = base::kInterpTypeLinear;
    pre_param->h_ = 640;
    pre_param->w_ = 640;

    KeypointPostParam *post_param =
        dynamic_cast<KeypointPostParam *>(post_->getParam());
    post_param->score_threshold_ = 0.5;
    post_param->nms_threshold_ = 0.45;
    post_param->num_classes_ = 1;
    post_param->num_keypoints_ = 17;
    post_param->model_h_ = 640;
    post_param->model_w_ = 640;
    post_param->version_ = 8;

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

  base::Status setVersion(int version) {
    KeypointPostParam *param =
        dynamic_cast<KeypointPostParam *>(post_->getParam());
    if (param == nullptr) {
      NNDEPLOY_LOGE("param is nullptr");
      return base::kStatusCodeErrorInvalidParam;
    }
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
  dag::Node *pre_ = nullptr;
  infer::Infer *infer_ = nullptr;
  dag::Node *post_ = nullptr;
};

}  // namespace keypoint
}  // namespace nndeploy

#endif /* _NNDEPLOY_KEYPOINT_YOLO_POSE_YOLO_POSE_H_ */
