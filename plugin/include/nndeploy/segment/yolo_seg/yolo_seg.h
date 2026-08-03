
#ifndef _NNDEPLOY_SEGMENT_YOLO_SEG_YOLO_SEG_H_
#define _NNDEPLOY_SEGMENT_YOLO_SEG_YOLO_SEG_H_

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
#include "nndeploy/segment/result.h"

namespace nndeploy {
namespace segment {

/**
 * @brief 实例分割后处理参数类
 *
 * 用于配置 YOLOv8/11/26-Seg 模型（实例分割）的后处理参数，
 * 包括分数阈值、NMS 阈值、类别数量、原型掩码相关等。
 * 输出为 BBoxResult（边界框）+ SegMaskResult（实例掩码），解耦设计。
 */
class NNDEPLOY_CC_API YoloSegPostParam : public base::Param {
 public:
  float score_threshold_ = 0.5;  // 分数阈值，用于过滤低置信度的检测框
  float nms_threshold_ = 0.45;   // 非最大抑制(NMS)阈值，用于合并重叠的检测框
  int num_classes_ = 80;         // 模型可以识别的类别数量（COCO 数据集为 80 类）
  int model_h_ = 640;            // 模型输入图像的高度
  int model_w_ = 640;            // 模型输入图像的宽度
  int version_ = 8;              // YOLO 版本号，支持 8/11/26

  using base::Param::serialize;
  virtual base::Status serialize(rapidjson::Value &json,
                                 rapidjson::Document::AllocatorType &allocator);
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value &json);
};

/**
 * @brief 实例分割后处理节点
 *
 * 将模型推理输出的检测 Tensor 和原型掩码 Tensor 转换为 BBoxResult + SegMaskResult。
 * 支持 YOLOv8/11/26-Seg 模型，
 * 包括边界框解码、类别分类、NMS 和掩码生成。
 * 解耦设计：边界框和掩码分离输出，各自独立绘制。
 */
class NNDEPLOY_CC_API YoloSegPostProcess : public dag::Node {
 public:
  YoloSegPostProcess(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::segment::YoloSegPostProcess";
    desc_ = "YOLOv8/11/26 seg postprocess[device::Tensor->BBoxResult|SegMaskResult]";
    param_ = std::make_shared<YoloSegPostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<detect::BBoxResult>("bbox");
    this->setOutputTypeInfo<SegMaskResult>("mask");
  }
  YoloSegPostProcess(const std::string &name, std::vector<dag::Edge *> inputs,
                     std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::segment::YoloSegPostProcess";
    desc_ = "YOLOv8/11/26 seg postprocess[device::Tensor->BBoxResult|SegMaskResult]";
    param_ = std::make_shared<YoloSegPostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<detect::BBoxResult>("bbox");
    this->setOutputTypeInfo<SegMaskResult>("mask");
  }
  virtual ~YoloSegPostProcess() {}

  virtual base::Status run();
};

/**
 * @brief 实例分割完整工作流图
 *
 * 封装了预处理(CvtResizeNormTrans) -> 推理(Infer) -> 后处理(YoloSegPostProcess)
 * 的完整 YOLOv8/11/26-Seg 实例分割流水线。
 * 推理输出有两个：检测 Tensor 和原型掩码 Tensor。
 */
class NNDEPLOY_CC_API YoloSegGraph : public dag::Graph {
 public:
  YoloSegGraph(const std::string &name) : dag::Graph(name) {
    key_ = "nndeploy::segment::YoloSegGraph";
    desc_ =
        "YOLOv8/11/26 seg graph[cv::Mat->preprocess->infer->postprocess->BBoxResult|SegMaskResult]";
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<detect::BBoxResult>("bbox");
    this->setOutputTypeInfo<SegMaskResult>("mask");
    pre_ = dynamic_cast<preprocess::CvtResizeNormTrans *>(
        this->createNode<preprocess::CvtResizeNormTrans>("preprocess"));
    infer_ =
        dynamic_cast<infer::Infer *>(this->createNode<infer::Infer>("infer"));
    post_ = dynamic_cast<YoloSegPostProcess *>(
        this->createNode<YoloSegPostProcess>("postprocess"));
  }

  YoloSegGraph(const std::string &name, std::vector<dag::Edge *> inputs,
               std::vector<dag::Edge *> outputs)
      : dag::Graph(name, inputs, outputs) {
    key_ = "nndeploy::segment::YoloSegGraph";
    desc_ =
        "YOLOv8/11/26 seg graph[cv::Mat->preprocess->infer->postprocess->BBoxResult|SegMaskResult]";
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<detect::BBoxResult>("bbox");
    this->setOutputTypeInfo<SegMaskResult>("mask");
    pre_ = dynamic_cast<preprocess::CvtResizeNormTrans *>(
        this->createNode<preprocess::CvtResizeNormTrans>("preprocess"));
    infer_ =
        dynamic_cast<infer::Infer *>(this->createNode<infer::Infer>("infer"));
    post_ = dynamic_cast<YoloSegPostProcess *>(
        this->createNode<YoloSegPostProcess>("postprocess"));
  }

  virtual ~YoloSegGraph() {}

  virtual base::Status defaultParam() {
    preprocess::CvtResizeNormTransParam *pre_param =
        dynamic_cast<preprocess::CvtResizeNormTransParam *>(pre_->getParam());
    pre_param->src_pixel_type_ = base::kPixelTypeBGR;
    pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
    pre_param->interp_type_ = base::kInterpTypeLinear;
    pre_param->h_ = 640;
    pre_param->w_ = 640;

    YoloSegPostParam *post_param =
        dynamic_cast<YoloSegPostParam *>(post_->getParam());
    post_param->score_threshold_ = 0.5;
    post_param->nms_threshold_ = 0.45;
    post_param->num_classes_ = 80;
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

  std::vector<dag::Edge *> forward(std::vector<dag::Edge *> inputs) {
    std::vector<dag::Edge *> pre_outputs = (*pre_)(inputs);
    // infer has 2 outputs: [0]=detection, [1]=proto masks
    std::vector<dag::Edge *> infer_outputs = (*infer_)(pre_outputs);
    std::vector<dag::Edge *> post_outputs = (*post_)(infer_outputs);
    return post_outputs;
  }

 private:
  dag::Node *pre_ = nullptr;
  infer::Infer *infer_ = nullptr;
  dag::Node *post_ = nullptr;
};

}  // namespace segment
}  // namespace nndeploy

#endif /* _NNDEPLOY_SEGMENT_YOLO_SEG_YOLO_SEG_H_ */
