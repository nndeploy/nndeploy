
#ifndef _NNDEPLOY_DETECT_YOLO_OBB_YOLO_OBB_H_
#define _NNDEPLOY_DETECT_YOLO_OBB_YOLO_OBB_H_

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
#include "nndeploy/detect/yolo_obb/result.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"
#include "nndeploy/infer/infer.h"
#include "nndeploy/preprocess/cvt_resize_norm_trans.h"

namespace nndeploy {
namespace detect {

/**
 * @brief 旋转框检测(OBB)后处理参数类
 *
 * 用于配置 YOLOv8/11/26-Obb 模型的后处理参数。
 * v8/v11: 密集预测 [batch, 5+N, 21504], 格式 [cx,cy,w,h, cls_0..cls_N-1, angle]
 * v26: NMS-free [batch, 300, 7], 格式 [cx,cy,w,h, score, class_id, angle]
 */
class NNDEPLOY_CC_API ObbPostParam : public base::Param {
 public:
  float score_threshold_ = 0.5;  // 分数阈值，用于过滤低置信度的检测框
  float nms_threshold_ = 0.45;   // 非最大抑制(NMS)阈值，用于合并重叠的检测框
  int num_classes_ = 15;         // 模型可以识别的类别数量（DOTA 数据集为 15 类）
  int model_h_ = 1024;           // 模型输入图像的高度（yolo11n/yolo26 OBB 均为 1024）
  int model_w_ = 1024;           // 模型输入图像的宽度（yolo11n/yolo26 OBB 均为 1024）
  int version_ = 8;              // YOLO 版本号，支持 8/11(v8格式) 或 26(NMS-free格式)

  using base::Param::serialize;
  virtual base::Status serialize(rapidjson::Value &json,
                                 rapidjson::Document::AllocatorType &allocator);
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value &json);
};

/**
 * @brief 旋转框检测(OBB)后处理节点
 *
 * 将模型推理输出的 Tensor 转换为 ObbResult。
 * v8/v11: 密集预测格式，逐行解码 angle 和 class scores
 * v26: NMS-free 格式，直接读取 score/class_id/angle
 */
class NNDEPLOY_CC_API ObbPostProcess : public dag::Node {
 public:
  ObbPostProcess(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::detect::ObbPostProcess";
    desc_ = "YOLOv8/11/26-Obb postprocess[device::Tensor->ObbResult]";
    param_ = std::make_shared<ObbPostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<ObbResult>();
  }
  ObbPostProcess(const std::string &name, std::vector<dag::Edge *> inputs,
                  std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::detect::ObbPostProcess";
    desc_ = "YOLOv8/11/26-Obb postprocess[device::Tensor->ObbResult]";
    param_ = std::make_shared<ObbPostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<ObbResult>();
  }
  virtual ~ObbPostProcess() {}

  virtual base::Status run();
};

/**
 * @brief 旋转框检测完整工作流图
 *
 * 封装了预处理(CvtResizeNormTrans) -> 推理(Infer) -> 后处理(ObbPostProcess)
 * 的完整 YOLOv8/11/26-Obb 旋转框检测流水线。
 */
class NNDEPLOY_CC_API ObbGraph : public dag::Graph {
 public:
  ObbGraph(const std::string &name) : dag::Graph(name) {
    key_ = "nndeploy::detect::ObbGraph";
    desc_ = "yolov8/11/26-obb graph[cv::Mat->preprocess->infer->postprocess->ObbResult]";
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<ObbResult>();
    pre_ = dynamic_cast<preprocess::CvtResizeNormTrans *>(
        this->createNode<preprocess::CvtResizeNormTrans>("preprocess"));
    infer_ =
        dynamic_cast<infer::Infer *>(this->createNode<infer::Infer>("infer"));
    post_ = dynamic_cast<ObbPostProcess *>(
        this->createNode<ObbPostProcess>("postprocess"));
  }

  ObbGraph(const std::string &name, std::vector<dag::Edge *> inputs,
            std::vector<dag::Edge *> outputs)
      : dag::Graph(name, inputs, outputs) {
    key_ = "nndeploy::detect::ObbGraph";
    desc_ = "yolov8/11/26-obb graph[cv::Mat->preprocess->infer->postprocess->ObbResult]";
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<ObbResult>();
    pre_ = dynamic_cast<preprocess::CvtResizeNormTrans *>(
        this->createNode<preprocess::CvtResizeNormTrans>("preprocess"));
    infer_ =
        dynamic_cast<infer::Infer *>(this->createNode<infer::Infer>("infer"));
    post_ = dynamic_cast<ObbPostProcess *>(
        this->createNode<ObbPostProcess>("postprocess"));
  }

  virtual ~ObbGraph() {}

  virtual base::Status defaultParam() {
    preprocess::CvtResizeNormTransParam *pre_param =
        dynamic_cast<preprocess::CvtResizeNormTransParam *>(pre_->getParam());
    pre_param->src_pixel_type_ = base::kPixelTypeBGR;
    pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
    pre_param->interp_type_ = base::kInterpTypeLinear;
    pre_param->h_ = 1024;
    pre_param->w_ = 1024;

    ObbPostParam *post_param =
        dynamic_cast<ObbPostParam *>(post_->getParam());
    post_param->score_threshold_ = 0.5;
    post_param->nms_threshold_ = 0.45;
    post_param->num_classes_ = 15;
    post_param->model_h_ = 1024;
    post_param->model_w_ = 1024;
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
    std::vector<dag::Edge *> infer_outputs = (*infer_)(pre_outputs);
    std::vector<dag::Edge *> post_outputs = (*post_)(infer_outputs);
    return post_outputs;
  }

 private:
  dag::Node *pre_ = nullptr;
  infer::Infer *infer_ = nullptr;
  dag::Node *post_ = nullptr;
};

}  // namespace detect
}  // namespace nndeploy

#endif /* _NNDEPLOY_DETECT_YOLO_OBB_YOLO_OBB_H_ */
