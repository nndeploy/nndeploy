#ifndef _NNDEPLOY_TRACK_FAIRMOT_FAIRMOT_H_
#define _NNDEPLOY_TRACK_FAIRMOT_FAIRMOT_H_

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
#include "nndeploy/preprocess/cvt_resize_norm_trans.h"
#include "nndeploy/preprocess/params.h"
#include "nndeploy/track/result.h"
#include "nndeploy/track/tracker.h"

namespace nndeploy {
namespace track {

class NNDEPLOY_CC_API FairMotPreParam : public base::Param {
 public:
  // 源图像的像素类型
  base::PixelType src_pixel_type_;
  // 目标图像的像素类型
  base::PixelType dst_pixel_type_;
  // 图像缩放时使用的插值类型
  base::InterpType interp_type_;
  // 目标输出的高度
  int h_ = -1;
  // 目标输出的宽度
  int w_ = -1;
  // 数据类型，默认为浮点型
  base::DataType data_type_ = base::dataTypeOf<float>();
  // 数据格式，默认为NCHW（通道数，图像高度，图像宽度）
  base::DataFormat data_format_ = base::DataFormat::kDataFormatNCHW;
  // 是否进行归一化处理
  bool normalize_ = true;
  // 归一化的比例因子，用于将像素值缩放到0-1范围
  float scale_[4] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f,
                     1.0f / 255.0f};
  // 归一化处理中的均值，用于数据中心化
  float mean_[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  // 归一化处理中的标准差，用于数据标准化
  float std_[4] = {1.0f, 1.0f, 1.0f, 1.0f};

  using base::Param::serialize;
  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json);
};

class NNDEPLOY_CC_API FairMotPostParam : public base::Param {
 public:
  float conf_thresh_ = 0.4f;
  float tracked_thresh_ = 0.4f;
  float min_box_area_ = 200.0f;

  using base::Param::serialize;
  virtual base::Status serialize(rapidjson::Value& json,
                                 rapidjson::Document::AllocatorType& allocator);
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value& json);
};

class NNDEPLOY_CC_API FairMotPreProcess : public dag::Node {
 public:
  FairMotPreProcess(const std::string& name) : dag::Node(name) {
    key_ = "nndeploy::track::FairMotPreProcess";
    desc_ = "FairMot preprocess[cv::Mat->device::Tensor]";
    param_ = std::make_shared<FairMotPreParam>();
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
  }
  FairMotPreProcess(const std::string& name, std::vector<dag::Edge*> inputs,
                    std::vector<dag::Edge*> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::track::FairMotPreProcess";
    desc_ = "FairMot preprocess[cv::Mat->device::Tensor]";
    param_ = std::make_shared<FairMotPreParam>();
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<device::Tensor>();
  }

  virtual ~FairMotPreProcess() {}

  virtual base::Status run();
};

class NNDEPLOY_CC_API FairMotPostProcess : public dag::Node {
 public:
  FairMotPostProcess(const std::string& name) : dag::Node(name) {
    key_ = "nndeploy::track::FairMotPostProcess";
    desc_ = "FairMot postprocess[device::Tensor->MOTResult]";
    param_ = std::make_shared<FairMotPostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<MOTResult>();
  }
  FairMotPostProcess(const std::string& name, std::vector<dag::Edge*> inputs,
                     std::vector<dag::Edge*> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::track::FairMotPostProcess";
    desc_ = "FairMot postprocess[device::Tensor->MOTResult]";
    param_ = std::make_shared<FairMotPostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<MOTResult>();
  }

  virtual ~FairMotPostProcess() {}

  virtual base::Status init();

  virtual base::Status deinit();

  virtual base::Status run();

  void FilterDets(const float conf_threshold, const cv::Mat& dets,
                  std::vector<int>* index);

 private:
  std::shared_ptr<JDETracker> jdeTracker_ = nullptr;
};

class NNDEPLOY_CC_API FairMotGraph : public dag::Graph {
 public:
  FairMotGraph(const std::string& name) : dag::Graph(name) {
    key_ = "nndeploy::track::FairMotGraph";
    desc_ =
        "FairMot "
        "graph[cv::Mat->preprocess->infer->postprocess->MOTResult]";
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<MOTResult>();
    pre_ = this->createNode<FairMotPreProcess>("preprocess");
    if (pre_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create fairmot preprocess node\n");
      constructed_ = false;
      return;
    }
    infer_ =
        dynamic_cast<infer::Infer*>(this->createNode<infer::Infer>("infer"));
    if (infer_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create inference node\n");
      constructed_ = false;
      return;
    }
    post_ = this->createNode<FairMotPostProcess>("post");
    if (post_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create postprocessing node\n");
      constructed_ = false;
      return;
    }
  }

  FairMotGraph(const std::string& name, std::vector<dag::Edge*> inputs,
               std::vector<dag::Edge*> outputs)
      : dag::Graph(name, inputs, outputs) {
    key_ = "nndeploy::track::FairMotGraph";
    desc_ =
        "FairMot "
        "graph[cv::Mat->preprocess->infer->postprocess->MOTResult]";
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<MOTResult>();
    pre_ = this->createNode<FairMotPreProcess>("preprocess");
    if (pre_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create fairmot preprocess node\n");
      constructed_ = false;
      return;
    }
    infer_ =
        dynamic_cast<infer::Infer*>(this->createNode<infer::Infer>("infer"));
    if (infer_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create inference node\n");
      constructed_ = false;
      return;
    }
    post_ = this->createNode<FairMotPostProcess>("post");
    if (post_ == nullptr) {
      NNDEPLOY_LOGE("Failed to create postprocessing node\n");
      constructed_ = false;
      return;
    }
  }

  virtual ~FairMotGraph() {}

  base::Status defaultParam() {
    FairMotPreParam* pre_param =
        dynamic_cast<FairMotPreParam*>(pre_->getParam());
    pre_param->src_pixel_type_ = base::kPixelTypeBGR;
    pre_param->dst_pixel_type_ = base::kPixelTypeRGB;
    pre_param->interp_type_ = base::kInterpTypeLinear;
    pre_param->h_ = 320;
    pre_param->w_ = 576;
    FairMotPostParam* post_param =
        dynamic_cast<FairMotPostParam*>(post_->getParam());

    return base::kStatusCodeOk;
  }

  base::Status make(const dag::NodeDesc& pre_desc,
                    const dag::NodeDesc& infer_desc,
                    base::InferenceType inference_type,
                    const dag::NodeDesc& post_desc) {
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

  base::Status setInferParam(base::DeviceType device_type,
                             base::ModelType model_type, bool is_path,
                             std::vector<std::string>& model_value) {
    auto param = dynamic_cast<inference::InferenceParam*>(infer_->getParam());
    param->device_type_ = device_type;
    param->model_type_ = model_type;
    param->is_path_ = is_path;
    param->model_value_ = model_value;
    return base::kStatusCodeOk;
  }

 private:
  dag::Node* pre_ = nullptr;
  infer::Infer* infer_ = nullptr;
  dag::Node* post_ = nullptr;
};

}  // namespace track
}  // namespace nndeploy

#endif