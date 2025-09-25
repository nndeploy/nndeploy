#ifndef _NNDEPLOY_DETECT_CLASSIFIER_OCR_OCR_H_
#define _NNDEPLOY_DETECT_CLASSIFIER_OCR_OCR_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
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
#include "nndeploy/ocr/ocr_postprocess_op.h"
#include "nndeploy/ocr/result.h"
#include "nndeploy/preprocess/opencv_convert.h"
#include "nndeploy/preprocess/params.h"

namespace nndeploy {
namespace ocr {

class NNDEPLOY_CC_API ClassifierParam : public base::Param {
 public:
  int version_ = -1;
  using base::Param::serialize;
  // virtual base::Status serialize(rapidjson::Value &json,
  //  rapidjson::Document::AllocatorType &allocator);
  using base::Param::deserialize;
  // virtual base::Status deserialize(rapidjson::Value &json);
};

class NNDEPLOY_CC_API ClassifierPreProcessParam : public base::Param {
 public:
  base::PixelType src_pixel_type_ = base::kPixelTypeBGR;
  base::PixelType dst_pixel_type_ = base::kPixelTypeBGR;
  base::InterpType interp_type_ = base::kInterpTypeLinear;
  base::DataType data_type_ = base::dataTypeOf<float>();
  base::DataFormat data_format_ = base::DataFormat::kDataFormatNCHW;
  std::vector<int> cls_image_shape_ = {3, 48, 192};
  int h_ = -1;
  int w_ = -1;
  bool normalize_ = true;
  float scale_[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  float mean_[3] = {0.0f, 0.0f, 0.0f};
  float std_[3] = {1.0f, 1.0f, 1.0f};

  base::BorderType border_type_ = base::kBorderTypeConstant;
  int top_ = 0;
  int bottom_ = 0;
  int left_ = 0;
  int right_ = 0;
  base::Scalar2d border_val_ = 0.0;

  using base::Param::serialize;
  virtual base::Status serialize(
      rapidjson::Value &json,
      rapidjson::Document::AllocatorType &allocator) override {
    std::string src_pixel_type_str = base::pixelTypeToString(src_pixel_type_);
    json.AddMember("src_pixel_type_",
                   rapidjson::Value(src_pixel_type_str.c_str(), allocator),
                   allocator);
    std::string dst_pixel_type_str = base::pixelTypeToString(dst_pixel_type_);
    json.AddMember("dst_pixel_type_",
                   rapidjson::Value(dst_pixel_type_str.c_str(), allocator),
                   allocator);
    std::string interp_type_str = base::interpTypeToString(interp_type_);
    json.AddMember("interp_type_",
                   rapidjson::Value(interp_type_str.c_str(), allocator),
                   allocator);
    std::string data_type_str = base::dataTypeToString(data_type_);
    json.AddMember("data_type_",
                   rapidjson::Value(data_type_str.c_str(), allocator),
                   allocator);
    std::string data_format_str = base::dataFormatToString(data_format_);
    json.AddMember("data_format_",
                   rapidjson::Value(data_format_str.c_str(), allocator),
                   allocator);
    json.AddMember("h_", h_, allocator);
    json.AddMember("w_", w_, allocator);
    json.AddMember("normalize_", normalize_, allocator);

    rapidjson::Value cls_image_shape(rapidjson::kArrayType);
    for (int i = 0; i < 3; i++) {
      cls_image_shape.PushBack(cls_image_shape_[i], allocator);
    }
    json.AddMember("cls_image_shape_", cls_image_shape, allocator);

    rapidjson::Value scale_array(rapidjson::kArrayType);
    rapidjson::Value mean_array(rapidjson::kArrayType);
    rapidjson::Value std_array(rapidjson::kArrayType);
    for (int i = 0; i < 3; i++) {
      scale_array.PushBack(scale_[i], allocator);
      mean_array.PushBack(mean_[i], allocator);
      std_array.PushBack(std_[i], allocator);
    }
    json.AddMember("scale_", scale_array, allocator);
    json.AddMember("mean_", mean_array, allocator);
    json.AddMember("std_", std_array, allocator);

    std::string border_type_str = base::borderTypeToString(border_type_);
    json.AddMember("border_type_",
                   rapidjson::Value(border_type_str.c_str(), allocator),
                   allocator);
    json.AddMember("top_", top_, allocator);
    json.AddMember("bottom_", bottom_, allocator);
    json.AddMember("left_", left_, allocator);
    json.AddMember("right_", right_, allocator);

    rapidjson::Value border_val_array(rapidjson::kArrayType);
    for (int i = 0; i < 4; i++) {
      border_val_array.PushBack(border_val_.val_[i], allocator);
    }
    json.AddMember("border_val_", border_val_array, allocator);

    return base::kStatusCodeOk;
  }

  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value &json) override {
    if (json.HasMember("src_pixel_type_") &&
        json["src_pixel_type_"].IsString()) {
      src_pixel_type_ =
          base::stringToPixelType(json["src_pixel_type_"].GetString());
    }
    if (json.HasMember("dst_pixel_type_") &&
        json["dst_pixel_type_"].IsString()) {
      dst_pixel_type_ =
          base::stringToPixelType(json["dst_pixel_type_"].GetString());
    }
    if (json.HasMember("interp_type_") && json["interp_type_"].IsString()) {
      interp_type_ = base::stringToInterpType(json["interp_type_"].GetString());
    }
    if (json.HasMember("data_type_") && json["data_type_"].IsString()) {
      data_type_ = base::stringToDataType(json["data_type_"].GetString());
    }
    if (json.HasMember("data_format_") && json["data_format_"].IsString()) {
      data_format_ = base::stringToDataFormat(json["data_format_"].GetString());
    }
    if (json.HasMember("h_") && json["h_"].IsInt()) {
      h_ = json["h_"].GetInt();
    }
    if (json.HasMember("w_") && json["w_"].IsInt()) {
      w_ = json["w_"].GetInt();
    }
    if (json.HasMember("normalize_") && json["normalize_"].IsBool()) {
      normalize_ = json["normalize_"].GetBool();
    }

    if (json.HasMember("cls_image_shape_") &&
        json["cls_image_shape_"].IsArray()) {
      const rapidjson::Value &cls_image_shape_array = json["cls_image_shape_"];
      for (int i = 0; i < 3 && i < cls_image_shape_array.Size(); i++) {
        if (cls_image_shape_array[i].IsInt()) {
          cls_image_shape_[i] = cls_image_shape_array[i].GetInt();
        }
      }
    }

    if (json.HasMember("scale_") && json["scale_"].IsArray()) {
      const rapidjson::Value &scale_array = json["scale_"];
      for (int i = 0; i < 3 && i < scale_array.Size(); i++) {
        if (scale_array[i].IsFloat()) {
          scale_[i] = scale_array[i].GetFloat();
        }
      }
    }
    if (json.HasMember("mean_") && json["mean_"].IsArray()) {
      const rapidjson::Value &mean_array = json["mean_"];
      for (int i = 0; i < 3 && i < mean_array.Size(); i++) {
        if (mean_array[i].IsFloat()) {
          mean_[i] = mean_array[i].GetFloat();
        }
      }
    }
    if (json.HasMember("std_") && json["std_"].IsArray()) {
      const rapidjson::Value &std_array = json["std_"];
      for (int i = 0; i < 3 && i < std_array.Size(); i++) {
        if (std_array[i].IsFloat()) {
          std_[i] = std_array[i].GetFloat();
        }
      }
    }

    if (json.HasMember("border_type_") && json["border_type_"].IsString()) {
      border_type_ = base::stringToBorderType(json["border_type_"].GetString());
    }
    if (json.HasMember("top_") && json["top_"].IsInt()) {
      top_ = json["top_"].GetInt();
    }
    if (json.HasMember("bottom_") && json["bottom_"].IsInt()) {
      bottom_ = json["bottom_"].GetInt();
    }
    if (json.HasMember("left_") && json["left_"].IsInt()) {
      left_ = json["left_"].GetInt();
    }
    if (json.HasMember("right_") && json["right_"].IsInt()) {
      right_ = json["right_"].GetInt();
    }

    if (json.HasMember("border_val_") && json["border_val_"].IsArray()) {
      const rapidjson::Value &border_val_array = json["border_val_"];
      for (int i = 0; i < 4 && i < border_val_array.Size(); i++) {
        if (border_val_array[i].IsFloat()) {
          border_val_.val_[i] = border_val_array[i].GetFloat();
        }
      }
    }

    return base::kStatusCodeOk;
  }
};

class NNDEPLOY_CC_API ClassifierPreProcess : public dag::Node {
 public:
  ClassifierPreProcess(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::ocr::ClassifierPreProcess";
    desc_ =
        "ocr classify preprocess cv::Mat to "
        "device::Tensor[resize->pad->normalize->transpose]";
    param_ = std::make_shared<ClassifierPreProcessParam>();
    this->setInputTypeInfo<OCRResult>();
    this->setOutputTypeInfo<device::Tensor>();
  }
  ClassifierPreProcess(const std::string &name, std::vector<dag::Edge *> inputs,
                       std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::ocr::ClassifierPreProcess";
    desc_ =
        "ocr classify preprocess cv::Mat to "
        "device::Tensor[resize->pad->normalize->transpose]";
    param_ = std::make_shared<ClassifierPreProcessParam>();
    this->setInputTypeInfo<OCRResult>();
    this->setOutputTypeInfo<device::Tensor>();
  }
  virtual ~ClassifierPreProcess() {}

  virtual base::Status run();
};

class NNDEPLOY_CC_API ClassifierPostParam : public base::Param {
 public:
  int version_ = 2;
  float cls_thresh_ = 0.9;

  using base::Param::serialize;
  virtual base::Status serialize(rapidjson::Value &json,
                                 rapidjson::Document::AllocatorType &allocator);
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value &json);
};

class NNDEPLOY_CC_API ClassifierPostProcess : public dag::Node {
 public:
  ClassifierPostProcess(const std::string &name) : dag::Node(name) {
    key_ = "nndeploy::ocr::ClassifierPostProcess";
    desc_ = "PPOcrClsv2 postprocess[device::Tensor->OcrResult]";
    param_ = std::make_shared<ClassifierPostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<OCRResult>();
  }
  ClassifierPostProcess(const std::string &name,
                        std::vector<dag::Edge *> inputs,
                        std::vector<dag::Edge *> outputs)
      : dag::Node(name, inputs, outputs) {
    key_ = "nndeploy::ocr::ClassifierPostProcess";
    desc_ = "PPOcrClsv2 postprocess[device::Tensor->OcrResult]";
    param_ = std::make_shared<ClassifierPostParam>();
    this->setInputTypeInfo<device::Tensor>();
    this->setOutputTypeInfo<OCRResult>();
  }
  virtual ~ClassifierPostProcess() {}
  PostProcessor util_post_processor_;
  virtual base::Status run();
};

// class NNDEPLOY_CC_API ClassifyBBoxResult : public base::Param {
//  public:
//   ClassifyBBoxResult(){};
//   virtual ~ClassifyBBoxResult() {
//     if (mask_ != nullptr) {
//       delete mask_;
//       mask_ = nullptr;
//     }
//   };
//   int index_;
//   int label_id_;
//   float score_;
//   std::array<float, 4> bbox_;  // xmin, ymin, xmax, ymax
//   device::Tensor *mask_ = nullptr;
// };

// class NNDEPLOY_CC_API ClassifyResult : public base::Param {
//  public:
//   ClassifyResult(){};
//   virtual ~ClassifyResult(){};
//   std::vector<ClassifyBBoxResult> bboxs_;
// };

class NNDEPLOY_CC_API ClassifierGraph : public dag::Graph {
 public:
  ClassifierGraph(const std::string &name) : dag::Graph(name) {
    key_ = "nndeploy::ocr::ClassifierGraph";
    desc_ =
        "PPOcrClsV2 graph[cv::Mat->preprocess->infer->postprocess->OcrResult]";
    this->setInputTypeInfo<OCRResult>();
    this->setOutputTypeInfo<OCRResult>();
    pre_ = dynamic_cast<ClassifierPreProcess *>(
        this->createNode<ClassifierPreProcess>("preprocess"));
    infer_ =
        dynamic_cast<infer::Infer *>(this->createNode<infer::Infer>("infer"));
    post_ = dynamic_cast<ClassifierPostProcess *>(
        this->createNode<ClassifierPostProcess>("postprocess"));
  }

  ClassifierGraph(const std::string &name, std::vector<dag::Edge *> inputs,
                  std::vector<dag::Edge *> outputs)
      : dag::Graph(name, inputs, outputs) {
    key_ = "nndeploy::ocr::ClassifierGraph";
    desc_ =
        "PPOcrClsV2 graph[cv::Mat->preprocess->infer->postprocess->OcrResult]";
    this->setInputTypeInfo<OCRResult>();
    this->setOutputTypeInfo<OCRResult>();
    pre_ = dynamic_cast<ClassifierPreProcess *>(
        this->createNode<ClassifierPreProcess>("preprocess"));
    infer_ =
        dynamic_cast<infer::Infer *>(this->createNode<infer::Infer>("infer"));
    post_ = dynamic_cast<ClassifierPostProcess *>(
        this->createNode<ClassifierPostProcess>("postprocess"));
  }

  virtual ~ClassifierGraph() {}

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

  virtual base::Status defaultParam() {
    ClassifierPreProcessParam *pre_param =
        dynamic_cast<ClassifierPreProcessParam *>(pre_->getParam());
    pre_param->src_pixel_type_ = base::kPixelTypeBGR;
    pre_param->dst_pixel_type_ = base::kPixelTypeBGR;
    pre_param->interp_type_ = base::kInterpTypeLinear;
    pre_param->cls_image_shape_ = {3, 48, 192};
    ClassifierPostParam *post_param =
        dynamic_cast<ClassifierPostParam *>(post_->getParam());
    post_param->cls_thresh_ = 0.6;
    post_param->version_ = 5;

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
    param->device_type_ = device_type;
    param->model_type_ = model_type;
    param->is_path_ = is_path;
    param->model_value_ = model_value;
    return base::kStatusCodeOk;
  }
  base::Status setVersion(int version) {
    ClassifierPostParam *param =
        dynamic_cast<ClassifierPostParam *>(post_->getParam());
    param->version_ = version;
    return base::kStatusCodeOk;
  }
  base::Status setClsThresh(float threshold) {
    ClassifierPostParam *param =
        dynamic_cast<ClassifierPostParam *>(post_->getParam());
    param->cls_thresh_ = threshold;
    return base::kStatusCodeOk;
  }
  base::Status setSrcPixelType(base::PixelType pixel_type) {
    ClassifierPreProcessParam *param =
        dynamic_cast<ClassifierPreProcessParam *>(pre_->getParam());
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
  dag::Node *pre_ = nullptr;       ///< Preprocessing node pointer
  infer::Infer *infer_ = nullptr;  ///< Inference node pointer
  dag::Node *post_ = nullptr;
};
}  // namespace ocr
}  // namespace nndeploy

#endif