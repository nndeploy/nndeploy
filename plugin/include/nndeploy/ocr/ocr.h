#ifndef _NNDEPLOY_OCR_OCR_H_
#define _NNDEPLOY_OCR_OCR_H_

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
#include "nndeploy/ocr/ocr_postprocess_op.h"
#include "nndeploy/ocr/result.h"
#include "nndeploy/preprocess/params.h"

namespace nndeploy {
namespace ocr {

class NNDEPLOY_CC_API RotateCropImage : public dag::Node {
 public:
  RotateCropImage(const std::string &name) : Node(name) {
    key_ = "nndeploy::ocr::RotateCropImage";
    desc_ = "RotateCropImage";
    this->setInputTypeInfo<OCRResult>();
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<OCRResult>();
  }
  RotateCropImage(const std::string &name, std::vector<dag::Edge *> inputs,
                  std::vector<dag::Edge *> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::ocr::RotateCropImage";
    desc_ = "RotateCropImage";
    this->setInputTypeInfo<OCRResult>();
    this->setInputTypeInfo<cv::Mat>();
    this->setOutputTypeInfo<OCRResult>();
  }
  PostProcessor util_post_processor_;
  virtual ~RotateCropImage(){};

  virtual base::Status run();
};

class NNDEPLOY_CC_API RotateImage180 : public dag::Node {
 public:
  RotateImage180(const std::string &name) : Node(name) {
    key_ = "nndeploy::ocr::RotateImage180";
    desc_ = "RotateImage180";
    this->setInputTypeInfo<OCRResult>();
    this->setInputTypeInfo<OCRResult>();
    this->setOutputTypeInfo<OCRResult>();
  }
  RotateImage180(const std::string &name, std::vector<dag::Edge *> inputs,
                 std::vector<dag::Edge *> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::ocr::RotateImage180";
    desc_ = "RotateImage180";
    this->setInputTypeInfo<OCRResult>();
    this->setInputTypeInfo<OCRResult>();
    this->setOutputTypeInfo<OCRResult>();
  }
  PostProcessor util_post_processor_;
  virtual ~RotateImage180(){};

  virtual base::Status run();
};

class NNDEPLOY_CC_API OcrText : public base::Param {
 public:
  std::vector<std::string> texts_;

  using base::Param::serialize;
  virtual base::Status serialize(
      rapidjson::Value &json,
      rapidjson::Document::AllocatorType &allocator) override {
    rapidjson::Value texts_json(rapidjson::kArrayType);
    for (const auto &text : texts_) {
      texts_json.PushBack(rapidjson::Value(text.c_str(), allocator), allocator);
    }
    json.AddMember("texts_", texts_json, allocator);
    return base::kStatusCodeOk;
  }

  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value &json) override {
    if (json.HasMember("texts_") && json["texts_"].IsArray()) {
      texts_.clear();
      for (const auto &text : json["texts_"].GetArray()) {
        texts_.push_back(text.GetString());
      }
    }
    return base::kStatusCodeOk;
  }
};

class NNDEPLOY_CC_API PrintOcrNodeParam : public base::Param {
 public:
  std::string path_ = "resources/others/ocr_out.txt";

  using base::Param::serialize;
  virtual base::Status serialize(rapidjson::Value &json,
                                 rapidjson::Document::AllocatorType &allocator);
  using base::Param::deserialize;
  virtual base::Status deserialize(rapidjson::Value &json);
};

class NNDEPLOY_CC_API PrintOcrNode : public dag::Node {
 public:
  PrintOcrNode(const std::string &name, std::vector<dag::Edge *> inputs,
               std::vector<dag::Edge *> outputs)
      : Node(name, inputs, outputs) {
    key_ = "nndeploy::ocr::PrintOcrNode";
    desc_ = "Print Text";
    param_ = std::make_shared<PrintOcrNodeParam>();
    this->setInputTypeInfo<OCRResult>();
    this->setNodeType(dag::NodeType::kNodeTypeOutput);
    this->setIoType(dag::IOType::kIOTypeText);
  }

  virtual ~PrintOcrNode() {}
  base::Status setPath(const std::string &path) {
    if (path.empty()) {
      return base::kStatusCodeErrorInvalidParam;
    }
    auto param = dynamic_cast<PrintOcrNodeParam *>(getParam());
    param->path_ = path;
    return base::kStatusCodeOk;
  }
  virtual base::Status run();
};

}  // namespace ocr
}  // namespace nndeploy

#endif