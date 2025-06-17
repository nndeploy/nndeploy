#ifndef _NNDEPLOY_CODEC_BATCH_CODEC_H_
#define _NNDEPLOY_CODEC_BATCH_CODEC_H_

#include "nndeploy/base/any.h"
#include "nndeploy/base/common.h"
#include "nndeploy/base/glic_stl_include.h"
#include "nndeploy/base/log.h"
#include "nndeploy/base/macro.h"
#include "nndeploy/base/object.h"
#include "nndeploy/base/opencv_include.h"
#include "nndeploy/base/status.h"
#include "nndeploy/base/string.h"
#include "nndeploy/codec/codec.h"
#include "nndeploy/dag/composite_node.h"
#include "nndeploy/dag/edge.h"
#include "nndeploy/dag/node.h"
#include "nndeploy/device/buffer.h"
#include "nndeploy/device/device.h"
#include "nndeploy/device/memory_pool.h"
#include "nndeploy/device/tensor.h"

namespace nndeploy {
namespace codec {

class NNDEPLOY_CC_API BatchOpenCvDecode : public dag::CompositeNode {
 public:
  BatchOpenCvDecode(const std::string &name) : dag::CompositeNode(name) {
    key_ = "nndeploy::codec::BatchOpenCvDecode";
    node_type_ = dag::NodeType::kNodeTypeInput;
    this->setOutputTypeInfo<std::vector<cv::Mat>>();
  }
  BatchOpenCvDecode(const std::string &name, std::vector<dag::Edge *> inputs,
                    std::vector<dag::Edge *> outputs)
      : dag::CompositeNode(name, inputs, outputs) {
    key_ = "nndeploy::codec::BatchOpenCvDecode";
    node_type_ = dag::NodeType::kNodeTypeInput;
    this->setOutputTypeInfo<std::vector<cv::Mat>>();
  }
  virtual ~BatchOpenCvDecode() {
    if (this->getInitialized()) {
      this->deinit();
      this->setInitializedFlag(false);
    }
  }

  void setBatchSize(int batch_size) { batch_size_ = batch_size; }
  base::Status setNodeKey(const std::string &key) {
    node_key_ = key;
    std::vector<std::string> input_names = this->getInputNames();
    std::vector<std::string> output_names = this->getRealOutputsName();
    dag::NodeDesc desc(node_key_, "inner_decode_node", input_names,
                       output_names);
    node_ = (Decode *)this->createNode(desc);
    if (!node_) {
      NNDEPLOY_LOGE("Node creation failed for node_key: %s\n",
                    node_key_.c_str());
      return base::kStatusCodeErrorInvalidParam;
    }
    return base::kStatusCodeOk;
  }

  base::Status setCodecFlag(base::CodecFlag flag) {
    if (node_) {
      return node_->setCodecFlag(flag);
    }
    return base::kStatusCodeErrorNullParam;
  }
  base::CodecFlag getCodecFlag() {
    if (node_) {
      return node_->getCodecFlag();
    }
    return base::kCodecFlagImage;  // 默认值
  }
  void setPath(const std::string &path) {
    if (node_) {
      node_->setPath(path);
    }
  }

  void setSize(int size) {
    if (node_) {
      node_->setSize(size);
    }
  }
  int getSize() {
    if (node_) {
      size_ = NNDEPLOY_UP_DIV((int)(node_->getSize()), batch_size_);
    }
    return size_;
  }

  double getFps() {
    if (node_) {
      return node_->getFps();
    }
    return 0.0;  // 默认值
  }
  int getWidth() {
    if (node_) {
      return node_->getWidth();
    }
    return 0;  // 默认值
  }
  int getHeight() {
    if (node_) {
      return node_->getHeight();
    }
    return 0;  // 默认值
  }

  virtual base::EdgeUpdateFlag updateInput() {
    if (index_ < size_) {
      return base::kEdgeUpdateFlagComplete;
    } else {
      if (size_ == 0) {
        return base::kEdgeUpdateFlagComplete;
      } else {
        return base::kEdgeUpdateFlagTerminate;
      }
    }
  }

  // virtual base::Status make() { return base::kStatusCodeOk; }

  // virtual base::Status init() {
  //   index_ = 0;
  //   if (node_) {
  //     base::Status status = node_->init();
  //     if (status != base::kStatusCodeOk) {
  //       NNDEPLOY_LOGE("node init failed");
  //       return status;
  //     }
  //     node_->setInitializedFlag(true);
  //   }
  //   return base::kStatusCodeErrorNullParam;
  // }

  // virtual base::Status deinit() {
  //   if (node_) {
  //     if (node_->getInitialized()) {
  //       base::Status status = node_->deinit();
  //       if (status != base::kStatusCodeOk) {
  //         NNDEPLOY_LOGE("node deinit failed");
  //         return status;
  //       }
  //       node_->setInitializedFlag(false);
  //     }
  //   }
  //   return base::kStatusCodeErrorNullParam;
  // }

  virtual base::Status run() {
    auto results = new std::vector<cv::Mat>();
    if (index_ >= size_) {
      outputs_[0]->setAny(results, false);
      return base::kStatusCodeOk;
    }
    for (int i = 0; i < batch_size_; i++) {
      // NNDEPLOY_LOGI("index_: %d, batch_size_: %d\n", index_, batch_size_);
      cv::Mat *single = nullptr;
      if (index_ * batch_size_ + i < node_->getSize()) {
        node_->run();
        dag::Edge *output = node_->getOutput();
        single = output->getCvMat(node_);
        if (single == nullptr) {
          NNDEPLOY_LOGE("single_tensor is nullptr");
          return base::kStatusCodeErrorInvalidParam;
        }
      }
      if (single != nullptr && !single->empty()) {
        // NNDEPLOY_LOGE("HW: %d, %d", single->rows, single->cols);
        cv::Mat res(single->rows, single->cols, single->type());
        single->copyTo(res);
        results->push_back(res);
      }
    }
    outputs_[0]->setAny(results, false);
    index_++;
    return base::kStatusCodeOk;
  }

  virtual base::Status serialize(
      rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
    base::Status status = dag::CompositeNode::serialize(json, allocator);
    if (status != base::kStatusCodeOk) {
      return status;
    }
    json.AddMember("batch_size_", batch_size_, allocator);
    json.AddMember(
        "node_key_",
        rapidjson::Value(node_key_.c_str(), node_key_.length(), allocator),
        allocator);
    return base::kStatusCodeOk;
  }
  // virtual std::string serialize() {
  //   rapidjson::Document doc;
  //   doc.SetObject();
  //   this->serialize(doc, doc.GetAllocator());
  //   rapidjson::StringBuffer buffer;
  //   rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  //   doc.Accept(writer);
  //   std::string json_str = buffer.GetString();
  //   if (node_ == nullptr) {
  //     return json_str;
  //   }
  //   json_str[json_str.length() - 1] = ',';
  //   json_str += "\"node_\": ";
  //   json_str += node_->serialize();
  //   json_str += "}";
  //   return json_str;
  // }
  virtual base::Status deserialize(rapidjson::Value &json) {
    base::Status status = dag::CompositeNode::deserialize(json);
    if (status != base::kStatusCodeOk) {
      return status;
    }
    if (json.HasMember("batch_size_") && json["batch_size_"].IsInt()) {
      batch_size_ = json["batch_size_"].GetInt();
    }
    if (json.HasMember("node_key_") && json["node_key_"].IsString()) {
      std::string node_key_str = json["node_key_"].GetString();
      this->setNodeKey(node_key_str);
    }
    return base::kStatusCodeOk;
  }

  // virtual base::Status deserialize(const std::string &json_str) {
  //   rapidjson::Document document;
  //   if (document.Parse(json_str.c_str()).HasParseError()) {
  //     NNDEPLOY_LOGE("parse json string failed\n");
  //     return base::kStatusCodeErrorInvalidParam;
  //   }
  //   rapidjson::Value &json = document;
  //   base::Status status = this->deserialize(json);
  //   if (status != base::kStatusCodeOk) {
  //     NNDEPLOY_LOGE("deserialize failed\n");
  //     return status;
  //   }
  //   if (json.HasMember("node_") && json["node_"].IsObject()) {
  //     rapidjson::Value &node_json = json["node_"];
  //     rapidjson::StringBuffer buffer;
  //     rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  //     node_json.Accept(writer);
  //     std::string node_json_str = buffer.GetString();
  //     base::Status status = node_->deserialize(node_json_str);
  //     if (status != base::kStatusCodeOk) {
  //       NNDEPLOY_LOGE("deserialize node failed\n");
  //       return status;
  //     }
  //     // node_ = node;
  //   }
  //   return base::kStatusCodeOk;
  // }

 private:
  int batch_size_ = 1;
  int index_ = 0;
  int size_ = 0;
  std::string node_key_ = "";
  Decode *node_ = nullptr;
};

class NNDEPLOY_CC_API BatchOpenCvEncode : public dag::CompositeNode {
 public:
  BatchOpenCvEncode(const std::string &name) : dag::CompositeNode(name) {
    key_ = "nndeploy::codec::BatchOpenCvEncode";
    node_type_ = dag::NodeType::kNodeTypeOutput;
    this->setInputTypeInfo<std::vector<cv::Mat>>();
  }
  BatchOpenCvEncode(const std::string &name, std::vector<dag::Edge *> inputs,
                    std::vector<dag::Edge *> outputs)
      : dag::CompositeNode(name, inputs, outputs) {
    key_ = "nndeploy::codec::BatchOpenCvEncode";
    node_type_ = dag::NodeType::kNodeTypeOutput;
    this->setInputTypeInfo<std::vector<cv::Mat>>();
  }
  virtual ~BatchOpenCvEncode() {
    if (this->getInitialized()) {
      this->deinit();
      this->setInitializedFlag(false);
    }
  }

  base::Status setNodeKey(const std::string &key) {
    node_key_ = key;
    dag::NodeDesc desc(node_key_, "inner_codec_node",
                       {"inner_codec_node.input"}, {});
    node_ = (EncodeNode *)this->createNode(desc);
    if (!node_) {
      NNDEPLOY_LOGE("Node creation failed for node_key: %s\n",
                    node_key_.c_str());
      return base::kStatusCodeErrorInvalidParam;
    }
    return base::kStatusCodeOk;
  }

  base::Status setCodecFlag(base::CodecFlag flag) {
    if (node_) {
      return node_->setCodecFlag(flag);
    }
    return base::kStatusCodeErrorNullParam;
  }
  base::CodecFlag getCodecFlag() {
    if (node_) {
      return node_->getCodecFlag();
    }
    return base::kCodecFlagImage;  // 默认值
  }
  void setPath(const std::string &path) {
    if (node_) {
      node_->setPath(path);
    }
  }
  void setRefPath(const std::string &ref_path) {
    if (node_) {
      node_->setRefPath(ref_path);
    }
  }
  void setFourcc(const std::string &fourcc) {
    if (node_) {
      node_->setFourcc(fourcc);
    }
  }
  void setFps(double fps) {
    if (node_) {
      node_->setFps(fps);
    }
  }
  void setWidth(int width) {
    if (node_) {
      node_->setWidth(width);
    }
  }
  void setHeight(int height) {
    if (node_) {
      node_->setHeight(height);
    }
  }
  void setSize(int size) {
    if (node_) {
      node_->setSize(size);
    }
  }
  int getSize() {
    if (node_) {
      return node_->getSize();
    }
    return 0;  // 默认值
  }
  int getIndex() {
    if (node_) {
      return node_->getIndex();
    }
    return 0;  // 默认值
  }

  // virtual base::Status make() { return base::kStatusCodeOk; }

  // virtual base::Status init() {
  //   if (node_) {
  //     base::Status status = node_->init();
  //     if (status != base::kStatusCodeOk) {
  //       NNDEPLOY_LOGE("node init failed");
  //       return status;
  //     }
  //     node_->setInitializedFlag(true);
  //   }
  //   return base::kStatusCodeErrorNullParam;
  // }

  // virtual base::Status deinit() {
  //   if (node_) {
  //     if (node_->getInitialized()) {
  //       // NNDEPLOY_LOGE("node[%s] deinit\n", node_->getName().c_str());
  //       base::Status status = node_->deinit();
  //       if (status != base::kStatusCodeOk) {
  //         NNDEPLOY_LOGE("node deinit failed");
  //         return status;
  //       }
  //       node_->setInitializedFlag(false);
  //     }
  //   }
  //   return base::kStatusCodeErrorNullParam;
  // }

  virtual base::Status run() {
    std::vector<cv::Mat> *cv_mats =
        inputs_[0]->getAny<std::vector<cv::Mat>>(this);
    if (cv_mats == nullptr) {
      NNDEPLOY_LOGE("cv_mats is nullptr");
      return base::kStatusCodeErrorInvalidParam;
    }
    dag::Edge *node_input = node_->getInput();
    for (int i = 0; i < cv_mats->size(); i++) {
      node_input->set((*cv_mats)[i]);
      node_->run();
    }
    return base::kStatusCodeOk;
  }

  virtual base::Status serialize(
      rapidjson::Value &json, rapidjson::Document::AllocatorType &allocator) {
    base::Status status = dag::CompositeNode::serialize(json, allocator);
    if (status != base::kStatusCodeOk) {
      return status;
    }
    json.AddMember(
        "node_key_",
        rapidjson::Value(node_key_.c_str(), node_key_.length(), allocator),
        allocator);
    return base::kStatusCodeOk;
  }
  // virtual std::string serialize() {
  //   rapidjson::Document doc;
  //   doc.SetObject();
  //   this->serialize(doc, doc.GetAllocator());
  //   rapidjson::StringBuffer buffer;
  //   rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  //   doc.Accept(writer);
  //   std::string json_str = buffer.GetString();
  //   if (node_ == nullptr) {
  //     return json_str;
  //   }
  //   json_str[json_str.length() - 1] = ',';
  //   json_str += "\"node_\": ";
  //   json_str += node_->serialize();
  //   json_str += "}";
  //   return json_str;
  // }
  virtual base::Status deserialize(rapidjson::Value &json) {
    base::Status status = dag::CompositeNode::deserialize(json);
    if (status != base::kStatusCodeOk) {
      return status;
    }
    if (json.HasMember("node_key_") && json["node_key_"].IsString()) {
      std::string node_key_str = json["node_key_"].GetString();
      this->setNodeKey(node_key_str);
    }
    return base::kStatusCodeOk;
  }

  // virtual base::Status deserialize(const std::string &json_str) {
  //   rapidjson::Document document;
  //   if (document.Parse(json_str.c_str()).HasParseError()) {
  //     NNDEPLOY_LOGE("parse json string failed\n");
  //     return base::kStatusCodeErrorInvalidParam;
  //   }
  //   rapidjson::Value &json = document;
  //   base::Status status = this->deserialize(json);
  //   if (status != base::kStatusCodeOk) {
  //     NNDEPLOY_LOGE("deserialize failed\n");
  //     return status;
  //   }
  //   if (json.HasMember("node_") && json["node_"].IsObject()) {
  //     rapidjson::Value &node_json = json["node_"];
  //     rapidjson::StringBuffer buffer;
  //     rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  //     node_json.Accept(writer);
  //     std::string node_json_str = buffer.GetString();
  //     base::Status status = node_->deserialize(node_json_str);
  //     if (status != base::kStatusCodeOk) {
  //       NNDEPLOY_LOGE("deserialize node failed\n");
  //       return status;
  //     }
  //     // node_ = node;
  //   }
  //   return base::kStatusCodeOk;
  // }

 private:
  std::string node_key_ = "";
  EncodeNode *node_ = nullptr;
};

}  // namespace codec
}  // namespace nndeploy

#endif /* _NNDEPLOY_CODEC_BATCH_CODEC_H_ */
